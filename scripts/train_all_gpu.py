#!/usr/bin/env python3
"""
ICDAR 2026 DocVQA — One-Shot Training, Inference & Evaluation
==============================================================
Competition : https://rrc.cvc.uab.es/?ch=34
Paper       : https://arxiv.org/abs/2504.01234

This script runs the FULL pipeline in one command:
  1. Downloads DocVQA 2026 dataset from HuggingFace
  2. Fine-tunes a Vision-Language Model with QLoRA
  3. Runs inference on val/test split
  4. Evaluates with official ANLS metric (per-domain breakdown)
  5. Compares against competition leaderboard
  6. Generates competition-ready submission JSON

Usage:
    python scripts/train_all_gpu.py                    # Full pipeline (default: Qwen 7B)
    python scripts/train_all_gpu.py --skip-training    # Inference + eval only
    python scripts/train_all_gpu.py --epochs 3         # Custom epochs
    python scripts/train_all_gpu.py --model qwen-3b    # Smaller model

Models (select via --model):
    qwen-7b  : Qwen/Qwen2.5-VL-7B-Instruct  (default, ~10 GB VRAM with 4-bit)
    qwen-3b  : Qwen/Qwen2.5-VL-3B-Instruct  (~5 GB VRAM with 4-bit)
"""

import os
import sys
import json
import time
import torch
import logging
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("train_all")

# ═══════════════════════════════════════════════════════════════════════════
# MODEL REGISTRY
# ═══════════════════════════════════════════════════════════════════════════
MODEL_REGISTRY = {
    "qwen-7b": {
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "params": "7B",
        "category": "Up to 8B parameters",
    },
    "qwen-3b": {
        "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "params": "3B",
        "category": "Up to 8B parameters",
    },
}

# Competition leaderboard (as of 2026-03-20) for benchmarking
LEADERBOARD = [
    {"method": "Uni-Parser + Mix-MLLMs",     "category": ">35B",   "score": 0.5125},
    {"method": "Uni-Parser + Gemini-3.1-Pro", "category": ">35B",   "score": 0.4625},
    {"method": "Gemini-3.1-Pro (Baseline)",   "category": ">35B",   "score": 0.3750},
    {"method": "Gemini-3-Flash (Baseline)",   "category": ">35B",   "score": 0.3563},
    {"method": "Uni-Parser + Qwen3.5-27B",   "category": "8B-35B", "score": 0.2938},
    {"method": "GPT-5.2 (Baseline)",          "category": ">35B",   "score": 0.2688},
    {"method": "Uni-Parser + Qwen3.5-4B",    "category": "≤8B",    "score": 0.1875},
    {"method": "Qwen3-VL-32B-Thinking",       "category": "8B-35B", "score": 0.1438},
    {"method": "Qwen3-VL-8B-Thinking",        "category": "≤8B",    "score": 0.0938},
    {"method": "Florence-2 + QLoRA",          "category": "≤8B",    "score": 0.0125},
]


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 — DATA
# ═══════════════════════════════════════════════════════════════════════════
def download_dataset(split="val", cache_dir="data/raw"):
    """Download DocVQA 2026 dataset from HuggingFace."""
    from datasets import load_dataset, load_from_disk

    cache_path = Path(cache_dir) / split
    if cache_path.exists():
        logger.info(f"Loading cached {split} dataset from {cache_path}")
        return load_from_disk(str(cache_path))

    logger.info(f"Downloading DocVQA 2026 {split} split...")
    dataset = load_dataset("VLR-CVC/DocVQA-2026", split=split)
    dataset.save_to_disk(str(cache_path))
    logger.info(f"Saved {len(dataset)} samples to {cache_path}")
    return dataset


def prepare_training_data(dataset):
    """Convert HuggingFace dataset into (image, question, answer) triplets."""
    from src.data_loader import get_all_qa_pairs

    samples = []
    for item in dataset:
        for qa in get_all_qa_pairs(item):
            if qa["ground_truth"] is not None:
                samples.append({
                    "image": qa["pages"][0] if qa["pages"] else None,
                    "question": qa["question"],
                    "answer": qa["ground_truth"],
                    "question_id": qa["question_id"],
                    "domain": qa["doc_category"],
                    "doc_id": qa["doc_id"],
                })
    logger.info(f"Prepared {len(samples)} QA training pairs")
    return samples


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 — TRAINING (QLoRA)
# ═══════════════════════════════════════════════════════════════════════════
def train_model(model_key, samples, epochs=2, batch_size=1, lr=2e-4,
                output_dir="checkpoints"):
    """
    Fine-tune a VLM with QLoRA (4-bit quantization + LoRA adapters).
    Uses gradient checkpointing + 8-bit optimizer to minimize VRAM.
    """
    from transformers import (
        AutoProcessor, AutoModelForVision2Seq,
        BitsAndBytesConfig, TrainingArguments, Trainer,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model_info = MODEL_REGISTRY[model_key]
    model_id = model_info["model_id"]
    output_path = Path(output_dir) / model_key

    logger.info(f"═══ TRAINING: {model_id} ({model_info['params']}) ═══")
    logger.info(f"  Category : {model_info['category']}")
    logger.info(f"  Epochs   : {epochs}  |  Samples: {len(samples)}")

    # 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA on attention projections
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Official competition prompt
    from src.evaluation.eval_utils import get_evaluation_prompt
    system_prompt = get_evaluation_prompt()

    # Dataset wrapper
    class DocVQADataset(torch.utils.data.Dataset):
        def __init__(self, data, proc, sys_prompt):
            self.data = data
            self.processor = proc
            self.system_prompt = sys_prompt

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": item["image"]},
                    {"type": "text", "text": f"{self.system_prompt}\n\nQuestion: {item['question']}"},
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": f"FINAL ANSWER: {item['answer']}"},
                ]},
            ]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            try:
                from qwen_vl_utils import process_vision_info
                image_inputs, _ = process_vision_info(messages)
                inputs = self.processor(
                    text=[text], images=image_inputs,
                    padding="max_length", max_length=512,
                    truncation=True, return_tensors="pt",
                )
            except (ImportError, Exception):
                inputs = self.processor(
                    text=text, images=item["image"],
                    padding="max_length", max_length=512,
                    truncation=True, return_tensors="pt",
                )

            input_ids = inputs["input_ids"].squeeze(0)
            attention_mask = inputs["attention_mask"].squeeze(0)
            pixel_values = inputs.get("pixel_values")
            if pixel_values is not None and pixel_values.dim() > 3:
                pixel_values = pixel_values.squeeze(0)

            result = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": input_ids.clone(),
            }
            if pixel_values is not None:
                result["pixel_values"] = pixel_values
            return result

    train_dataset = DocVQADataset(samples, processor, system_prompt)

    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=8,
        learning_rate=lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        fp16=not (torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False),
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
        run_name=f"docvqa-{model_key}-{datetime.now().strftime('%Y%m%d_%H%M')}",
    )

    # Initialize WandB with entity/project from .env
    if os.environ.get("WANDB_API_KEY"):
        import wandb
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "docvqa2026"),
            entity=os.environ.get("WANDB_ENTITY"),
            name=training_args.run_name,
            config={"model": model_id, "epochs": epochs, "lr": lr, "lora_r": 16},
        )

    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)

    logger.info("Starting QLoRA fine-tuning...")
    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time
    logger.info(f"Training completed in {elapsed/60:.1f} minutes")

    adapter_path = output_path / "final_adapter"
    model.save_pretrained(str(adapter_path))
    processor.save_pretrained(str(adapter_path))
    logger.info(f"Adapter saved to {adapter_path}")
    return str(adapter_path)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 — INFERENCE
# ═══════════════════════════════════════════════════════════════════════════
def run_inference(model_key, adapter_path, samples, max_samples=None):
    """Run inference with the (optionally fine-tuned) model."""
    from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
    from src.reasoner.answer_formatter import extract_and_format_answer
    from src.evaluation.eval_utils import get_evaluation_prompt
    from tqdm import tqdm

    model_info = MODEL_REGISTRY[model_key]
    model_id = model_info["model_id"]
    logger.info(f"═══ INFERENCE: {model_id} ═══")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    if adapter_path and Path(adapter_path).exists():
        logger.info(f"Loading fine-tuned adapter from {adapter_path}")
        from peft import PeftModel
        base_model = AutoModelForVision2Seq.from_pretrained(
            model_id, quantization_config=bnb_config,
            device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16,
        )
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        logger.info("Running zero-shot inference (no adapter)")
        model = AutoModelForVision2Seq.from_pretrained(
            model_id, quantization_config=bnb_config,
            device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16,
        )
    model.eval()

    system_prompt = get_evaluation_prompt()
    if max_samples:
        samples = samples[:max_samples]

    predictions = []
    for item in tqdm(samples, desc="Inference"):
        messages = [{"role": "user", "content": [
            {"type": "image", "image": item["image"]},
            {"type": "text", "text": f"{system_prompt}\n\nQuestion: {item['question']}"},
        ]}]
        try:
            from qwen_vl_utils import process_vision_info
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(messages)
            inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt")
        except (ImportError, Exception):
            prompt = f"User: <image>\n{system_prompt}\n\nQuestion: {item['question']}\nAssistant:"
            inputs = processor(text=prompt, images=item["image"], return_tensors="pt")

        inputs = inputs.to(model.device)
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)

        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        full_output = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
        formatted = extract_and_format_answer(full_output)

        predictions.append({
            "question_id": item["question_id"],
            "answer": formatted,
            "full_answer": full_output,
            "_domain": item["domain"],
            "_ground_truth": item.get("answer"),
            "_question": item["question"],
        })

    logger.info(f"Inference complete: {len(predictions)} predictions")
    return predictions


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 — EVALUATION + BENCHMARKING
# ═══════════════════════════════════════════════════════════════════════════
def evaluate_and_benchmark(predictions, model_key):
    """Evaluate with ANLS and print comparison against competition leaderboard."""
    from src.evaluation.local_evaluator import compute_anls

    domain_scores = {}
    all_scores = []

    for pred in predictions:
        gt = pred.get("_ground_truth", "")
        if gt:
            score = compute_anls(pred["answer"], gt)
            all_scores.append(score)
            domain_scores.setdefault(pred.get("_domain", "unknown"), []).append(score)

    overall = sum(all_scores) / len(all_scores) if all_scores else 0.0
    n_unknown = sum(1 for p in predictions if p["answer"] == "Unknown")

    # ── Per-Domain Results ──
    print("\n" + "═" * 70)
    print("  📊 EVALUATION RESULTS — ANLS")
    print("═" * 70)
    print(f"  Total evaluated : {len(all_scores)}")
    print(f"  Unknown answers : {n_unknown}/{len(predictions)}")
    print(f"  OVERALL ANLS    : {overall:.4f}")
    print("─" * 70)

    domain_results = {}
    for domain in sorted(domain_scores.keys()):
        scores = domain_scores[domain]
        avg = sum(scores) / len(scores)
        domain_results[domain] = avg
        bar = "█" * int(avg * 20) + "░" * (20 - int(avg * 20))
        print(f"  {domain:<25} {bar} {avg:.4f}  ({len(scores)} Qs)")

    # ── Leaderboard Benchmarking ──
    model_info = MODEL_REGISTRY[model_key]
    our_entry = {"method": f"Ours ({model_info['model_id'].split('/')[-1]})",
                 "category": model_info["category"], "score": overall}

    combined = LEADERBOARD + [our_entry]
    combined.sort(key=lambda x: x["score"], reverse=True)

    print("\n" + "═" * 70)
    print("  🏆 COMPETITION LEADERBOARD COMPARISON")
    print("═" * 70)
    print(f"  {'Rank':<5} {'Method':<45} {'Category':<10} {'Score':<8}")
    print("─" * 70)

    our_rank = None
    for i, entry in enumerate(combined, 1):
        is_ours = entry["method"].startswith("Ours")
        marker = " ◀ YOU" if is_ours else ""
        if is_ours:
            our_rank = i
        print(f"  {i:<5} {entry['method']:<45} {entry['category']:<10} {entry['score']:.4f}{marker}")

    print("═" * 70)
    if our_rank:
        print(f"\n  📍 Your rank: #{our_rank} out of {len(combined)} methods")

    # ── Category-specific ranking ──
    our_cat = model_info["category"]
    cat_entries = [e for e in combined if e["category"] == our_cat or
                   (our_cat == "Up to 8B parameters" and e["category"] == "≤8B")]
    if cat_entries:
        cat_rank = next((i+1 for i, e in enumerate(cat_entries)
                         if e["method"].startswith("Ours")), None)
        if cat_rank:
            print(f"  📍 In '{our_cat}' category: #{cat_rank} out of {len(cat_entries)}")

    print("═" * 70 + "\n")

    return {
        "overall_anls": overall,
        "per_domain": domain_results,
        "total_evaluated": len(all_scores),
        "unknown_count": n_unknown,
        "leaderboard_rank": our_rank,
        "leaderboard_total": len(combined),
    }


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5 — SAVE
# ═══════════════════════════════════════════════════════════════════════════
def save_results(predictions, eval_results, model_key, output_dir="results"):
    """Save submission JSON, full predictions, evaluation summary, and benchmark."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Competition submission (strips internal fields)
    submission = [
        {"question_id": p["question_id"], "answer": p["answer"],
         "full_answer": p.get("full_answer", "")}
        for p in predictions
    ]
    sub_path = out / f"submission_{model_key}_{ts}.json"
    with open(sub_path, "w", encoding="utf-8") as f:
        json.dump(submission, f, indent=2, ensure_ascii=False)

    # Full predictions (for debugging)
    pred_path = out / f"predictions_{model_key}_{ts}.json"
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False, default=str)

    # Evaluation + benchmark summary
    eval_path = out / f"evaluation_{model_key}_{ts}.json"
    eval_results["model"] = MODEL_REGISTRY[model_key]["model_id"]
    eval_results["timestamp"] = ts
    eval_results["leaderboard_snapshot"] = LEADERBOARD
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)

    print(f"\n📁 Results saved to {out}/:")
    print(f"   submission_{model_key}_{ts}.json    ← Upload to RRC portal")
    print(f"   predictions_{model_key}_{ts}.json   ← Full debug predictions")
    print(f"   evaluation_{model_key}_{ts}.json    ← Scores + leaderboard comparison")
    return str(sub_path)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description="ICDAR 2026 DocVQA — One-Shot Training + Evaluation"
    )
    p.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), default="qwen-7b",
                   help="Model to train/evaluate (default: qwen-7b)")
    p.add_argument("--epochs", type=int, default=2, help="Training epochs")
    p.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    p.add_argument("--batch-size", type=int, default=1, help="Per-device batch size")
    p.add_argument("--skip-training", action="store_true", help="Skip training")
    p.add_argument("--max-samples", type=int, default=None, help="Limit samples (debug)")
    p.add_argument("--adapter-path", type=str, default=None, help="Pre-trained adapter path")
    p.add_argument("--split", choices=["val", "test"], default="val", help="Dataset split")
    return p.parse_args()


def main():
    args = parse_args()
    info = MODEL_REGISTRY[args.model]

    print("\n" + "═" * 65)
    print("  🚀 ICDAR 2026 DocVQA — Training Pipeline")
    print("═" * 65)
    print(f"  Model    : {info['model_id']} ({info['params']})")
    print(f"  Category : {info['category']}")
    if torch.cuda.is_available():
        print(f"  GPU      : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM     : {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    else:
        print(f"  GPU      : None (CPU mode — will be slow)")
    print(f"  Split    : {args.split}")
    print(f"  Training : {'SKIP' if args.skip_training else f'{args.epochs} epochs'}")
    print("═" * 65 + "\n")

    # Step 1: Data
    logger.info("━━━ STEP 1/5: Dataset ━━━")
    dataset = download_dataset(split=args.split)
    samples = prepare_training_data(dataset)

    # Step 2: Training
    adapter_path = args.adapter_path
    if not args.skip_training and args.split == "val":
        logger.info("━━━ STEP 2/5: QLoRA Fine-Tuning ━━━")
        adapter_path = train_model(
            args.model, samples, args.epochs, args.batch_size, args.lr,
        )
    else:
        logger.info("━━━ STEP 2/5: Training SKIPPED ━━━")
        if not adapter_path:
            default = Path("checkpoints") / args.model / "final_adapter"
            if default.exists():
                adapter_path = str(default)
                logger.info(f"Found existing adapter: {adapter_path}")

    # Step 3: Inference
    logger.info("━━━ STEP 3/5: Inference ━━━")
    predictions = run_inference(args.model, adapter_path, samples, args.max_samples)

    # Step 4: Evaluation + Benchmarking
    logger.info("━━━ STEP 4/5: Evaluation & Benchmarking ━━━")
    eval_results = evaluate_and_benchmark(predictions, args.model)

    # Step 5: Save
    logger.info("━━━ STEP 5/5: Saving Results ━━━")
    sub_path = save_results(predictions, eval_results, args.model)

    print("\n" + "═" * 65)
    print("  ✅ PIPELINE COMPLETE")
    print("═" * 65)
    print(f"  ANLS Score  : {eval_results['overall_anls']:.4f}")
    print(f"  Rank        : #{eval_results.get('leaderboard_rank', '?')}")
    print(f"  Submission  : {sub_path}")
    print(f"\n  Submit at: https://rrc.cvc.uab.es/?ch=34&com=mymethods&task=1")
    print("═" * 65 + "\n")


if __name__ == "__main__":
    main()
