#!/usr/bin/env python3
"""
ICDAR 2026 DocVQA — Zero-Shot Inference & Evaluation Pipeline
==============================================================
Competition : https://rrc.cvc.uab.es/?ch=34
Dataset     : https://huggingface.co/datasets/VLR-CVC/DocVQA-2026

Architecture (Uni-Parser inspired):
  1. Download DocVQA 2026 val/test from HuggingFace
  2. Parse documents with Docling (text extraction for context)
  3. Run zero-shot VLM inference with Qwen2.5-VL (image + parsed text)
  4. Post-process answers (dates, numbers, percentages, etc.)
  5. Evaluate with official ANLS metric + per-domain breakdown (val only)
  6. Generate competition-ready submission JSON

IMPORTANT: Competition rules forbid training on val/test sets.
           This pipeline is INFERENCE ONLY (zero-shot or adapter-enhanced).

Usage:
    python scripts/train_all_gpu.py                              # Val: infer + eval
    python scripts/train_all_gpu.py --split test                 # Test: submission JSON
    python scripts/train_all_gpu.py --split both                 # Both val + test
    python scripts/train_all_gpu.py --model qwen-3b              # Smaller model
    python scripts/train_all_gpu.py --adapter-path <PATH>        # Fine-tuned adapter
    python scripts/train_all_gpu.py --no-parser                  # Skip Docling, VLM-only
    python scripts/train_all_gpu.py --max-samples 5              # Debug

Models (select via --model):
    qwen-7b  : Qwen/Qwen2.5-VL-7B-Instruct  (default, ~10 GB VRAM with 4-bit)
    qwen-3b  : Qwen/Qwen2.5-VL-3B-Instruct  (~5 GB VRAM with 4-bit)
"""

import gc
import os
import sys
import json
import time
import torch
import logging
import argparse
from pathlib import Path
from datetime import datetime

# ── OOM fix: reduce allocator fragmentation ───────────────────────────────
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("docvqa_pipeline")


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def free_gpu_memory(*objects):
    """Delete objects, run Python GC and empty the CUDA cache."""
    for obj in objects:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        freed = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
        logger.info(
            "GPU cache cleared. Allocated: %.1f GB | Reserved: %.1f GB | Free: %.1f GB",
            torch.cuda.memory_allocated() / 1e9,
            torch.cuda.memory_reserved() / 1e9,
            freed / 1e9,
        )


# ═══════════════════════════════════════════════════════════════════════════
# MODEL REGISTRY
# ═══════════════════════════════════════════════════════════════════════════
MODEL_REGISTRY = {
    "qwen-7b": {
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "params": "7B",
        "category": "≤8B",
    },
    "qwen-3b": {
        "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "params": "3B",
        "category": "≤8B",
    },
}

# Competition leaderboard (as of 2026-03-20) for benchmarking
LEADERBOARD = [
    {"method": "Uni-Parser + Mix-MLLMs",      "category": ">35B",   "score": 0.5125},
    {"method": "Uni-Parser + Gemini-3.1-Pro", "category": ">35B",   "score": 0.4625},
    {"method": "Gemini-3.1-Pro (Baseline)",   "category": ">35B",   "score": 0.3750},
    {"method": "Gemini-3-Flash (Baseline)",   "category": ">35B",   "score": 0.3563},
    {"method": "Uni-Parser + Qwen3.5-27B",    "category": "8B-35B", "score": 0.2938},
    {"method": "GPT-5.2 (Baseline)",          "category": ">35B",   "score": 0.2688},
    {"method": "Uni-Parser + Qwen3.5-4B",     "category": "≤8B",    "score": 0.1875},
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

    logger.info(f"Downloading DocVQA 2026 {split} split from HuggingFace...")
    dataset = load_dataset("VLR-CVC/DocVQA-2026", split=split)
    dataset.save_to_disk(str(cache_path))
    logger.info(f"Saved {len(dataset)} documents to {cache_path}")
    return dataset


def prepare_eval_data(dataset):
    """
    Convert dataset into inference samples (NO ground truth) + separate GT lookup.

    This prevents data leakage: the model never sees answers during inference.
    Ground truth is kept in a separate dict for post-inference evaluation only.

    Returns:
        eval_samples: List of dicts with image, question, question_id, domain, doc_id, pages
                      (NO 'answer' key — model must not see ground truth)
        ground_truth: Dict mapping question_id -> answer (empty for test split)
    """
    from src.data_loader import get_all_qa_pairs

    eval_samples = []
    ground_truth = {}

    for item in dataset:
        for qa in get_all_qa_pairs(item):
            eval_samples.append({
                "image": qa["pages"][0] if qa["pages"] else None,
                "pages": qa["pages"],       # All pages for multi-page parsing
                "question": qa["question"],
                "question_id": qa["question_id"],
                "domain": qa["doc_category"],
                "doc_id": qa["doc_id"],
                # NO "answer" key — inference must be blind to ground truth
            })
            if qa["ground_truth"] is not None:
                ground_truth[qa["question_id"]] = qa["ground_truth"]

    # Sanity check: ensure no leakage
    for s in eval_samples:
        assert "answer" not in s, (
            f"DATA LEAKAGE: Sample {s['question_id']} contains 'answer' key!"
        )

    logger.info(
        f"Prepared {len(eval_samples)} inference samples, "
        f"{len(ground_truth)} ground truth answers available"
    )
    if not ground_truth:
        logger.info("No ground truth available (test split) — evaluation will be skipped")

    return eval_samples, ground_truth


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 — DOCUMENT PARSING (Docling)
# ═══════════════════════════════════════════════════════════════════════════
def parse_documents(samples, use_parser=True):
    """
    Parse all unique documents using the Docling parser for text extraction.

    Follows the Uni-Parser approach: parse once per doc_id, then attach
    the parsed text to each QA sample as supplemental context for the VLM.

    Args:
        samples: List of eval sample dicts
        use_parser: If False, skip parsing (pure VLM-only mode)

    Returns:
        Dict mapping doc_id -> parsed_text (Markdown)
    """
    if not use_parser:
        logger.info("Parser disabled (--no-parser). Using VLM-only mode.")
        return {}

    from src.parser.docling_parser import DoclingParser

    parser = DoclingParser({"output_format": "markdown"})

    # Find unique documents
    unique_docs = {}
    for s in samples:
        if s["doc_id"] not in unique_docs:
            unique_docs[s["doc_id"]] = {
                "pages": s["pages"],
                "domain": s["domain"],
            }

    logger.info(f"Parsing {len(unique_docs)} unique documents with Docling...")

    doc_texts = {}
    for doc_id, doc_info in unique_docs.items():
        try:
            parsed_text = parser.parse(doc_info["pages"])
            doc_texts[doc_id] = parsed_text
            logger.info(
                f"  ✓ Parsed {doc_id} ({doc_info['domain']}): "
                f"{len(parsed_text)} chars"
            )
        except Exception as e:
            logger.warning(f"  ✗ Failed to parse {doc_id}: {e}")
            doc_texts[doc_id] = ""

    n_success = sum(1 for t in doc_texts.values() if t)
    logger.info(
        f"Parsing complete: {n_success}/{len(unique_docs)} documents parsed successfully"
    )
    return doc_texts


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 — INFERENCE (Qwen VLM + parsed context)
# ═══════════════════════════════════════════════════════════════════════════
def run_inference(model_key, adapter_path, samples, doc_texts, max_samples=None):
    """
    Run zero-shot inference with Qwen VLM.

    Combines:
      - Image input (via Qwen2.5-VL vision encoder)
      - Parsed document text from Docling (injected into prompt)
      - Domain-specific prompt supplements
      - Competition formatting rules

    This is the Uni-Parser-inspired approach: the model sees both the
    raw image AND the parsed text for maximum accuracy.

    Args:
        model_key: Key from MODEL_REGISTRY
        adapter_path: Path to fine-tuned adapter (or None for zero-shot)
        samples: List of dicts WITHOUT 'answer' key
        doc_texts: Dict mapping doc_id -> parsed text from Docling
        max_samples: Limit number of samples (for debugging)

    Returns:
        List of prediction dicts
    """
    from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
    from src.reasoner.answer_formatter import extract_and_format_answer
    from src.reasoner.prompt_builder import build_prompt
    from tqdm import tqdm

    model_info = MODEL_REGISTRY[model_key]
    model_id = model_info["model_id"]
    logger.info(f"═══ INFERENCE: {model_id} ═══")

    if adapter_path:
        logger.info(f"Mode: Adapter-enhanced ({adapter_path})")
    else:
        logger.info("Mode: Zero-shot")
    logger.info(f"Parsed context: {'Enabled' if doc_texts else 'Disabled (VLM-only)'}")

    # Log available VRAM
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        free = (torch.cuda.get_device_properties(0).total_memory
                - torch.cuda.memory_allocated()) / 1e9
        logger.info(f"VRAM: {free:.1f} GB free / {total:.1f} GB total")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    processor.image_processor.min_pixels = 256 * 28 * 28
    processor.image_processor.max_pixels = 512 * 28 * 28

    # Force all layers onto GPU
    total_vram = (
        torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
    )
    max_memory = {0: f"{int(total_vram * 0.90 / 1e9)}GiB", "cpu": "0GiB"}

    if adapter_path and Path(adapter_path).exists():
        logger.info(f"Loading fine-tuned adapter from {adapter_path}")
        from peft import PeftModel
        base_model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            max_memory=max_memory,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        logger.info("Loading base model for zero-shot inference")
        model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            max_memory=max_memory,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
    model.eval()

    predictions = []
    for item in tqdm(samples, desc="Inference"):
        # ── Build context-augmented prompt ────────────────────────────────
        domain = item.get("domain", "unknown")
        parsed_text = doc_texts.get(item["doc_id"], "")
        
        question_with_context = build_prompt(
            question=item["question"],
            parsed_context=parsed_text,
            domain=domain
        )

        # ── Construct VLM message ─────────────────────────────────────────
        messages = [{"role": "user", "content": [
            {"type": "image", "image": item["image"]},
            {"type": "text", "text": question_with_context},
        ]}]

        try:
            from qwen_vl_utils import process_vision_info
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, _ = process_vision_info(messages)
            inputs = processor(
                text=[text], images=image_inputs, padding=True, return_tensors="pt"
            )
        except (ImportError, Exception):
            prompt = (
                f"User: <image>\n{question_with_context}\nAssistant:"
            )
            inputs = processor(text=prompt, images=item["image"], return_tensors="pt")

        inputs = inputs.to(model.device)
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, max_new_tokens=256, do_sample=False
            )

        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        full_output = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
        formatted = extract_and_format_answer(full_output)

        predictions.append({
            "question_id": item["question_id"],
            "answer": formatted,
            "full_answer": full_output,
            "_domain": domain,
            "_question": item["question"],
            # NOTE: No _ground_truth — evaluation uses separate GT lookup
        })

    logger.info(f"Inference complete: {len(predictions)} predictions")
    free_gpu_memory(model)
    return predictions


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 — EVALUATION + BENCHMARKING
# ═══════════════════════════════════════════════════════════════════════════
def evaluate_and_benchmark(predictions, ground_truth, model_key):
    """
    Evaluate predictions against ground truth and compare to leaderboard.

    Args:
        predictions: List of prediction dicts from run_inference()
        ground_truth: Dict mapping question_id -> answer (from prepare_eval_data)
        model_key: Key from MODEL_REGISTRY

    Note: ground_truth is a SEPARATE data structure from predictions.
          This prevents data leakage.
    """
    from src.evaluation.local_evaluator import compute_anls

    if not ground_truth:
        logger.warning("No ground truth available — skipping evaluation")
        return {"overall_anls": None, "per_domain": {}, "total_evaluated": 0}

    domain_scores = {}
    all_scores = []
    missing_gt = 0

    for pred in predictions:
        qid = pred["question_id"]
        gt = ground_truth.get(qid)
        if gt is None:
            missing_gt += 1
            continue
        score = compute_anls(pred["answer"], gt)
        all_scores.append(score)
        domain = pred.get("_domain", "unknown")
        domain_scores.setdefault(domain, []).append(score)

    if missing_gt:
        logger.warning(f"{missing_gt} predictions had no matching ground truth")

    overall = sum(all_scores) / len(all_scores) if all_scores else 0.0
    n_unknown = sum(1 for p in predictions if p["answer"] == "Unknown")

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

    model_info = MODEL_REGISTRY[model_key]
    our_entry = {
        "method": f"Ours ({model_info['model_id'].split('/')[-1]})",
        "category": model_info["category"],
        "score": overall,
    }
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
        print(
            f"  {i:<5} {entry['method']:<45} "
            f"{entry['category']:<10} {entry['score']:.4f}{marker}"
        )

    print("═" * 70)
    if our_rank:
        print(f"\n  📍 Your rank: #{our_rank} out of {len(combined)} methods")

    our_cat = model_info["category"]
    cat_entries = [
        e for e in combined
        if e["category"] == our_cat
    ]
    if cat_entries:
        cat_rank = next(
            (i + 1 for i, e in enumerate(cat_entries) if e["method"].startswith("Ours")),
            None,
        )
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
def save_results(predictions, eval_results, model_key, split, output_dir="results"):
    """Save submission JSON, full predictions, and evaluation summary."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Submission JSON — matches official format ─────────────────────────
    # Format: [{"question_id": ..., "answer": ..., "full_answer": ...}]
    submission = [
        {
            "question_id": p["question_id"],
            "answer": p["answer"],
            "full_answer": p.get("full_answer", ""),
        }
        for p in predictions
    ]
    sub_path = out / f"submission_{split}_{model_key}_{ts}.json"
    with open(sub_path, "w", encoding="utf-8") as f:
        json.dump(submission, f, indent=2, ensure_ascii=False)

    # ── Full predictions with metadata (for debugging) ────────────────────
    pred_path = out / f"predictions_{split}_{model_key}_{ts}.json"
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False, default=str)

    # ── Evaluation summary ────────────────────────────────────────────────
    if eval_results.get("overall_anls") is not None:
        eval_path = out / f"evaluation_{split}_{model_key}_{ts}.json"
        eval_results["model"] = MODEL_REGISTRY[model_key]["model_id"]
        eval_results["timestamp"] = ts
        eval_results["split"] = split
        eval_results["leaderboard_snapshot"] = LEADERBOARD
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(eval_results, f, indent=2)

    print(f"\n📁 Results saved to {out}/:")
    print(f"   {sub_path.name:<50} ← Upload to RRC portal")
    print(f"   {pred_path.name:<50} ← Full debug predictions")
    if eval_results.get("overall_anls") is not None:
        eval_name = f"evaluation_{split}_{model_key}_{ts}.json"
        print(f"   {eval_name:<50} ← Scores + leaderboard")
    return str(sub_path)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description="ICDAR 2026 DocVQA — Zero-Shot Inference & Evaluation"
    )
    p.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), default="qwen-7b",
                   help="Model to use (default: qwen-7b)")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Limit samples for debugging")
    p.add_argument("--adapter-path", type=str, default=None,
                   help="Path to a LoRA adapter (trained on EXTERNAL data only)")
    p.add_argument("--split", choices=["val", "test", "both"], default="val",
                   help="Dataset split: val (local eval), test (submission), both")
    p.add_argument("--no-parser", action="store_true",
                   help="Skip Docling parsing (pure VLM-only, no text context)")
    return p.parse_args()


def run_pipeline_for_split(split, args):
    """Run the full pipeline for a single split (val or test)."""
    info = MODEL_REGISTRY[args.model]

    print("\n" + "═" * 65)
    print(f"  🚀 ICDAR 2026 DocVQA — {split.upper()} Split")
    print("═" * 65)
    print(f"  Model      : {info['model_id']} ({info['params']})")
    print(f"  Category   : {info['category']}")
    if torch.cuda.is_available():
        print(f"  GPU        : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM       : {vram:.1f} GB")
    else:
        print("  GPU        : None (CPU mode — will be slow)")
    print(f"  Split      : {split}")
    print(f"  Mode       : {'Adapter' if args.adapter_path else 'Zero-shot'}")
    print(f"  Parser     : {'Docling' if not args.no_parser else 'Disabled (VLM-only)'}")
    print("═" * 65 + "\n")

    # Step 1: Download dataset
    logger.info(f"━━━ STEP 1/5: Loading {split} dataset ━━━")
    dataset = download_dataset(split=split)

    # Prepare inference data — SEPARATE from ground truth
    eval_samples, ground_truth = prepare_eval_data(dataset)

    # Step 2: Parse documents with Docling
    logger.info("━━━ STEP 2/5: Document Parsing (Docling) ━━━")
    doc_texts = parse_documents(eval_samples, use_parser=not args.no_parser)

    # Step 3: Inference (model never sees ground truth)
    logger.info("━━━ STEP 3/5: VLM Inference ━━━")
    predictions = run_inference(
        args.model, args.adapter_path, eval_samples, doc_texts, args.max_samples
    )

    # Step 4: Evaluation (val only — test has no public GT)
    logger.info("━━━ STEP 4/5: Evaluation & Benchmarking ━━━")
    eval_results = evaluate_and_benchmark(predictions, ground_truth, args.model)

    # Step 5: Save submission JSON
    logger.info("━━━ STEP 5/5: Saving Results ━━━")
    sub_path = save_results(predictions, eval_results, args.model, split)

    print("\n" + "═" * 65)
    print(f"  ✅ {split.upper()} PIPELINE COMPLETE")
    print("═" * 65)
    if eval_results.get("overall_anls") is not None:
        print(f"  ANLS Score  : {eval_results['overall_anls']:.4f}")
        print(f"  Rank        : #{eval_results.get('leaderboard_rank', '?')}")
    else:
        print("  ANLS Score  : N/A (test split — submit to RRC portal)")
    print(f"  Submission  : {sub_path}")
    print(f"\n  Submit at: https://rrc.cvc.uab.es/?ch=34&com=mymethods&task=1")
    print("═" * 65 + "\n")

    return sub_path, eval_results


def main():
    args = parse_args()

    splits = ["val", "test"] if args.split == "both" else [args.split]

    for split in splits:
        run_pipeline_for_split(split, args)


if __name__ == "__main__":
    main()