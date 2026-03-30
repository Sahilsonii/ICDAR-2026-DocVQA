# ICDAR 2026 DocVQA — Competitive System

**Competition:** [ICDAR 2026 DocVQA](https://rrc.cvc.uab.es/?ch=34&com=introduction)  
**Deadline:** Test Submission — 03 April 2026 · Report — 17 April 2026  
**Category:** Up to 8B parameters

---

## Architecture

```
 Document Images (8 domains)
       │
       ▼
┌──────────────────────────────────────────────────────┐
│  train_all_gpu.py — Zero-Shot Pipeline               │
│                                                      │
│  1. Download    DocVQA 2026 from HuggingFace         │
│  2. Parse       Docling text extraction per document  │
│  3. Inference   Qwen2.5-VL (image + parsed text)     │
│  4. Evaluate    ANLS metric (per-domain breakdown)   │
│  5. Submit      Competition-ready JSON               │
└──────────────────────────────────────────────────────┘
```

### Approach (Uni-Parser Inspired)

Our approach draws from insights in the [Uni-Parser](https://arxiv.org/abs/2512.15098) pipeline (1st place, 0.5125 ANLS):

- **Parser + VLM fusion**: Docling extracts document text → injected as context into Qwen2.5-VL prompt alongside the image
- **Domain-aware prompting**: Specialized prompt supplements for each of the 8 document domains (maps, comics, engineering, etc.)
- **Official evaluation prompt**: Uses the exact `get_evaluation_prompt()` from the competition's `eval_utils.py`
- **Zero-shot inference**: No training on val/test (per competition rules)
- **Competition-compliant formatting**: Automatic date (→YYYY-MM-DD), number, percentage, and list formatting

### Models

| Model | Params | Role | VRAM (4-bit) |
|-------|--------|------|-------------|
| [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | 7B | Primary VLM | ~10 GB |
| [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | 3B | Lighter alternative | ~5 GB |
| [Docling](https://github.com/DS4SD/docling) | — | Document text extraction | CPU |

---

## Project Structure

```
docvqa2026/
├── scripts/
│   ├── train_all_gpu.py         ⭐ Zero-shot: parse → infer → evaluate → submit
│   ├── download_data.py         Download dataset from HuggingFace
│   ├── run_inference.py         Alternative OCR + reasoner pipeline
│   ├── evaluate_local.py        ANLS evaluation
│   └── prepare_submission.py    Validate submission JSON
├── src/
│   ├── data_loader.py           Dataset loading & QA pair extraction
│   ├── parser/                  Domain-aware document parsing
│   │   ├── parser_router.py     Routes domains → parser backends
│   │   ├── docling_parser.py    Docling (primary parser)
│   │   ├── paddleocr_vl.py      PaddleOCR-VL (alternative)
│   │   └── olmocr_parser.py     olmOCR (noisy scans)
│   ├── reasoner/                Question answering
│   │   ├── prompt_builder.py    Domain-specific prompts
│   │   └── answer_formatter.py  Post-processing rules
│   ├── pipeline/
│   │   └── docvqa_pipeline.py   Alternative orchestrator
│   └── evaluation/
│       ├── eval_utils.py        Official competition evaluation
│       ├── local_evaluator.py   ANLS scoring
│       └── error_analyzer.py    Failure pattern analysis
├── results/                     Submissions, predictions, benchmarks
├── configs/                     YAML configs + prompt templates
├── tests/                       Unit + integration tests
├── report/                      ICDAR competition report template
├── requirements.txt             Pinned versions (CUDA 12.1)
└── .env.example                 Environment variable template
```

---

## Quick Start

### 1. Clone & Environment

```bash
git clone https://github.com/YOUR_USERNAME/docvqa2026.git
cd docvqa2026
python -m venv env
source env/bin/activate        # Linux/Mac
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
nano .env   # Add your HF_TOKEN (required)
```

### 3. Run Pipeline

```bash
# Validation set: inference + ANLS evaluation
python scripts/train_all_gpu.py --split val

# Test set: generate submission JSON
python scripts/train_all_gpu.py --split test

# Both at once
python scripts/train_all_gpu.py --split both
```

**Options:**
```bash
python scripts/train_all_gpu.py --model qwen-3b         # Smaller model (~5 GB)
python scripts/train_all_gpu.py --no-parser              # Skip Docling, VLM-only
python scripts/train_all_gpu.py --max-samples 5          # Quick debug
python scripts/train_all_gpu.py --adapter-path <PATH>    # Use fine-tuned adapter
```

### 4. Submit

```bash
# Validate submission format first
python scripts/prepare_submission.py results/submission_test_*.json

# Then upload at:
# https://rrc.cvc.uab.es/?ch=34&com=mymethods&task=1
```

---

## Data Flow (No Leakage)

```
                    ┌─────────────────────┐
                    │  HuggingFace        │
                    │  VLR-CVC/DocVQA-2026│
                    └────────┬────────────┘
                             │
            ┌────────────────┴────────────────┐
            │                                 │
      prepare_eval_data()              prepare_eval_data()
            │                                 │
    ┌───────┴───────┐                ┌────────┴──────┐
    │ eval_samples  │                │ ground_truth  │
    │ (NO answers)  │                │ (separate)    │
    └───────┬───────┘                └────────┬──────┘
            │                                 │
     parse_documents()                        │
            │                                 │
     run_inference()                          │
            │                                 │
    ┌───────┴───────┐                         │
    │  predictions  │                         │
    └───────┬───────┘                         │
            │                                 │
            └────────────┬────────────────────┘
                         │
              evaluate_and_benchmark()
                         │
                   ┌─────┴─────┐
                   │ ANLS Score │
                   └───────────┘
```

**Key**: The model NEVER sees ground truth. Evaluation uses a separate GT lookup.

---

## Scripts Reference

| Script | What it does | GPU? |
|--------|-------------|:----:|
| `train_all_gpu.py` | **Main pipeline**: parse → infer → evaluate → submit | ✅ |
| `download_data.py` | Download DocVQA 2026 val/test sets from HuggingFace | ❌ |
| `run_inference.py` | Alternative OCR→reasoner inference pipeline | Depends |
| `evaluate_local.py` | Evaluate predictions JSON using ANLS | ❌ |
| `prepare_submission.py` | Validate submission JSON format | ❌ |

---

## Evaluation

**ANLS** (Average Normalized Levenshtein Similarity) — official competition metric:
- `1.0` = perfect match
- `0.0` = completely wrong
- Threshold: 0.5 (below = 0 score)

Output includes per-domain breakdown and automatic leaderboard comparison:

```
═══════════════════════════════════════════════════════════════════
  🏆 COMPETITION LEADERBOARD COMPARISON
═══════════════════════════════════════════════════════════════════
  Rank  Method                                   Category   Score
─────────────────────────────────────────────────────────────────
  1     Uni-Parser + Mix-MLLMs                   >35B       0.5125
  2     Uni-Parser + Gemini-3.1-Pro              >35B       0.4625
  ...
  8     Ours (Qwen2.5-VL-7B-Instruct)           ≤8B        0.XXXX  ◀ YOU
  ...
═══════════════════════════════════════════════════════════════════
```

---

## Competition Rules (Followed)

- ✅ **No training on val/test** — zero-shot inference only
- ✅ Answer format: `FINAL ANSWER: [answer]`
- ✅ Dates → YYYY-MM-DD, Percentages → no space, Numbers → no comma separators
- ✅ Multiple answers separated by `, ` (not "and")
- ✅ Unanswerable → `Unknown`
- ✅ JSON format: `[{question_id, answer, full_answer}]`
- ✅ Parameter count declared per category

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 8 GB (3B model) | 16+ GB (7B model) |
| RAM | 16 GB | 32+ GB |
| Storage | 50 GB | 100+ GB |
| CUDA | 12.1 | 12.1 |
| Python | 3.10.x | 3.10.x |

---

## Required Configuration

| Variable | Required? | Get it at | Purpose |
|----------|:---------:|----------|---------|
| `HF_TOKEN` | ✅ **Yes** | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) | Download gated models |
| `WANDB_API_KEY` | ❌ Optional | [wandb.ai/authorize](https://wandb.ai/authorize) | Experiment tracking |

---

## References

| Resource | Link |
|----------|------|
| Competition Portal | https://rrc.cvc.uab.es/?ch=34 |
| Official Eval Code | https://github.com/VLR-CVC/DocVQA2026 |
| Dataset | https://huggingface.co/datasets/VLR-CVC/DocVQA-2026 |
| Uni-Parser (1st Place) | https://arxiv.org/abs/2512.15098 |
| Qwen2.5-VL | https://arxiv.org/abs/2502.13923 |

---

*Built for the ICDAR 2026 DocVQA competition · Category: Up to 8B parameters*
