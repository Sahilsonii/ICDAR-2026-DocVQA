# [Your Team Name] — ICDAR 2026 DocVQA System Description

> **Competition:** ICDAR 2026 Document Visual Question Answering Challenge
> **Submission Portal:** https://rrc.cvc.uab.es/?ch=34&com=mymethods&task=1
> **Contact:** docvqa@cvc.uab.cat
> **Model Category:** Up to 8B parameters

---

## 1. System Overview

We present a VLM-based pipeline using Qwen2.5-VL-7B-Instruct fine-tuned with
QLoRA on the DocVQA 2026 training data. The system operates in an OCR-free
manner — processing document images directly through the vision encoder —
while maintaining competition-compliant answer formatting.

---

## 2. Method

### 2.1 Model Architecture

We use Qwen2.5-VL-7B-Instruct [1], a vision-language model with 7B parameters,
fine-tuned with QLoRA (4-bit NF4 quantization + LoRA adapters) on the DocVQA
training data. Key training details:

- **Quantization:** 4-bit NF4 with double quantization (`bitsandbytes`)
- **LoRA config:** rank=16, alpha=32, dropout=0.05, applied to q/k/v/o projections
- **Optimizer:** Paged AdamW 8-bit
- **Gradient checkpointing:** Enabled (VRAM optimization)
- **Effective batch size:** 8 (gradient accumulation)

### 2.2 Inference Pipeline

The model processes document images directly (OCR-free) using Qwen2.5-VL's
native vision encoder. For the parsing-based pipeline, we also support:

| Domain | Parser | Rationale |
|--------|--------|-----------|
| Maps | PaddleOCR-VL / Docling | Spatial layout, irregular text |
| Comics | PaddleOCR-VL / Docling | Non-standard panels, rotated bubbles |
| Engineering Drawing | PaddleOCR-VL / Docling | Dense technical annotation |
| Science Poster | PaddleOCR-VL / Docling | Mixed visual/text density |
| Infographics | PaddleOCR-VL / Docling | Embedded data labels |
| Science Paper | Docling | Born-digital, clean PDF |
| Business Report | Docling | Born-digital, structured |
| Slide | Docling | Clean layout |

### 2.3 Answer Formatting

Post-processing enforces all competition formatting rules:
- Dates → YYYY-MM-DD format
- Numbers with units: space between number and unit (e.g., "50 kg")
- Percentages: no space before % (e.g., "50%")
- Thousands separators removed (1000 not 1,000)
- Common filler phrases removed ("The answer is...", "Based on...")
- List items joined with ", " (not "and")

### 2.4 Approach Insights

Drawing from the Uni-Parser pipeline (1st place, 0.5125 ANLS) [4]:
- OCR-free processing when possible (VLM handles image directly)
- Domain-specific prompt engineering for each of the 8 document types
- Official competition prompt from `eval_utils.py` used during both training and inference

---

## 3. Results

<!-- Fill in after running: python scripts/train_all_gpu.py -->

| Domain | ANLS Score | N Questions |
|--------|-----------|-------------|
| Business Report | X.XX | XX |
| Comics | X.XX | XX |
| Engineering Drawing | X.XX | XX |
| Infographics | X.XX | XX |
| Maps | X.XX | XX |
| Science Paper | X.XX | XX |
| Science Poster | X.XX | XX |
| Slide | X.XX | XX |
| **Overall** | **X.XX** | **XXX** |

---

## 4. Parameter Count

| Component | Parameters | Role |
|-----------|-----------|------|
| Qwen2.5-VL-7B-Instruct | 7B | VLM (fine-tuned with QLoRA) |
| LoRA Adapter | ~50M | Fine-tuning adapter |

**Total:** ~7B parameters → category: **Up to 8B parameters**

---

## 5. Conclusions

<!-- Update after obtaining results -->

The QLoRA fine-tuning approach enables competitive performance in the ≤8B
category by adapting a strong vision-language model to the specific document
domains and answer formats required by the competition.

The most challenging domains are `maps` and `engineering_drawing` due to their
need for spatial reasoning beyond text extraction. Future work includes:
- Multi-scale inference (tiling high-res images)
- Domain-specific LoRA adapters
- Ensemble with parsing-based pipeline

---

## References

[1] Qwen2.5-VL: https://arxiv.org/abs/2502.13923
[2] PaddleOCR-VL: https://arxiv.org/abs/2505.09966
[3] Docling: https://github.com/DS4SD/docling
[4] Uni-Parser (1st place): https://arxiv.org/abs/2512.15098
[5] DocVQA 2026 Official: https://github.com/VLR-CVC/DocVQA2026
