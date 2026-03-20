"""
olmOCR Parser wrapper — Fallback for severely degraded scanned documents.

Used for: Any domain with heavy noise, low resolution, or poor scan quality.
olmOCR is Allen AI's 7B VLM fine-tuned specifically for OCR on real-world scans.

Reference: https://arxiv.org/abs/2502.18443
Install:   pip install olmocr[gpu]
Model:     allenai/olmOCR-7B-0225-preview

Why olmOCR for degraded scans:
  - Trained on millions of scanned pages from Common Crawl
  - Handles bleed-through, coffee stains, handwriting mixing
  - 7B parameters — fits alongside Gemma on dual-GPU setup with quantization
"""

import logging
from typing import List

logger = logging.getLogger(__name__)


class OlmOCRParser:
    """
    olmOCR-7B wrapper for severely degraded or noisy scanned documents.

    Args:
        config (dict): Optional configuration. Recognized keys:
            - model_path (str): HuggingFace model ID (default: allenai/olmOCR-7B-0225-preview)
            - batch_size (int): Images per inference batch (default: 1)
            - max_new_tokens (int): Max tokens per page output (default: 2048)
    """

    def __init__(self, config: dict = None):
        config = config or {}
        self.model_path = config.get("model_path", "allenai/olmOCR-7B-0225-preview")
        self.batch_size = config.get("batch_size", 1)
        self.max_new_tokens = config.get("max_new_tokens", 2048)
        self._model = None
        self._processor = None
        logger.info(f"OlmOCRParser configured (model: {self.model_path}, lazy init)")

    def _load_model(self):
        """Lazily load olmOCR model and processor on first use."""
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForCausalLM
        except ImportError as e:
            raise ImportError(
                "transformers not installed. Run: pip install transformers>=4.45.0"
            ) from e

        import torch
        logger.info(f"Loading olmOCR model: {self.model_path}...")
        self._processor = AutoProcessor.from_pretrained(self.model_path)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self._model.eval()
        logger.info("olmOCR-7B loaded successfully")

    def parse(self, pages) -> str:
        """
        Parse degraded scanned pages using olmOCR.

        Args:
            pages: List of PIL.Image objects.

        Returns:
            Extracted text as Markdown string.
        """
        self._load_model()
        import torch

        full_output = []
        device = next(self._model.parameters()).device

        for page_idx, page_img in enumerate(pages):
            try:
                page_text = self._parse_single_page(page_img, page_idx + 1, device)
                full_output.append(page_text)
            except Exception as e:
                logger.error(f"olmOCR failed on page {page_idx + 1}: {e}")
                full_output.append(f"[Page {page_idx + 1}: OCR failed]")

        return "\n\n---PAGE_BREAK---\n\n".join(full_output)

    def _parse_single_page(self, page_img, page_num: int, device) -> str:
        """Run olmOCR inference on a single page."""
        import torch

        PROMPT = (
            "Read all text from this document image exactly as it appears. "
            "Preserve the layout and structure. Output only the text content."
        )

        inputs = self._processor(
            text=PROMPT,
            images=page_img,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self._processor.tokenizer.eos_token_id,
            )

        text = self._processor.tokenizer.decode(
            output_ids[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        return f"## Page {page_num}\n{text.strip()}"
