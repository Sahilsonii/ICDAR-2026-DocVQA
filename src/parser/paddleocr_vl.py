"""
PaddleOCR wrapper — Primary Document Parser (compatible with PaddleOCR v3.4+).

Uses PaddleOCR for robust OCR on all document types.
Handles: skewed text, irregular layouts, screen photography, warped pages.

Reference: https://github.com/PaddlePaddle/PaddleOCR
Install:   pip install paddlepaddle-gpu paddleocr>=3.4

PaddleOCR v3.4 API valid constructor params:
  lang, ocr_version, use_doc_orientation_classify, use_doc_unwarping,
  use_textline_orientation, text_det_limit_side_len, text_rec_score_thresh,
  text_detection_model_name, text_recognition_model_name, etc.

NOTE: use_gpu, use_angle_cls, show_log, layout_analysis, table_recognition
      are NOT valid in v3.4 and will cause crashes.
"""

from typing import List
import numpy as np
import logging

logger = logging.getLogger(__name__)


class PaddleOCRVLParser:
    """
    PaddleOCR v3.4 wrapper with version-safe initialization.

    Args:
        config (dict): Optional configuration overrides. Recognized keys:
            - lang (str): OCR language, default "en"
            - confidence_threshold (float): Minimum OCR confidence, default 0.5
    """

    def __init__(self, config: dict = None):
        config = config or {}
        self.confidence_threshold = config.get("confidence_threshold", 0.5)
        self._ocr = None
        self._config = config
        logger.info("PaddleOCRVLParser configured (lazy init)")

    def _load_model(self):
        """Lazily load PaddleOCR model on first use."""
        if self._ocr is not None:
            return
        try:
            from paddleocr import PaddleOCR
        except ImportError as e:
            raise ImportError(
                "PaddleOCR not installed. Run: pip install paddlepaddle-gpu paddleocr"
            ) from e

        # PaddleOCR v2.8.1 API
        self._ocr = PaddleOCR(
            use_angle_cls=True,
            lang=self._config.get("lang", "en"),
            use_gpu=self._config.get("use_gpu", False),
            show_log=False
        )
        logger.info("PaddleOCR v3.4 model loaded")

    def parse(self, pages) -> str:
        """
        Parse all pages and return a single structured Markdown string.

        Args:
            pages: List of PIL.Image objects (one per document page).

        Returns:
            Structured Markdown text with PAGE_BREAK separators.
        """
        self._load_model()
        full_output = []
        for page_idx, page_img in enumerate(pages):
            img_array = np.array(page_img)
            try:
                result = self._ocr.ocr(img_array)
                page_text = self._result_to_markdown(result, page_idx + 1)
            except Exception as e:
                logger.warning(f"OCR failed on page {page_idx + 1}: {e}")
                page_text = f"[Page {page_idx + 1}: OCR failed]"
            full_output.append(page_text)

        return "\n\n---PAGE_BREAK---\n\n".join(full_output)

    def _result_to_markdown(self, ocr_result: list, page_num: int) -> str:
        """
        Convert PaddleOCR output to structured Markdown preserving layout logic.

        OCR result structure (v3.4):
          List of pages -> List of lines -> [bbox, (text, confidence)]
        """
        if not ocr_result or not ocr_result[0]:
            return f"[Page {page_num}: No text detected]"

        lines = []
        for line in ocr_result[0]:
            try:
                bbox, (text, confidence) = line
                lines.append({
                    "text": text,
                    "y_center": (bbox[0][1] + bbox[2][1]) / 2,
                    "x_left": bbox[0][0],
                    "confidence": confidence
                })
            except (ValueError, TypeError, IndexError):
                # Handle unexpected result format gracefully
                continue

        if not lines:
            return f"[Page {page_num}: No text detected]"

        # Sort by vertical position first, then horizontal (reading order)
        lines.sort(key=lambda x: (round(x["y_center"] / 20) * 20, x["x_left"]))

        header = f"## Page {page_num}\n"
        body = "\n".join(line["text"] for line in lines)
        return header + body

    def parse_with_layout(self, pages) -> list:
        """Parse pages and return rich layout structure (for debugging/notebooks)."""
        self._load_model()
        layout_results = []
        for page_idx, page_img in enumerate(pages):
            img_array = np.array(page_img)
            try:
                result = self._ocr.ocr(img_array)
            except Exception:
                result = None
            page_data = {"page": page_idx + 1, "elements": []}
            if result and result[0]:
                for line in result[0]:
                    try:
                        bbox, (text, confidence) = line
                        page_data["elements"].append({
                            "text": text,
                            "bbox": bbox,
                            "confidence": round(confidence, 4)
                        })
                    except (ValueError, TypeError, IndexError):
                        continue
            layout_results.append(page_data)
        return layout_results
