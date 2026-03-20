"""
Docling Parser wrapper — Fallback for born-digital, clean PDFs.

Used for: Science Papers, Business Reports, Slides (clean structured layouts).
Docling converts PDFs -> rich Markdown preserving headers, tables, and lists.

Reference: https://arxiv.org/abs/2408.09869 (IBM Docling)
Install:   pip install docling>=2.5.0
"""

import io
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


class DoclingParser:
    """
    Docling-based parser for clean, born-digital documents.

    Args:
        config (dict): Optional configuration. Recognized keys:
            - output_format (str): "markdown" or "json" (default: "markdown")
    """

    def __init__(self, config: dict = None):
        config = config or {}
        self.output_format = config.get("output_format", "markdown")
        self._converter = None
        logger.info("DoclingParser configured (lazy init)")

    def _load_model(self):
        """Lazily load Docling converter on first use."""
        if self._converter is not None:
            return
        try:
            from docling.document_converter import DocumentConverter
        except ImportError as e:
            raise ImportError(
                "Docling not installed. Run: pip install docling>=2.5.0"
            ) from e

        # Use bare DocumentConverter — no PipelineOptions to avoid deprecation issues
        self._converter = DocumentConverter()
        logger.info("Docling DocumentConverter loaded")

    def parse(self, pages) -> str:
        """
        Parse document pages to structured Markdown.

        Args:
            pages: List of PIL.Image objects.

        Returns:
            Structured Markdown string.
        """
        self._load_model()
        try:
            return self._parse_from_images(pages)
        except Exception as e:
            logger.warning(f"Docling parsing failed: {e}. Falling back to raw text extraction.")
            return self._fallback_text(pages)

    def _parse_from_images(self, pages) -> str:
        """Convert PIL images to temp PDF bytes in chunks and run Docling to prevent OOM."""
        import tempfile
        import os

        # Process in chunks of 5 pages to prevent std::bad_alloc on large documents
        CHUNK_SIZE = 5
        results = []

        for offset in range(0, len(pages), CHUNK_SIZE):
            chunk = pages[offset : offset + CHUNK_SIZE]
            
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp_path = tmp.name

            try:
                if len(chunk) == 1:
                    chunk[0].save(tmp_path, format="PDF")
                else:
                    chunk[0].save(
                        tmp_path, format="PDF",
                        save_all=True, append_images=chunk[1:]
                    )

                result = self._converter.convert(tmp_path)

                if self.output_format == "markdown":
                    results.append(result.document.export_to_markdown())
                else:
                    results.append(str(result.document.export_to_dict()))
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        return "\n\n---CHUNK_BREAK---\n\n".join(results)

    def _fallback_text(self, pages) -> str:
        """Emergency fallback: extract text from PIL images using pytesseract if available."""
        try:
            import pytesseract
            texts = []
            for i, page in enumerate(pages):
                text = pytesseract.image_to_string(page)
                texts.append(f"## Page {i+1}\n{text}")
            return "\n\n---PAGE_BREAK---\n\n".join(texts)
        except ImportError:
            return "[DOCLING PARSING FAILED: install pytesseract as emergency fallback]"
