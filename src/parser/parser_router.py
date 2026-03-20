"""
Domain-aware parser selection router.

Routing logic (backed by benchmark analysis):
  - Maps, Engineering Drawings, Comics → PaddleOCR-VL-1.5
    (Handles skew, irregular layouts, non-standard text orientation)
  - Science Papers, Business Reports, Slides → Docling
    (Clean structured documents, born-digital PDFs)
  - Science Posters, Infographics → PaddleOCR-VL-1.5
    (High visual density, mixed layout)
  - Severely degraded scans (any domain) → olmOCR
    (Triggered programmatically, not by domain mapping)
"""

from enum import Enum
from typing import List, Optional
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class ParserBackend(Enum):
    PADDLEOCR_VL = "paddleocr_vl"  # Primary: handles skewed, irregular, scanned
    DOCLING = "docling"              # Fallback: born-digital, clean PDFs
    OLMOCR = "olmocr"                # Fallback: severely degraded scans


# Domain-to-parser routing table
# Update these assignments based on ablation study results in notebook 02.
DOMAIN_PARSER_MAP = {
    "maps":                  ParserBackend.PADDLEOCR_VL,   # Spatial, complex layout
    "comics":                ParserBackend.PADDLEOCR_VL,   # Irregular panels, rotated text
    "engineering_drawing":   ParserBackend.PADDLEOCR_VL,   # Dense technical layout
    "science_poster":        ParserBackend.PADDLEOCR_VL,   # Mixed visual/text density
    "infographics":          ParserBackend.PADDLEOCR_VL,   # Non-standard layout
    "science_paper":         ParserBackend.DOCLING,         # Clean, structured PDF
    "business_report":       ParserBackend.DOCLING,         # Clean, structured PDF
    "slide":                 ParserBackend.DOCLING,         # Clean layout
}


class ParserRouter:
    """
    Routes documents to the appropriate parser based on their domain category.

    Implements lazy initialization — parsers are only loaded when first needed,
    so importing this module doesn't load any models.

    Args:
        configs (dict): Configuration dict with keys "paddleocr_vl", "docling", "olmocr".
                        Each value is a config dict passed to the respective parser.
    """

    def __init__(self, configs: dict = None):
        self.configs = configs or {}
        self._parsers: dict = {}  # Lazy initialization cache

    def get_parser(self, domain: str):
        """Get (or lazily initialize) the appropriate parser for a domain."""
        backend = DOMAIN_PARSER_MAP.get(domain.lower(), ParserBackend.PADDLEOCR_VL)
        if backend not in self._parsers:
            logger.info(f"Loading parser backend: {backend.value} for domain: {domain}")
            self._parsers[backend] = self._load_parser(backend)
        return self._parsers[backend]

    def get_backend_for_domain(self, domain: str) -> ParserBackend:
        """Return the routing decision without loading any models."""
        return DOMAIN_PARSER_MAP.get(domain.lower(), ParserBackend.PADDLEOCR_VL)

    def _load_parser(self, backend: ParserBackend):
        """Instantiate a parser by backend enum value."""
        if backend == ParserBackend.PADDLEOCR_VL:
            from src.parser.paddleocr_vl import PaddleOCRVLParser
            return PaddleOCRVLParser(self.configs.get("paddleocr_vl", {}))
        elif backend == ParserBackend.DOCLING:
            from src.parser.docling_parser import DoclingParser
            return DoclingParser(self.configs.get("docling", {}))
        elif backend == ParserBackend.OLMOCR:
            from src.parser.olmocr_parser import OlmOCRParser
            return OlmOCRParser(self.configs.get("olmocr", {}))
        else:
            raise ValueError(f"Unknown parser backend: {backend}")

    def parse(self, pages: List[Image.Image], domain: str) -> str:
        """
        Parse a document using the domain-appropriate parser.

        Args:
            pages: List of PIL.Image objects (all pages of the document).
            domain: Document category string (e.g., "maps", "science_paper").

        Returns:
            Structured Markdown/text representation of the document.
        """
        parser = self.get_parser(domain)
        return parser.parse(pages)

    def override_backend(self, domain: str, backend: ParserBackend):
        """
        Manually override the routing for a domain (useful for ablation studies).

        Example:
            router.override_backend("science_paper", ParserBackend.PADDLEOCR_VL)
        """
        DOMAIN_PARSER_MAP[domain.lower()] = backend
        # Invalidate cached parser if backend changed
        if backend not in self._parsers:
            logger.info(f"Routing override: {domain} → {backend.value}")

    @classmethod
    def from_config(cls, config: dict) -> "ParserRouter":
        """Create a ParserRouter from a config dict (e.g., loaded from pipeline_config.yaml)."""
        return cls(configs=config.get("parsers", {}))
