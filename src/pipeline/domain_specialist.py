"""
Domain-specific pre/post-processing hooks for the DocVQA pipeline.

This module allows domain specialists to:
  1. Pre-process parsed context before it's sent to the reasoner
  2. Post-process raw answers before formatting

Extend this as you discover domain-specific failure modes in the error analysis notebooks.
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class DomainSpecialist:
    """
    Pluggable domain-specific processing hooks.

    All methods return the (possibly modified) input unchanged by default.
    Override specific domain handlers as you discover failure patterns.
    """

    def preprocess_context(self, context: str, domain: str) -> str:
        """
        Domain-specific transformations on parsed document text before reasoning.

        Current implementations:
          - maps: Highlight grid coordinate patterns
          - engineering_drawing: Emphasize dimension tables
          - comics: Reorganize panel text by spatial order hint
        """
        handler = getattr(self, f"_preprocess_{domain.lower()}", None)
        if handler:
            return handler(context)
        return context

    def postprocess_answer(self, answer: str, domain: str, question: str) -> str:
        """
        Domain-specific answer corrections after formatting.

        Current implementations:
          - maps: Normalize grid coordinates to uppercase
          - engineering_drawing: Normalize unit abbreviations
        """
        handler = getattr(self, f"_postprocess_{domain.lower()}", None)
        if handler:
            return handler(answer, question)
        return answer

    # ------------------------------------------------------------------
    # Pre-processing handlers by domain
    # ------------------------------------------------------------------

    def _preprocess_maps(self, context: str) -> str:
        """Highlight alphanumeric grid coordinates for maps."""
        # Add a hint banner so the model pays attention to grid refs
        if re.search(r"\b[A-Z]-\d+\b", context):
            context = "[MAP GRID COORDINATES PRESENT: e.g. A-1, B-10]\n\n" + context
        return context

    def _preprocess_engineering_drawing(self, context: str) -> str:
        """Emphasize dimension tables in engineering drawings."""
        if "BILL OF MATERIALS" in context.upper() or "BOM" in context.upper():
            context = "[BILL OF MATERIALS TABLE DETECTED]\n\n" + context
        return context

    def _preprocess_comics(self, context: str) -> str:
        """Add panel reading order hint for comics."""
        return "[READ PANELS: left-to-right, top-to-bottom]\n\n" + context

    # ------------------------------------------------------------------
    # Post-processing handlers by domain
    # ------------------------------------------------------------------

    def _postprocess_maps(self, answer: str, question: str) -> str:
        """Normalize map grid coordinates to uppercase (e.g., 'e-10' → 'E-10')."""
        return re.sub(r"\b([a-z])-(\d+)\b", lambda m: f"{m.group(1).upper()}-{m.group(2)}", answer)

    def _postprocess_engineering_drawing(self, answer: str, question: str) -> str:
        """Normalize common engineering unit abbreviations."""
        replacements = {
            r"\bmillimeters?\b": "mm",
            r"\bcentimeters?\b": "cm",
            r"\bmeters?\b": "m",
            r"\bkilometers?\b": "km",
            r"\binches?\b": "in",
            r"\bfeets?\b": "ft",
        }
        for pattern, replacement in replacements.items():
            answer = re.sub(pattern, replacement, answer, flags=re.IGNORECASE)
        return answer
