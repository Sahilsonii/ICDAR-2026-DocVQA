"""
Competition-compliant prompt builder for DocVQA 2026.

The official evaluator (`evaluate_docvqa_prediction`) requires:
  - The output to contain the marker "FINAL ANSWER:"
  - The answer after the marker is what gets evaluated

Design Notes:
  - The official prompt from `get_evaluation_prompt()` is designed for 35B+ models.
  - For 7B models (Qwen2.5-VL-7B), that prompt is too complex — the model defaults
    to "Unknown" because it can't handle 8 formatting rules + a reasoning protocol.
  - This implementation uses a SIMPLIFIED prompt optimized for 7B models:
    1. Short system instruction (encourages answering)
    2. Parsed document text (from Docling)
    3. Minimal formatting rules (only the essential ones)
    4. The question + FINAL ANSWER: tag
  - The key formatting rules (dates → YYYY-MM-DD, no comma separators, etc.)
    are handled POST-HOC by answer_formatter.py, which is more reliable than
    asking a 7B model to follow complex rules mid-generation.
"""

from typing import List, Optional
from PIL import Image


# ---------------------------------------------------------------------------
# Simplified prompt optimized for 7B VLM models
# ---------------------------------------------------------------------------
# Key differences from the official prompt:
#   1. Much shorter — 7B models get overwhelmed by long instructions
#   2. "Unknown" is de-emphasized — moved to end, framed as last resort
#   3. Formatting rules are minimal — heavy lifting done by answer_formatter.py
#   4. Directly asks to output FINAL ANSWER: tag
SYSTEM_PROMPT = (
    "You are an expert document analyst. Look at the document image carefully "
    "and answer the question. Always try your best to provide an answer based "
    "on what you can see in the document.\n\n"
    "Rules:\n"
    "- Give ONLY the answer, no explanations or sentences.\n"
    '- For multiple answers, separate with ", " (comma space).\n'
    "- Dates must be in YYYY-MM-DD format.\n"
    "- Numbers: no comma separators (use 1000 not 1,000).\n"
    "- Only say \"Unknown\" if the question is truly impossible to answer "
    "from the document.\n\n"
    "You MUST end your response with:\nFINAL ANSWER: <your answer>"
)


# ---------------------------------------------------------------------------
# Domain-specific supplemental prompts
# ---------------------------------------------------------------------------
DOMAIN_SUPPLEMENTS: dict[str, str] = {
    "maps": (
        "\nHint: This is a MAP. Look for place names, road labels, distances, "
        "grid coordinates, and legend entries in the image."
    ),
    "comics": (
        "\nHint: This is a COMIC. Read panels left-to-right, top-to-bottom. "
        "Look at speech bubbles, character names, and narrative text."
    ),
    "engineering_drawing": (
        "\nHint: This is an ENGINEERING DRAWING. Look for dimension labels, "
        "title block info, part numbers, and measurement annotations."
    ),
    "infographics": (
        "\nHint: This is an INFOGRAPHIC. Look for chart labels, data values, "
        "percentages, and legend entries."
    ),
    "science_poster": (
        "\nHint: This is a SCIENCE POSTER. Check the title, abstract, results "
        "section, figure captions, and author information."
    ),
    "science_paper": (
        "\nHint: This is a SCIENCE PAPER. Check the abstract, tables, figure "
        "captions, equations, and references."
    ),
    "slide": (
        "\nHint: This is a PRESENTATION SLIDE. Check the slide title, bullet "
        "points, and any charts or tables."
    ),
    "business_report": (
        "\nHint: This is a BUSINESS REPORT. Look for financial tables, dates, "
        "company names, and numerical data."
    ),
}


def build_prompt(
    question: str,
    parsed_context: str,
    domain: str,
    images: Optional[List[Image.Image]] = None,
) -> str:
    """
    Build a concise, 7B-optimized prompt for DocVQA.

    The prompt structure:
        1. System instruction (short, encouraging)
        2. Parsed document text (from Docling, if available)
        3. Domain-specific hint (one line)
        4. The question

    Args:
        question: The question string.
        parsed_context: Structured Markdown text from the parser.
        domain: Document category (e.g., "maps", "science_paper").
        images: Unused (kept for API compatibility).

    Returns:
        Complete prompt string ready for tokenization.
    """
    domain_key = domain.lower().replace(" ", "_")
    domain_hint = DOMAIN_SUPPLEMENTS.get(domain_key, "")

    parts = [SYSTEM_PROMPT]

    # Add parsed document context — only if it's substantial
    if parsed_context and len(parsed_context.strip()) > 20:
        MAX_CONTEXT_CHARS = 4000
        ctx = parsed_context[:MAX_CONTEXT_CHARS] if len(parsed_context) > MAX_CONTEXT_CHARS else parsed_context
        parts.append(f"\n\nDocument text:\n{ctx}")

    # Add domain hint (single line)
    if domain_hint:
        parts.append(domain_hint)

    # Add the question
    parts.append(f"\nQuestion: {question}")

    return "\n".join(parts)
