"""
Competition-compliant prompt builder for Gemma 3 27B.

The official master prompt structure is defined in eval_utils.py at:
https://github.com/VLR-CVC/DocVQA2026/blob/main/eval_utils.py

This file extends it with:
  - Domain-specific context injection (maps, comics, engineering_drawing)
  - Structured document context from PaddleOCR-VL
  - Context length management (128K Gemma context)
"""

from typing import List, Optional
from PIL import Image

# ---------------------------------------------------------------------------
# Master competition prompt template
# ---------------------------------------------------------------------------
MASTER_PROMPT_TEMPLATE = """You are an expert document analyst. Your task is to answer questions about the provided document.

DOCUMENT CONTENT (extracted by PaddleOCR-VL layout analysis):
{parsed_context}

DOCUMENT DOMAIN: {domain}

QUESTION: {question}

STRICT FORMATTING RULES — You MUST follow these exactly:
1. Answer ONLY with information found in the document. If unanswerable, reply exactly: Unknown
2. Multiple answers: separate with ", " (comma space). Never use "and".
3. Numbers with units: add one space between number and unit (e.g., "50 kg", "10 USD")
4. Percentages: NO space between number and % (e.g., "50%")
5. Dates: always use YYYY-MM-DD format (e.g., "Jan 1st 24" → "2024-01-01")
6. Decimals: use period "." never comma (e.g., "3.14")
7. Large numbers: NO comma separator (e.g., "1000" not "1,000")
8. No filler text: output ONLY the answer, never "The answer is..."

FINAL ANSWER: """


# ---------------------------------------------------------------------------
# Domain-specific supplemental prompts
# ---------------------------------------------------------------------------
DOMAIN_SUPPLEMENTS: dict[str, str] = {
    "maps": """
ADDITIONAL CONTEXT FOR MAPS:
- Road types and names are visually encoded (line styles, colors)
- Grid coordinates are alphanumeric (e.g., "E-10")
- Mileage is shown on road segments as small numbers
- Legend is typically in the corner — consult it for symbol meanings
""",
    "comics": """
ADDITIONAL CONTEXT FOR COMICS:
- Read panels in order: left-to-right, top-to-bottom
- Speech bubbles belong to the character nearest their tail
- Sound effects are large stylized text (e.g., "POW", "CRASH")
- Narrator boxes are usually rectangular, speech is in rounded bubbles
""",
    "engineering_drawing": """
ADDITIONAL CONTEXT FOR ENGINEERING DRAWINGS:
- Measurements appear as dimension lines with arrows
- Title block is typically bottom-right
- Tolerances appear as ± values next to dimensions
- Bill of Materials (BOM) lists components with quantities
""",
    "infographics": """
ADDITIONAL CONTEXT FOR INFOGRAPHICS:
- Data values are embedded in chart bars, pie slices, or data labels
- Legend maps colors/patterns to categories
- Axis labels define units — always include them in numeric answers
- Flow arrows indicate sequence or causality between elements
""",
    "science_poster": """
ADDITIONAL CONTEXT FOR SCIENCE POSTERS:
- Abstract section summarizes the entire poster — read it first
- Results section contains key quantitative findings
- Figure captions are directly below figures
- Bullet points under headings are the primary information carriers
""",
    "slide": """
ADDITIONAL CONTEXT FOR SLIDES:
- Each slide typically covers one topic — look at the slide title first
- Bullet points are usually ordered by importance
- Speaker notes (if extracted) provide extra context
""",
}

# Max context characters before truncation (conservative for 128K token limit)
MAX_CONTEXT_CHARS = 90_000


def build_prompt(
    question: str,
    parsed_context: str,
    domain: str,
    images: Optional[List[Image.Image]] = None,
) -> str:
    """
    Build the full competition-compliant prompt for Gemma 3 27B.

    Args:
        question: The question string.
        parsed_context: Structured Markdown text from the parser.
        domain: Document category (e.g., "maps", "science_paper").
        images: Optionally provided for future multimodal prompt variants (unused for text-only).

    Returns:
        Complete prompt string ready for tokenization.
    """
    domain_key = domain.lower().replace(" ", "_")
    domain_supplement = DOMAIN_SUPPLEMENTS.get(domain_key, "")
    context = parsed_context + domain_supplement

    # Truncate if necessary to stay within safe token budget
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS] + "\n\n[Context truncated for length]"

    return MASTER_PROMPT_TEMPLATE.format(
        parsed_context=context,
        domain=domain.upper().replace("_", " "),
        question=question,
    )


def build_multimodal_prompt(
    question: str,
    parsed_context: str,
    domain: str,
) -> str:
    """
    Build a shorter prompt for use in multimodal (image+text) inference mode.
    The parsed_context is kept brief so the vision encoder's output dominates.
    """
    context_brief = parsed_context[:20_000] if len(parsed_context) > 20_000 else parsed_context
    return build_prompt(
        question=question,
        parsed_context=context_brief,
        domain=domain,
    )
