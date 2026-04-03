"""
Competition-compliant prompt builder for DocVQA 2026.

Uses the OFFICIAL master prompt from the competition evaluation code:
  https://github.com/VLR-CVC/DocVQA2026/blob/main/eval_utils.py

The official evaluator (`evaluate_docvqa_prediction`) requires:
  - The output to contain the marker "FINAL ANSWER:" 
  - The answer after the marker is what gets evaluated

This file extends the official prompt with:
  - Domain-specific context injection (maps, comics, engineering_drawing, etc.)
  - Structured document context from Docling parser
  - Context length management for VRAM-constrained GPUs
"""

from typing import List, Optional
from PIL import Image


# ---------------------------------------------------------------------------
# Official master prompt (from DocVQA2026/eval_utils.py::get_evaluation_prompt)
# ---------------------------------------------------------------------------
OFFICIAL_MASTER_PROMPT = (
    "ACT AS an expert Document Visual Question Answering (DocVQA) system. "
    "ANALYZE the provided images to extract precise information.\n\n"
    "### MANDATORY RESPONSE RULES:\n"
    '1. SOURCE ADHERENCE: If the question is unanswerable from the document, respond ONLY with "Unknown".\n'
    '2. LIST FORMATTING: List multiple answers in order of appearance, separated by a comma and a single space (e.g., "Answer A, Answer B"). Do NOT use "and".\n'
    "3. NUMBERS & UNITS:\n"
    '   - Convert units to their standardized abbreviation (e.g., use "kg" not "kilograms", "m" not "meters").\n'
    '   - Place a single space between the number and the unit (e.g., "50 kg", "10 USD").\n'
    '4. PERCENTAGES: For percentages, attach the \'%\' symbol directly to the number with NO space (e.g., "50%", not "50 %").\n'
    '5. DATE FORMATTING: Convert all dates to YYYY-MM-DD format (e.g., convert "Jan 1st 24" to "2024-01-01").\n'
    '6. DECIMAL FORMATTING: Decimals should be separated by a single period (e.g., "3.14", not "3,14").\n'
    '7. THOUSANDS SEPARATOR: Do NOT use commas as thousands separators (e.g., "1000", not "1,000").\n'
    '8. NO FILLER: Output ONLY the result. Do not frame with sentences like "The answer is...".'
    "\n\n### REASONING PROTOCOL:\n"
    "1. Perform exhaustive step-by-step reasoning to locate and verify the data.\n"
    "2. Verify if the data contains a date, number, or unit.\n"
    "3. Step-by-step, transform the data to match the MANDATORY RESPONSE RULES (e.g., converting date format).\n"
    "\n\n### OUTPUT FORMAT:\n"
    "After your analysis, you MUST provide the final result in the following format:\n"
    "FINAL ANSWER: [Your exact formatted answer]\n"
    "Ensure the content inside [FINAL ANSWER] strictly follows the MANDATORY RESPONSE RULES."
)


# ---------------------------------------------------------------------------
# Domain-specific supplemental prompts
# ---------------------------------------------------------------------------
DOMAIN_SUPPLEMENTS: dict[str, str] = {
    "maps": (
        "\n\nADDITIONAL CONTEXT FOR MAPS:\n"
        "- Road types and names are visually encoded (line styles, colors)\n"
        '- Grid coordinates are alphanumeric (e.g., "E-10")\n'
        "- Mileage is shown on road segments as small numbers\n"
        "- Legend is typically in the corner — consult it for symbol meanings\n"
    ),
    "comics": (
        "\n\nADDITIONAL CONTEXT FOR COMICS:\n"
        "- Read panels in order: left-to-right, top-to-bottom\n"
        "- Speech bubbles belong to the character nearest their tail\n"
        '- Sound effects are large stylized text (e.g., "POW", "CRASH")\n'
        "- Narrator boxes are usually rectangular, speech is in rounded bubbles\n"
    ),
    "engineering_drawing": (
        "\n\nADDITIONAL CONTEXT FOR ENGINEERING DRAWINGS:\n"
        "- Measurements appear as dimension lines with arrows\n"
        "- Title block is typically bottom-right\n"
        "- Tolerances appear as ± values next to dimensions\n"
        "- Bill of Materials (BOM) lists components with quantities\n"
    ),
    "infographics": (
        "\n\nADDITIONAL CONTEXT FOR INFOGRAPHICS:\n"
        "- Data values are embedded in chart bars, pie slices, or data labels\n"
        "- Legend maps colors/patterns to categories\n"
        "- Axis labels define units — always include them in numeric answers\n"
        "- Flow arrows indicate sequence or causality between elements\n"
    ),
    "science_poster": (
        "\n\nADDITIONAL CONTEXT FOR SCIENCE POSTERS:\n"
        "- Abstract section summarizes the entire poster — read it first\n"
        "- Results section contains key quantitative findings\n"
        "- Figure captions are directly below figures\n"
        "- Bullet points under headings are the primary information carriers\n"
    ),
    "science_paper": (
        "\n\nADDITIONAL CONTEXT FOR SCIENCE PAPERS:\n"
        "- Abstract and conclusion sections contain key findings\n"
        "- Tables and figures have numbered captions with quantitative data\n"
        "- References are numbered in brackets [1], [2], etc.\n"
        "- Equations are numbered on the right margin\n"
    ),
    "slide": (
        "\n\nADDITIONAL CONTEXT FOR SLIDES:\n"
        "- Each slide typically covers one topic — look at the slide title first\n"
        "- Bullet points are usually ordered by importance\n"
        "- Speaker notes (if extracted) provide extra context\n"
    ),
    "business_report": (
        "\n\nADDITIONAL CONTEXT FOR BUSINESS REPORTS:\n"
        "- Financial data is usually in tables with row/column headers\n"
        "- Currency values may include symbols ($, €, £) or abbreviations (USD, EUR)\n"
        "- Dates are often fiscal year references (FY2024, Q3 2025)\n"
        "- Footnotes contain important qualifying details\n"
    ),
}


def build_prompt(
    question: str,
    parsed_context: str,
    domain: str,
    images: Optional[List[Image.Image]] = None,
) -> str:
    """
    Build the full competition-compliant prompt.

    The prompt structure:
        1. Official master prompt (reasoning + formatting rules)
        2. Parsed document text context (from Docling)
        3. Domain-specific hints
        4. The actual question

    Args:
        question: The question string.
        parsed_context: Structured Markdown text from the parser.
        domain: Document category (e.g., "maps", "science_paper").
        images: Unused (kept for API compatibility).

    Returns:
        Complete prompt string ready for tokenization.
    """
    domain_key = domain.lower().replace(" ", "_")
    domain_supplement = DOMAIN_SUPPLEMENTS.get(domain_key, "")

    # Start with the official master prompt
    parts = [OFFICIAL_MASTER_PROMPT]

    # Add parsed document context if available
    if parsed_context and parsed_context.strip():
        # Cap context to prevent OOM during tokenization
        MAX_CONTEXT_CHARS = 4000
        ctx = parsed_context[:MAX_CONTEXT_CHARS] if len(parsed_context) > MAX_CONTEXT_CHARS else parsed_context
        parts.append(f"\n\n### EXTRACTED DOCUMENT TEXT:\n{ctx}")

    # Add domain-specific supplement
    if domain_supplement:
        parts.append(domain_supplement)

    # Add the question
    parts.append(f"\n\n### QUESTION:\n{question}")

    return "\n".join(parts)
