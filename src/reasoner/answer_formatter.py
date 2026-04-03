"""
Answer post-processing: enforces all competition formatting rules on raw model output.

Rules from: https://github.com/VLR-CVC/DocVQA2026/blob/main/README.md

This is one of the most important files — incorrect formatting costs real ANLS points.
Every rule is tested in tests/test_formatter.py. Add new tests when you add new rules.

Formatting rules implemented:
  1. Extract answer after "FINAL ANSWER:" prefix (or last non-empty line as fallback)
  2. Normalize dates → YYYY-MM-DD
  3. Remove thousands comma separators (1,000 → 1000)
  4. Ensure space between number and unit (50kg → 50 kg)
  5. Remove space before % (50 % → 50%)
  6. Remove common model filler phrases
  7. Replace " and " list separators with ", "
"""

import re
from datetime import datetime
from typing import Optional


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_and_format_answer(raw_output: str) -> str:
    """
    Extract the final answer from model output and apply all formatting rules.

    Args:
        raw_output: Raw string from the language model.

    Returns:
        Clean, competition-formatted answer string. Returns "Unknown" if empty.
    """
    answer = _extract_answer(raw_output)
    answer = _normalize_dates(answer)
    answer = _normalize_numbers(answer)
    answer = _normalize_percentages(answer)
    answer = _remove_filler_text(answer)
    answer = _normalize_list_separators(answer)
    answer = answer.strip()
    return answer if answer else "Unknown"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_answer(raw_output: str) -> str:
    """Extract the answer portion from raw model output.
    
    Uses the same extraction method as the official evaluator:
    split on "FINAL ANSWER:" and take the last part.
    """
    marker = "FINAL ANSWER:"
    
    # Primary: official split method (case-insensitive)
    upper = raw_output.upper()
    if marker in upper:
        # Find position in the original string to preserve casing
        idx = upper.rfind(marker)
        answer = raw_output[idx + len(marker):].strip()
        # Take first line only (stop at newline) to avoid trailing reasoning
        answer = answer.split("\n")[0].strip()
        if answer:
            return answer

    # Secondary: look for "Answer:" prefix (common Qwen output pattern)
    for prefix in ["Answer:", "ANSWER:", "answer:"]:
        if prefix in raw_output:
            idx = raw_output.rfind(prefix)
            answer = raw_output[idx + len(prefix):].strip()
            answer = answer.split("\n")[0].strip()
            if answer:
                return answer

    # Fallback: take the last non-empty line (model might just output the answer directly)
    lines = [l.strip() for l in raw_output.strip().split("\n") if l.strip()]
    if lines:
        # Return the last line, which is most likely the answer
        return lines[-1]
    
    return "Unknown"


def _normalize_dates(text: str) -> str:
    """Convert date formats to YYYY-MM-DD."""
    # "Jan 1st 2024", "January 1, 2024", "Jan 1, 2024"
    text = re.sub(
        r"\b([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s*(\d{4})\b",
        lambda m: _parse_month_day_year(m.group(1), m.group(2), m.group(3)),
        text
    )
    # "01/01/2024" or "1/1/24" — assume MM/DD/YYYY
    text = re.sub(
        r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b",
        lambda m: _parse_slash_date(m.group(1), m.group(2), m.group(3)),
        text
    )
    # "2024-1-5" → "2024-01-05" (zero-pad partial ISO dates)
    text = re.sub(
        r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b",
        lambda m: f"{m.group(1)}-{m.group(2).zfill(2)}-{m.group(3).zfill(2)}",
        text
    )
    return text


def _normalize_numbers(text: str) -> str:
    """Remove thousands commas; ensure space between number and unit."""
    # Remove thousands separators: 1,000 → 1000 (but not decimal commas: 3,14)
    # Apply in a loop to handle multi-group numbers: 1,234,567 → 1234567
    prev = None
    while prev != text:
        prev = text
        text = re.sub(r"(\d),(\d{3})\b", r"\1\2", text)
    # Ensure space between number and unit abbreviation
    units = r"(kg|g|mg|t|lb|oz|m|km|cm|mm|mi|ft|in|yd|L|ml|l|USD|EUR|GBP|JPY|CNY|INR|W|kW|MW|GW|V|A|Hz|MHz|GHz)"
    text = re.sub(rf"(\d)({units})(?!\w)", r"\1 \2", text, flags=re.IGNORECASE)
    return text


def _normalize_percentages(text: str) -> str:
    """Remove space between number and %: '50 %' → '50%'."""
    return re.sub(r"(\d+\.?\d*)\s+%", r"\1%", text)


def _remove_filler_text(text: str) -> str:
    """Strip common model preamble/filler phrases."""
    fillers = [
        r"^the answer is\s*:?\s*",
        r"^the final answer is\s*:?\s*",
        r"^based on the document,?\s*",
        r"^based on the provided document,?\s*",
        r"^according to the document,?\s*",
        r"^from the document,?\s*",
        r"^looking at the document,?\s*",
        r"^the document states?\s*:?\s*",
        r"^it states?\s*:?\s*",
    ]
    for filler in fillers:
        text = re.sub(filler, "", text, flags=re.IGNORECASE)
    return text.strip()


def _normalize_list_separators(text: str) -> str:
    """Replace ' and ' between distinct items with ', '."""
    # Only replace 'and' flanked by whitespace (not mid-word)
    text = re.sub(r"\s+and\s+", ", ", text)
    return text


# ---------------------------------------------------------------------------
# Date parsing utilities
# ---------------------------------------------------------------------------

_MONTH_ABBREVS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    "january": 1, "february": 2, "march": 3, "april": 4, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10,
    "november": 11, "december": 12,
}


def _parse_month_day_year(month_str: str, day: str, year: str) -> str:
    """Convert 'January 5 2024' or 'Jan 5 2024' → '2024-01-05'."""
    month_num = _MONTH_ABBREVS.get(month_str.lower())
    if month_num is None:
        # Unknown month name — return year only as safe fallback
        return year
    return f"{year}-{str(month_num).zfill(2)}-{day.zfill(2)}"


def _parse_slash_date(month: str, day: str, year: str) -> str:
    """Convert MM/DD/YY or MM/DD/YYYY → YYYY-MM-DD."""
    if len(year) == 2:
        year = "20" + year if int(year) < 50 else "19" + year
    return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
