"""
Unit tests for answer_formatter.py.

These tests run without any GPU or model — pure Python.
Every formatting rule has at least 2 test cases (positive + negative).

Run with: python -m pytest tests/test_formatter.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.reasoner.answer_formatter import extract_and_format_answer


# ---------------------------------------------------------------------------
# Date normalization
# ---------------------------------------------------------------------------

class TestDateNormalization:
    def test_slash_date_mm_dd_yyyy(self):
        raw = "FINAL ANSWER: 01/05/2024"
        assert extract_and_format_answer(raw) == "2024-01-05"

    def test_slash_date_two_digit_year(self):
        raw = "FINAL ANSWER: 1/5/24"
        assert extract_and_format_answer(raw) == "2024-01-05"

    def test_textual_date_full_month(self):
        raw = "FINAL ANSWER: January 5, 2024"
        assert extract_and_format_answer(raw) == "2024-01-05"

    def test_textual_date_abbreviated_month(self):
        raw = "FINAL ANSWER: Jan 5th 2024"
        assert extract_and_format_answer(raw) == "2024-01-05"

    def test_already_iso_date_unchanged(self):
        raw = "FINAL ANSWER: 2024-01-05"
        assert extract_and_format_answer(raw) == "2024-01-05"

    def test_partial_iso_zero_padded(self):
        raw = "FINAL ANSWER: 2024-1-5"
        assert extract_and_format_answer(raw) == "2024-01-05"

    def test_december(self):
        raw = "FINAL ANSWER: December 31, 2023"
        assert extract_and_format_answer(raw) == "2023-12-31"


# ---------------------------------------------------------------------------
# Number normalization
# ---------------------------------------------------------------------------

class TestNumberNormalization:
    def test_thousands_comma_removed(self):
        raw = "FINAL ANSWER: 1,000"
        assert extract_and_format_answer(raw) == "1000"

    def test_millions_comma_removed(self):
        raw = "FINAL ANSWER: 1,234,567"
        # Both commas should be removed
        result = extract_and_format_answer(raw)
        assert "," not in result

    def test_unit_space_added_kg(self):
        raw = "FINAL ANSWER: 50kg"
        assert extract_and_format_answer(raw) == "50 kg"

    def test_unit_space_added_km(self):
        raw = "FINAL ANSWER: 100km"
        assert extract_and_format_answer(raw) == "100 km"

    def test_unit_space_already_present_unchanged(self):
        raw = "FINAL ANSWER: 50 kg"
        assert extract_and_format_answer(raw) == "50 kg"

    def test_decimal_no_change(self):
        raw = "FINAL ANSWER: 3.14"
        assert extract_and_format_answer(raw) == "3.14"


# ---------------------------------------------------------------------------
# Percentage normalization
# ---------------------------------------------------------------------------

class TestPercentageNormalization:
    def test_space_before_percent_removed(self):
        raw = "FINAL ANSWER: 50 %"
        assert extract_and_format_answer(raw) == "50%"

    def test_no_space_already_correct(self):
        raw = "FINAL ANSWER: 50%"
        assert extract_and_format_answer(raw) == "50%"

    def test_decimal_percentage(self):
        raw = "FINAL ANSWER: 3.14 %"
        assert extract_and_format_answer(raw) == "3.14%"


# ---------------------------------------------------------------------------
# Filler text removal
# ---------------------------------------------------------------------------

class TestFillerTextRemoval:
    def test_the_answer_is_removed(self):
        raw = "FINAL ANSWER: The answer is Paris"
        assert extract_and_format_answer(raw) == "Paris"

    def test_based_on_removed(self):
        raw = "FINAL ANSWER: Based on the document, the value is 42"
        assert extract_and_format_answer(raw) == "the value is 42"

    def test_according_to_removed(self):
        raw = "FINAL ANSWER: According to the document, revenue was 1000"
        assert extract_and_format_answer(raw) == "revenue was 1000"

    def test_clean_answer_unchanged(self):
        raw = "FINAL ANSWER: Paris"
        assert extract_and_format_answer(raw) == "Paris"


# ---------------------------------------------------------------------------
# List separator normalization
# ---------------------------------------------------------------------------

class TestListSeparatorNormalization:
    def test_and_replaced_by_comma(self):
        raw = "FINAL ANSWER: red and blue"
        result = extract_and_format_answer(raw)
        assert "and" not in result
        assert ", " in result

    def test_multiple_ands(self):
        raw = "FINAL ANSWER: red and blue and green"
        result = extract_and_format_answer(raw)
        assert "and" not in result


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

class TestAnswerExtraction:
    def test_final_answer_prefix_extracted(self):
        raw = "Thinking...\nFINAL ANSWER: 42"
        assert extract_and_format_answer(raw) == "42"

    def test_no_prefix_last_line_used(self):
        raw = "The document says many things.\nThe number is 42."
        assert extract_and_format_answer(raw) == "The number is 42."

    def test_empty_output_returns_unknown(self):
        assert extract_and_format_answer("") == "Unknown"

    def test_only_whitespace_returns_unknown(self):
        assert extract_and_format_answer("   \n  \n  ") == "Unknown"

    def test_case_insensitive_prefix(self):
        raw = "final answer: done"
        assert extract_and_format_answer(raw) == "done"
