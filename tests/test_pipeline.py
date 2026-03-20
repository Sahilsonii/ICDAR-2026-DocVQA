"""
Integration tests for the DocVQA pipeline using mocked parsers and reasoners.

These tests verify pipeline orchestration logic without loading any real models.

Run with: python -m pytest tests/test_pipeline.py -v
"""

import sys
import json
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import MagicMock, patch


def make_synthetic_sample(doc_id="maps_001", domain="maps", n_questions=2):
    """Create a synthetic dataset sample for testing."""
    from PIL import Image
    pages = [Image.new("RGB", (100, 100), color="white")]
    question_ids = [f"{doc_id}_q{i}" for i in range(n_questions)]
    questions = [f"What is shown on page {i}?" for i in range(n_questions)]
    answers = [f"Answer {i}" for i in range(n_questions)]

    return {
        "doc_id": doc_id,
        "doc_category": domain,
        "document": pages,
        "questions": {
            "question_id": question_ids,
            "question": questions,
        },
        "answers": {
            "question_id": question_ids,
            "answer": answers,
        }
    }


class TestDocVQAPipeline:
    def setup_method(self):
        from src.pipeline.docvqa_pipeline import DocVQAPipeline
        from src.parser.parser_router import ParserRouter
        from src.reasoner.gemma_reasoner import GemmaReasoner

        # Mock both parser and reasoner
        self.mock_parser = MagicMock(spec=ParserRouter)
        self.mock_parser.parse.return_value = "Mocked parsed document text."

        self.mock_reasoner = MagicMock(spec=GemmaReasoner)
        self.mock_reasoner.answer.return_value = {
            "answer": "42",
            "full_answer": "FINAL ANSWER: 42"
        }

        self.pipeline = DocVQAPipeline(
            parser_router=self.mock_parser,
            reasoner=self.mock_reasoner,
            save_intermediates=False,
        )

    def test_single_sample_produces_results(self):
        sample = make_synthetic_sample(n_questions=2)
        results = self.pipeline.run_on_dataset([sample])
        assert len(results) == 2
        assert all(r["answer"] == "42" for r in results)

    def test_document_cached_after_first_parse(self):
        """Parser should only be called ONCE even if there are multiple questions."""
        sample = make_synthetic_sample(n_questions=5)
        self.pipeline.run_on_dataset([sample])
        # Parser called only once per document, not once per question
        assert self.mock_parser.parse.call_count == 1

    def test_multiple_docs_parsed_separately(self):
        samples = [
            make_synthetic_sample("maps_001", "maps", 2),
            make_synthetic_sample("science_001", "science_paper", 3),
        ]
        results = self.pipeline.run_on_dataset(samples)
        assert len(results) == 5  # 2 + 3 questions
        assert self.mock_parser.parse.call_count == 2  # One per document

    def test_inference_error_returns_unknown(self):
        """If reasoner raises an exception, answer should be 'Unknown'."""
        self.mock_reasoner.answer.side_effect = RuntimeError("GPU OOM")
        sample = make_synthetic_sample(n_questions=1)
        results = self.pipeline.run_on_dataset([sample])
        assert len(results) == 1
        assert results[0]["answer"] == "Unknown"
        assert "_error" in results[0]

    def test_parsing_error_uses_fallback_text(self):
        """If parsing fails, question answering should still proceed with error marker."""
        self.mock_parser.parse.side_effect = RuntimeError("Parser crashed")
        self.mock_reasoner.answer.return_value = {"answer": "fallback", "full_answer": "fallback"}
        sample = make_synthetic_sample(n_questions=1)
        results = self.pipeline.run_on_dataset([sample])
        assert len(results) == 1
        # Reasoner still called with the error text
        call_args = self.mock_reasoner.answer.call_args
        assert "PARSING FAILED" in call_args.kwargs["parsed_context"]

    def test_submission_json_strips_internal_fields(self, tmp_path):
        """Saved submission JSON must not contain internal _ prefixed fields."""
        sample = make_synthetic_sample(n_questions=2)
        results = self.pipeline.run_on_dataset([sample])

        submission_path = tmp_path / "submission.json"
        self.pipeline.save_submission(results, str(submission_path))

        with open(submission_path) as f:
            data = json.load(f)

        assert isinstance(data, list)
        for entry in data:
            assert "question_id" in entry
            assert "answer" in entry
            # No internal fields
            for key in entry:
                assert not key.startswith("_"), f"Internal field found: {key}"

    def test_result_question_ids_match_input(self):
        sample = make_synthetic_sample("maps_001", "maps", 3)
        expected_ids = sample["questions"]["question_id"]
        results = self.pipeline.run_on_dataset([sample])
        result_ids = [r["question_id"] for r in results]
        assert sorted(result_ids) == sorted(expected_ids)
