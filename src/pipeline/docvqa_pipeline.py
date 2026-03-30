"""
Full DocVQA 2026 pipeline: loads data → parses → reasons → formats → saves.

Usage:
    pipeline = DocVQAPipeline.from_config("configs/pipeline_config.yaml")
    results = pipeline.run_on_dataset(dataset, split="val")
    pipeline.save_submission(results, "data/submissions/submission_v1.json")
"""

import json
import logging
from pathlib import Path
from tqdm import tqdm
from typing import List

from src.data_loader import get_all_qa_pairs
from src.parser.parser_router import ParserRouter

logger = logging.getLogger(__name__)


class DocVQAPipeline:
    """
    End-to-end DocVQA pipeline.

    Implements document caching: each document is parsed only once, even if it
    has multiple questions. Parsed text is stored in doc_cache keyed by doc_id.
    """

    def __init__(
        self,
        parser_router: ParserRouter,
        reasoner,
        save_intermediates: bool = False,
        processed_dir: str = "data/processed",
    ):
        self.parser = parser_router
        self.reasoner = reasoner
        self.save_intermediates = save_intermediates
        self.processed_dir = Path(processed_dir)
        if save_intermediates:
            self.processed_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(cls, config_path: str) -> "DocVQAPipeline":
        """Create pipeline from a YAML configuration file."""
        import yaml
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        parser = ParserRouter(config.get("parsers", {}))

        reasoner_cfg = config.get("reasoner", {})
        backend = reasoner_cfg.get("backend", "hf_api")

        if backend == "hf_api":
            from src.reasoner.hf_api_reasoner import HuggingFaceAPIReasoner
            reasoner = HuggingFaceAPIReasoner(reasoner_cfg.get("hf_api", {}))
        else:
            from src.reasoner.gemma_reasoner import GemmaReasoner
            reasoner = GemmaReasoner(reasoner_cfg.get("gemma_local", {}))

        pipeline_cfg = config.get("pipeline", {})

        return cls(
            parser_router=parser,
            reasoner=reasoner,
            save_intermediates=pipeline_cfg.get("save_intermediates", True),
            processed_dir=config.get("output", {}).get("processed_path", "data/processed"),
        )

    def run_on_dataset(self, dataset, split: str = "val") -> List[dict]:
        """
        Run the full pipeline on every QA pair in the dataset.

        Returns list of result dicts with question_id, answer, full_answer,
        and internal debugging fields (_doc_id, _domain, etc.).
        """
        results = []
        doc_cache: dict = {}

        for sample in tqdm(dataset, desc=f"Processing {split}"):
            doc_id = sample["doc_id"]
            domain = sample["doc_category"]
            pages = sample["document"]

            for qa in get_all_qa_pairs(sample):
                # Parse document (cached per doc_id)
                if doc_id not in doc_cache:
                    try:
                        parsed_text = self.parser.parse(pages, domain)
                        doc_cache[doc_id] = parsed_text
                        if self.save_intermediates:
                            self._save_parsed(doc_id, parsed_text)
                    except Exception as e:
                        logger.error(f"Parsing failed for {doc_id}: {e}")
                        doc_cache[doc_id] = f"[PARSING FAILED: {e}]"

                parsed_text = doc_cache[doc_id]

                try:
                    response = self.reasoner.answer(
                        pages=pages,
                        question=qa["question"],
                        parsed_context=parsed_text,
                        domain=domain,
                    )
                except Exception as e:
                    logger.error(f"Inference failed for {qa['question_id']}: {e}")
                    response = {"answer": "Unknown", "full_answer": str(e)}

                results.append({
                    "question_id": qa["question_id"],
                    "answer": response.get("answer", "Unknown"),
                    "full_answer": response.get("full_answer", ""),
                    "_doc_id": doc_id,
                    "_domain": domain,
                    "_question": qa["question"],
                    # NOTE: Ground truth is NOT attached to predictions.
                    # Evaluation must use a separate GT lookup from the dataset
                    # to prevent data leakage.
                })

        logger.info(f"Pipeline complete: {len(results)} answers generated")
        return results

    def save_submission(self, results: List[dict], output_path: str):
        """Save results as competition-valid JSON (strips internal fields)."""
        submission = [
            {
                "question_id": r["question_id"],
                "answer": r["answer"],
                "full_answer": r.get("full_answer", ""),
            }
            for r in results
        ]
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(submission, f, indent=2, ensure_ascii=False)
        logger.info(f"Submission saved to {output_path} ({len(submission)} answers)")

    def save_predictions(self, results: List[dict], output_path: str):
        """Save full results (including internal fields) for local evaluation."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Full predictions saved to {output_path}")

    def _save_parsed(self, doc_id: str, text: str):
        """Save parsed document text to data/processed/ for inspection."""
        out_file = self.processed_dir / f"{doc_id}.md"
        out_file.write_text(text, encoding="utf-8")
