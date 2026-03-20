"""
Loads the DocVQA 2026 dataset and provides structured access per document.
Dataset: https://huggingface.co/datasets/VLR-CVC/DocVQA-2026

Each sample has:
  - doc_id: str (e.g., "maps_2")
  - doc_category: str (one of 8 domains)
  - document: List[PIL.Image] (one per page)
  - questions: {"question_id": [...], "question": [...]}
  - answers: {"question_id": [...], "answer": [...]} (only in val split)
"""

from datasets import load_from_disk, load_dataset
from pathlib import Path
from typing import Generator, List, Optional
import logging

logger = logging.getLogger(__name__)

DOMAIN_LIST = [
    "business_report", "comics", "engineering_drawing",
    "infographics", "maps", "science_paper",
    "science_poster", "slide"
]


def load_val_dataset(cache_dir: str = "data/raw/val"):
    """Load the validation set with public answers for local evaluation."""
    path = Path(cache_dir)
    if path.exists():
        logger.info(f"Loading cached dataset from {cache_dir}")
        return load_from_disk(str(path))
    logger.info("Downloading dataset from HuggingFace...")
    dataset = load_dataset("VLR-CVC/DocVQA-2026", split="val")
    dataset.save_to_disk(str(path))
    return dataset


def load_test_dataset(cache_dir: str = "data/raw/test"):
    """Load the test set (no public answers — for final submission)."""
    path = Path(cache_dir)
    if path.exists():
        logger.info(f"Loading cached test dataset from {cache_dir}")
        return load_from_disk(str(path))
    logger.info("Downloading test dataset from HuggingFace...")
    dataset = load_dataset("VLR-CVC/DocVQA-2026", split="test")
    dataset.save_to_disk(str(path))
    return dataset


def iter_samples_by_domain(dataset, domain: str) -> Generator:
    """Iterate only over samples from a specific document domain."""
    for sample in dataset:
        if sample["doc_category"] == domain:
            yield sample


def get_all_qa_pairs(sample: dict) -> List[dict]:
    """
    Flatten a sample into a list of (question_id, question, answer) dicts.
    For the test set, 'answer' will be None.
    """
    pairs = []
    questions = sample["questions"]
    answers = sample.get("answers", {})
    ans_map = dict(zip(
        answers.get("question_id", []),
        answers.get("answer", [])
    ))
    for qid, q in zip(questions["question_id"], questions["question"]):
        pairs.append({
            "doc_id": sample["doc_id"],
            "doc_category": sample["doc_category"],
            "question_id": qid,
            "question": q,
            "ground_truth": ans_map.get(qid),
            "pages": sample["document"]  # List of PIL Images
        })
    return pairs


def get_domain_statistics(dataset) -> dict:
    """Compute per-domain question counts for dataset overview."""
    stats = {d: {"docs": set(), "questions": 0} for d in DOMAIN_LIST}
    for sample in dataset:
        domain = sample["doc_category"]
        if domain in stats:
            stats[domain]["docs"].add(sample["doc_id"])
            stats[domain]["questions"] += len(sample["questions"]["question_id"])
    return {
        d: {"docs": len(v["docs"]), "questions": v["questions"]}
        for d, v in stats.items()
    }
