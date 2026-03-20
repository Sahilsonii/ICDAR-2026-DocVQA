"""
Local ANLS evaluator — run before submitting to https://rrc.cvc.uab.es

Uses the official eval_utils.py from: https://github.com/VLR-CVC/DocVQA2026
You must copy eval_utils.py there before running this script:
    git clone https://github.com/VLR-CVC/DocVQA2026.git
    cp DocVQA2026/eval_utils.py src/evaluation/eval_utils.py

ANLS (Average Normalized Levenshtein Similarity) is the official metric:
  - 1.0 = perfect match
  - 0.0 = completely wrong or "Unknown"
  - Penalizes string edits proportionally
"""

import json
import sys
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def compute_anls(prediction: str, ground_truth: str, threshold: float = 0.5) -> float:
    """
    Compute ANLS score between a prediction and ground truth.

    This is a local re-implementation for when eval_utils.py is not yet downloaded.
    The official eval_utils.py from the competition repo should be preferred.

    Args:
        prediction: Model's predicted answer.
        ground_truth: Reference answer string.
        threshold: ANLS threshold (default 0.5 per DocVQA convention).

    Returns:
        Float in [0, 1].
    """
    try:
        from rapidfuzz.distance import Levenshtein
    except ImportError:
        raise ImportError("rapidfuzz required: pip install rapidfuzz>=3.9.0")

    if not prediction or not ground_truth:
        return 0.0

    pred = prediction.lower().strip()
    gt = ground_truth.lower().strip()

    if not pred or not gt:
        return 0.0

    edit_dist = Levenshtein.distance(pred, gt)
    max_len = max(len(pred), len(gt))
    similarity = 1.0 - (edit_dist / max_len)

    return similarity if similarity >= threshold else 0.0


def run_local_eval(
    predictions_path: str,
    dataset=None,
    cache_dir: str = "data/raw/val",
    use_official_eval: bool = True,
) -> dict:
    """
    Evaluate predictions against validation ground truth.

    Args:
        predictions_path: Path to JSON file with predictions.
        dataset: Optional pre-loaded HuggingFace dataset. Loaded from cache if None.
        cache_dir: Where to load/cache the val dataset from.
        use_official_eval: If True, try to import official eval_utils.py first.

    Returns:
        Dict with overall ANLS score and per-domain breakdown.
    """
    with open(predictions_path, encoding="utf-8") as f:
        predictions = json.load(f)

    # Build prediction lookup
    pred_map = {p["question_id"]: p["answer"] for p in predictions}

    # Load dataset if not provided
    if dataset is None:
        sys.path.insert(0, "src")
        from src.data_loader import load_val_dataset
        dataset = load_val_dataset(cache_dir)

    # Build ground truth lookup and domain mapping
    gt_map = {}
    domain_map = {}
    for sample in dataset:
        domain = sample["doc_category"]
        for qid, ans in zip(
            sample["answers"]["question_id"],
            sample["answers"]["answer"]
        ):
            gt_map[qid] = ans
            domain_map[qid] = domain

    # Choose evaluator
    if use_official_eval:
        try:
            sys.path.insert(0, "src/evaluation")
            from eval_utils import evaluate_predictions as official_anls
            score_fn = lambda pred, gt: official_anls(pred, gt)
            logger.info("Using official eval_utils.py")
        except ImportError:
            logger.warning(
                "eval_utils.py not found. Using local ANLS implementation. "
                "Run: cp DocVQA2026/eval_utils.py src/evaluation/eval_utils.py"
            )
            score_fn = compute_anls
    else:
        score_fn = compute_anls

    # Evaluate
    domain_scores: dict = {}
    total_scores = []
    missing = 0

    for qid, gt in gt_map.items():
        pred = pred_map.get(qid, "Unknown")
        if qid not in pred_map:
            missing += 1
        score = score_fn(pred, gt)
        total_scores.append(score)
        domain = domain_map.get(qid, "unknown")
        domain_scores.setdefault(domain, []).append(score)

    overall = sum(total_scores) / len(total_scores) if total_scores else 0.0

    # Print results
    print(f"\n{'='*55}")
    print(f"  LOCAL EVALUATION RESULTS")
    print(f"{'='*55}")
    print(f"  Predictions evaluated : {len(total_scores)}")
    print(f"  Missing predictions   : {missing}")
    print(f"  OVERALL ANLS          : {overall:.4f}")
    print(f"{'='*55}")
    for domain, scores in sorted(domain_scores.items()):
        avg = sum(scores) / len(scores)
        print(f"  {domain:<25} {avg:.4f}  ({len(scores)} questions)")
    print(f"{'='*55}\n")

    return {
        "overall_anls": overall,
        "per_domain": {d: sum(s)/len(s) for d, s in domain_scores.items()},
        "total_questions": len(total_scores),
        "missing_predictions": missing,
    }


if __name__ == "__main__":
    import argparse
    parser_args = argparse.ArgumentParser(description="Local ANLS evaluation")
    parser_args.add_argument("predictions", help="Path to predictions JSON")
    parser_args.add_argument("--cache-dir", default="data/raw/val")
    parser_args.add_argument("--no-official-eval", action="store_true")
    args = parser_args.parse_args()

    run_local_eval(
        predictions_path=args.predictions,
        cache_dir=args.cache_dir,
        use_official_eval=not args.no_official_eval,
    )
