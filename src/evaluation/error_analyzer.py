"""
Error analyzer — identify which domains, question types, and patterns fail most.

Run after local evaluation to understand where to spend improvement effort.
Results guide which domains to prioritize in ablation study (notebook 05).
"""

import json
import re
from collections import Counter, defaultdict
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


def analyze_errors(predictions_path: str, gt_map: dict, domain_map: dict):
    """
    Analyze prediction errors to find systematic failure patterns.

    Args:
        predictions_path: Path to full predictions JSON (with _ground_truth field).
        gt_map: dict of {question_id: ground_truth_answer}
        domain_map: dict of {question_id: domain}

    Prints a structured error analysis report.
    """
    with open(predictions_path) as f:
        predictions = json.load(f)

    errors_by_domain: dict = defaultdict(list)
    error_types: Counter = Counter()
    total_by_domain: Counter = Counter()

    for pred in predictions:
        qid = pred["question_id"]
        answer = pred.get("answer", "Unknown")
        gt = gt_map.get(qid, "")
        domain = pred.get("_domain") or domain_map.get(qid, "unknown")
        total_by_domain[domain] += 1

        if not _is_correct(answer, gt):
            error_type = _classify_error(answer, gt, pred.get("_question", ""))
            errors_by_domain[domain].append({
                "qid": qid,
                "predicted": answer,
                "ground_truth": gt,
                "error_type": error_type,
                "question": pred.get("_question", ""),
            })
            error_types[error_type] += 1

    print("\n" + "="*60)
    print("  ERROR ANALYSIS REPORT")
    print("="*60)
    print("\n📊 Error rate by domain:")
    for domain in sorted(total_by_domain.keys()):
        n_errors = len(errors_by_domain[domain])
        n_total = total_by_domain[domain]
        rate = n_errors / n_total if n_total > 0 else 0
        bar = "█" * int(rate * 20) + "░" * (20 - int(rate * 20))
        print(f"  {domain:<25} {bar} {rate:.1%} ({n_errors}/{n_total})")

    print("\n🏷️  Error types (top 10):")
    for error_type, count in error_types.most_common(10):
        print(f"  {error_type:<35} {count}")

    print("\n❌ Sample errors per domain (top 3 per domain):")
    for domain, errors in sorted(errors_by_domain.items()):
        print(f"\n  [{domain.upper()}]")
        for err in errors[:3]:
            print(f"    Q: {err['question'][:80]}")
            print(f"    Pred: {err['predicted'][:60]}  |  GT: {err['ground_truth'][:60]}")
            print(f"    Type: {err['error_type']}")
    print("="*60)


def _is_correct(prediction: str, ground_truth: str, threshold: float = 0.5) -> bool:
    """Quick correctness check using edit distance."""
    try:
        from rapidfuzz.distance import Levenshtein
        if not ground_truth:
            return False
        pred = prediction.lower().strip()
        gt = ground_truth.lower().strip()
        dist = Levenshtein.distance(pred, gt)
        similarity = 1.0 - dist / max(len(pred), len(gt), 1)
        return similarity >= threshold
    except ImportError:
        return prediction.lower().strip() == ground_truth.lower().strip()


def _classify_error(prediction: str, ground_truth: str, question: str) -> str:
    """Heuristically classify the type of error."""
    pred = prediction.lower().strip()
    gt = ground_truth.lower().strip()

    if pred == "unknown":
        return "model_abstained"
    if not pred:
        return "empty_prediction"
    if re.search(r"\b(the answer is|based on|according to)\b", pred):
        return "filler_text_not_removed"
    if re.search(r"\d{1,2}/\d{1,2}/\d{2,4}", pred) and re.search(r"\d{4}-\d{2}-\d{2}", gt):
        return "date_format_wrong"
    if re.search(r"\d,\d{3}", pred) and not re.search(r"\d,\d{3}", gt):
        return "thousands_comma_not_removed"
    if re.search(r"\d\s+%", pred) and re.search(r"\d%", gt):
        return "percentage_space_wrong"
    if "what" in question.lower() and any(w in pred for w in ["who", "where", "when"]):
        return "wrong_entity_type"
    if len(pred) > len(gt) * 3:
        return "answer_too_verbose"
    if len(pred) < len(gt) // 3:
        return "answer_too_short"
    return "content_mismatch"
