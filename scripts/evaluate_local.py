"""
Local ANLS evaluation script.

Usage:
    python scripts/evaluate_local.py data/predictions/raw_predictions.json
    python scripts/evaluate_local.py data/predictions/raw_predictions.json --cache-dir data/raw/val
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import argparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Run local ANLS evaluation")
    parser.add_argument("predictions", help="Path to predictions JSON file")
    parser.add_argument("--cache-dir", default="data/raw/val",
                        help="Dataset cache directory")
    parser.add_argument("--no-official-eval", action="store_true",
                        help="Use local ANLS implementation instead of official eval_utils.py")
    args = parser.parse_args()

    from src.evaluation.local_evaluator import run_local_eval
    results = run_local_eval(
        predictions_path=args.predictions,
        cache_dir=args.cache_dir,
        use_official_eval=not args.no_official_eval,
    )

    overall = results["overall_anls"]
    if overall >= 0.65:
        emoji = "🏆"
    elif overall >= 0.50:
        emoji = "✅"
    elif overall >= 0.35:
        emoji = "⚠️"
    else:
        emoji = "❌"

    print(f"\n{emoji}  Overall ANLS: {overall:.4f}")
    print("\nNext steps:")
    if overall < 0.65:
        print("  → Run: python scripts/prepare_submission.py data/submissions/submission.json")
        print("  → Then submit to: https://rrc.cvc.uab.es/?ch=34&com=mymethods&task=1")


if __name__ == "__main__":
    main()
