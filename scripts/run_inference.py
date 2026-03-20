"""
CLI for running the full DocVQA inference pipeline.

Usage:
    python scripts/run_inference.py --split val
    python scripts/run_inference.py --split test --config configs/pipeline_config.yaml
    python scripts/run_inference.py --split val --domain maps  # single-domain run
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Ensure project root is on the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ICDAR 2026 DocVQA inference pipeline"
    )
    parser.add_argument(
        "--split", choices=["val", "test"], default="val",
        help="Dataset split to run inference on"
    )
    parser.add_argument(
        "--config", default="configs/pipeline_config.yaml",
        help="Path to pipeline YAML config"
    )
    parser.add_argument(
        "--domain", default=None,
        help="Run only on a specific domain (e.g., 'maps', 'comics')"
    )
    parser.add_argument(
        "--output-dir", default="data",
        help="Base output directory (predictions and submissions go here)"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit to N samples (useful for quick debugging)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info(f"Starting inference: split={args.split}, config={args.config}")

    # Load pipeline
    from src.pipeline.docvqa_pipeline import DocVQAPipeline
    pipeline = DocVQAPipeline.from_config(args.config)

    # Load dataset
    from src.data_loader import load_val_dataset, load_test_dataset
    if args.split == "val":
        dataset = load_val_dataset()
    else:
        dataset = load_test_dataset()

    # Optionally filter by domain
    if args.domain:
        logger.info(f"Filtering to domain: {args.domain}")
        dataset = dataset.filter(lambda x: x["doc_category"] == args.domain)

    # Optionally limit samples
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
        logger.info(f"Limited to {args.max_samples} samples")

    # Run inference
    results = pipeline.run_on_dataset(dataset, split=args.split)

    # Save outputs
    output_dir = Path(args.output_dir)
    suffix = f"_{args.domain}" if args.domain else ""

    predictions_path = output_dir / "predictions" / f"raw_predictions{suffix}.json"
    pipeline.save_predictions(results, str(predictions_path))
    logger.info(f"Raw predictions saved: {predictions_path}")

    submission_path = output_dir / "submissions" / f"submission{suffix}.json"
    pipeline.save_submission(results, str(submission_path))
    logger.info(f"Submission JSON saved: {submission_path}")

    # Quick stats
    n_unknown = sum(1 for r in results if r["answer"] == "Unknown")
    logger.info(f"Total answers: {len(results)} | Unknown: {n_unknown} ({n_unknown/len(results):.1%})")

    print(f"\n✅ Done! Files saved:")
    print(f"   Predictions : {predictions_path}")
    print(f"   Submission  : {submission_path}")
    print(f"\nNext step — run local evaluation:")
    print(f"   python scripts/evaluate_local.py {predictions_path}")


if __name__ == "__main__":
    main()
