"""
Dataset download script.
Downloads the official DocVQA 2026 validation set and Comics supplemental dataset.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --split test  # Also download test set
"""

import argparse
import sys
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Download DocVQA 2026 datasets")
    parser.add_argument("--split", choices=["val", "test", "both"], default="val")
    parser.add_argument("--include-comics", action="store_true",
                        help="Also download ComicsPAP supplemental dataset")
    args = parser.parse_args()

    from datasets import load_dataset

    if args.split in ("val", "both"):
        logger.info("Downloading validation set...")
        val = load_dataset("VLR-CVC/DocVQA-2026", split="val")
        val.save_to_disk("data/raw/val")
        logger.info(f"Validation set saved: {len(val)} documents")
        logger.info(f"Domains: {set(val['doc_category'])}")

    if args.split in ("test", "both"):
        logger.info("Downloading test set...")
        test = load_dataset("VLR-CVC/DocVQA-2026", split="test")
        test.save_to_disk("data/raw/test")
        logger.info(f"Test set saved: {len(test)} documents")

    if args.include_comics:
        logger.info("Downloading ComicsPAP supplemental dataset...")
        comics = load_dataset("VLR-CVC/ComicsPAP", split="train")
        comics.save_to_disk("data/raw/comics_pap")
        logger.info(f"ComicsPAP saved: {len(comics)} examples")

    print("\n✅ Download complete!")
    print("   Next: python scripts/run_inference.py --split val")


if __name__ == "__main__":
    main()
