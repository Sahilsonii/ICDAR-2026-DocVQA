"""
Final submission validator — run this BEFORE uploading to the RRC portal.
https://rrc.cvc.uab.es/?ch=34&com=mymethods&task=1

Validates:
  1. Valid JSON format
  2. Root element is an array
  3. Each entry has question_id, answer, full_answer
  4. answer field is not empty or None
  5. answer does not contain filler phrases
  6. Dates are YYYY-MM-DD formatted (not MM/DD/YYYY)
  7. Percentages have no space before %
  8. No thousands commas (1,000 → should be 1000)

Usage:
    python scripts/prepare_submission.py data/submissions/submission.json
"""

import json
import re
import sys
from pathlib import Path


def validate_submission(json_path: str) -> bool:
    print(f"\n🔍 Validating: {json_path}")
    errors = []
    warnings = []

    # Check 1: File exists
    if not Path(json_path).exists():
        print(f"FATAL: File not found: {json_path}")
        return False

    # Check 2: Valid JSON
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"FATAL: Invalid JSON — {e}")
        return False

    # Check 3: Root is array
    if not isinstance(data, list):
        print("FATAL: Root element must be a JSON array (list)")
        return False

    if len(data) == 0:
        print("FATAL: Submission array is empty")
        return False

    # Per-entry validation
    filler_patterns = [
        r"^the answer is",
        r"^the final answer is",
        r"^based on",
        r"^according to",
        r"^from the document",
    ]
    date_wrong = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")
    percent_wrong = re.compile(r"\d+\s+%")
    thousands_wrong = re.compile(r"\d{1,3},\d{3}")

    for entry in data:
        qid = entry.get("question_id", "UNKNOWN")

        # Required fields
        if "question_id" not in entry:
            errors.append(f"[{qid}] Missing field: question_id")
        if "answer" not in entry:
            errors.append(f"[{qid}] Missing field: answer")
            continue
        if "full_answer" not in entry:
            warnings.append(f"[{qid}] Missing field: full_answer (optional)")

        answer = entry.get("answer", "")

        if not answer:
            errors.append(f"[{qid}] EMPTY answer field")
            continue

        # Internal fields check (should have been stripped)
        for field in ("_doc_id", "_domain", "_ground_truth", "_error"):
            if field in entry:
                warnings.append(f"[{qid}] Internal field '{field}' not stripped")

        # Content checks
        for pat in filler_patterns:
            if re.search(pat, answer, re.IGNORECASE):
                warnings.append(f"[{qid}] Filler text detected: '{answer[:80]}'")

        if date_wrong.search(answer):
            warnings.append(f"[{qid}] Date not in YYYY-MM-DD: '{answer}'")

        if percent_wrong.search(answer):
            errors.append(f"[{qid}] Space before %: '{answer}'")

        if thousands_wrong.search(answer):
            errors.append(f"[{qid}] Thousands comma: '{answer}'")

    print(f"   Total entries : {len(data)}")
    print(f"   Errors        : {len(errors)}")
    print(f"   Warnings      : {len(warnings)}")

    if errors:
        print("\n❌ ERRORS (must fix before submitting):")
        for e in errors[:25]:
            print(f"     {e}")
        if len(errors) > 25:
            print(f"     ... and {len(errors) - 25} more errors")

    if warnings:
        print("\n⚠️  WARNINGS (review these):")
        for w in warnings[:20]:
            print(f"     {w}")

    if not errors:
        print(f"\n✅ Submission is VALID — ready to upload!")
        print(f"   Upload at: https://rrc.cvc.uab.es/?ch=34&com=mymethods&task=1")
    else:
        print(f"\n❌ Submission has {len(errors)} error(s) — fix before uploading.")

    return len(errors) == 0


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/submissions/submission.json"
    ok = validate_submission(path)
    sys.exit(0 if ok else 1)
