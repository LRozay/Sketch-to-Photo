"""Tally pairwise survey votes after raters fill the blank sheet.

Check-in 3 and Check-in 4 ask for a human preference between the pretrained
scribble baseline and the LoRA adapter. This script only counts votes that
already exist in the CSV. It does not invent preferences.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


ALLOWED = {"baseline", "lora", "tie"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize pairwise sketch-following survey votes")
    parser.add_argument("--survey", default="outputs/pairwise_survey.csv")
    parser.add_argument("--output", default="results/survey_summary.json")
    args = parser.parse_args()

    survey = Path(args.survey)
    if not survey.is_file():
        raise SystemExit(f"Missing {survey}. Build it with evaluate_generations.py survey first.")

    counts: Counter[str] = Counter()
    blank = 0
    invalid = 0
    rows = 0
    with survey.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if "follows_sketch_better" not in (reader.fieldnames or []):
            raise SystemExit("Survey CSV needs a follows_sketch_better column.")
        for row in reader:
            rows += 1
            vote = (row.get("follows_sketch_better") or "").strip().lower()
            if not vote:
                blank += 1
                continue
            if vote not in ALLOWED:
                invalid += 1
                continue
            counts[vote] += 1

    decided = counts["baseline"] + counts["lora"] + counts["tie"]
    payload = {
        "status": "pending_votes" if decided == 0 else "recorded",
        "survey_path": str(survey),
        "row_count": rows,
        "blank_votes": blank,
        "invalid_votes": invalid,
        "baseline_preferred": counts["baseline"],
        "lora_preferred": counts["lora"],
        "ties": counts["tie"],
        "decided_votes": decided,
        "lora_win_rate_among_decided": (
            None if decided == 0 else round(counts["lora"] / decided, 4)
        ),
        "notes": "Fill follows_sketch_better with baseline, lora, or tie before summarizing.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
