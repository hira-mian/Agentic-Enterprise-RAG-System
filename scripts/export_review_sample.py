"""Export a stratified sample of predictions as CSV for manual scoring with the
qualitative rubric (docs/QUALITATIVE_RUBRIC.md).

    python scripts/export_review_sample.py results/<run>/predictions.jsonl --per-type 10
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

RUBRIC_COLUMNS = ["correctness_0_2", "grounding_0_2", "multi_hop_coverage_0_2",
                  "abstention_0_2", "clarity_0_2", "failure_category", "notes"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("predictions", type=Path)
    p.add_argument("--per-type", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    rows = [json.loads(line) for line in args.predictions.open(encoding="utf-8")]
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["question_type"]].append(r)
    rng = random.Random(args.seed)
    sample = []
    for t in sorted(by_type):
        sample += rng.sample(by_type[t], min(args.per_type, len(by_type[t])))

    out = args.out or args.predictions.with_name("review_sample.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["qid", "question_type", "query", "gold_answer", "prediction",
                    "auto_accuracy", "gold_urls", "retrieved_urls_top_k"] + RUBRIC_COLUMNS)
        for r in sample:
            top = [c["url"] for c in r.get("retrieved_chunks", [])]
            w.writerow([r["qid"], r["question_type"], r["query"], r["gold_answer"], r.get("prediction"),
                        r.get("accuracy"), " | ".join(r["gold_urls"]), " | ".join(top)] + [""] * len(RUBRIC_COLUMNS))
    print(f"Wrote {len(sample)} rows to {out}")


if __name__ == "__main__":
    main()
