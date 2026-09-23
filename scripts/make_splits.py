"""Create the fixed dev/test query splits (stratified by question type, seed 42).

There is no training split: every system in this project is used zero-shot, so
the queries are divided into a dev split (prompt/parameter tuning, error
analysis) and a held-out test split (reported numbers only). The corpus is
shared by both splits, as in the original benchmark.
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from enterprise_rag.data import SPLITS_DIR, load_queries  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    by_type = defaultdict(list)
    for q in load_queries():
        by_type[q.question_type].append(q.qid)

    rng = random.Random(args.seed)
    dev, test = [], []
    for qtype in sorted(by_type):
        ids = sorted(by_type[qtype])
        rng.shuffle(ids)
        n_dev = round(len(ids) * args.dev_fraction)
        dev += ids[:n_dev]
        test += ids[n_dev:]

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    meta = {"seed": args.seed, "dev_fraction": args.dev_fraction, "stratified_by": "question_type"}
    for name, ids in (("dev", dev), ("test", test)):
        with open(SPLITS_DIR / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump({**meta, "qids": sorted(ids)}, f, indent=0)
        print(f"{name}: {len(ids)} queries")


if __name__ == "__main__":
    main()
