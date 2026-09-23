"""Create a pinned, bounded inspection corpus and evaluation split manifests."""
import argparse
from collections import Counter
from hashlib import sha256
from itertools import islice
import json
from pathlib import Path

from datasets import load_dataset

from src.config import DATASET_ID, DATASET_REVISION, ROOT
from src.evaluation.splits import build_splits


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def profile(rows: list[dict], id_field: str):
    fields = sorted({key for row in rows for key in row})
    return {"rows": len(rows), "fields": fields,
            "duplicate_ids": len(rows)-len({r[id_field] for r in rows}),
            "empty_or_null": {k: sum(r.get(k) in (None, "", []) for r in rows) for k in fields}}


def audit(documents, questions, revision):
    manifest = build_splits(questions)
    manifest.update(dataset_id=DATASET_ID, dataset_revision=revision)
    doc_ids = {d["doc_id"] for d in documents}
    dev_ids = set(manifest["splits"]["development"])
    coverage = []
    for q in questions:
        if q["question_id"] in dev_ids:
            expected = set(q["expected_doc_ids"])
            coverage.append({"question_id": q["question_id"], "expected": len(expected),
                             "present": len(expected & doc_ids),
                             "missing_doc_ids": sorted(expected-doc_ids)})
    report = {"dataset_id": DATASET_ID, "dataset_revision": revision,
              "selection": "first N documents in pinned upstream order; inspection only, not representative",
              "documents": profile(documents, "doc_id"),
              "questions": profile(questions, "question_id"),
              "document_source_counts": dict(Counter(d["source_type"] for d in documents)),
              "development_reference_coverage": coverage,
              "document_ids_sha256": sha256("\n".join(sorted(doc_ids)).encode()).hexdigest()}
    return report, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=int, default=256)
    parser.add_argument("--revision", default=DATASET_REVISION)
    parser.add_argument("--output", type=Path, default=ROOT / ".cache" / "audit")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "evaluation")
    args = parser.parse_args()
    if args.documents < 1:
        parser.error("--documents must be positive")
    common = dict(path=DATASET_ID, split="test", streaming=True,
                  revision=args.revision, cache_dir=str(ROOT / ".cache" / "huggingface"))
    print("Reading question metadata and bounded document sample...", flush=True)
    questions = list(load_dataset(name="questions", **common))
    documents = list(islice(load_dataset(name="documents", **common), args.documents))
    report, manifest = audit(documents, questions, args.revision)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in [("documents", documents), ("questions", questions)]:
        (args.output / f"{name}.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    write_json(args.artifacts / "data_audit.json", report)
    write_json(args.artifacts / "splits.json", manifest)
    # Only development examples are exported for inspection, never held-out answers.
    wanted = set(manifest["golden_development_ids"])
    examples = [q for q in questions if q["question_id"] in wanted]
    write_json(args.output / "golden_development.json", examples)
    seen = set()
    previews = []
    for q in sorted(examples, key=lambda q: q["question_id"]):
        if q["question_type"] not in seen:
            seen.add(q["question_type"])
            previews.append({k:q[k] for k in ("question_id", "question_type", "question", "expected_doc_ids")})
    write_json(args.artifacts / "development_examples.json", previews)
    print(json.dumps({"documents": len(documents), "questions": len(questions),
                      "splits": {k:len(v) for k,v in manifest["splits"].items()},
                      "golden_examples":len(examples)}, indent=2))


if __name__ == "__main__":
    main()
