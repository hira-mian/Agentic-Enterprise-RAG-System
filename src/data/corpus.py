"""Build the full EnterpriseRAG-Bench corpus as Parquet from the pinned upstream repo."""

import argparse
import json
import os
import shutil
import subprocess
from collections import Counter
from hashlib import sha256
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.config import CORPUS_COMMIT, CORPUS_DIR, CORPUS_REPO_URL, ROOT

SCHEMA = pa.schema(
    [
        ("doc_id", pa.string()),
        ("source_type", pa.string()),
        ("title", pa.string()),
        ("content", pa.string()),
        ("path", pa.string()),
    ]
)


def extract_document(data: dict) -> tuple[str, str]:
    """Mirror upstream `extract_document_content`: labeled title and content fields."""
    title_field = data.get("title_field_name")
    fields = data.get("content_field_names")
    if title_field not in data or not isinstance(fields, list) or not fields:
        raise ValueError("Missing or invalid title/content field labels")
    if any(name not in data for name in fields):
        raise ValueError("Content field label points to a missing field")
    if len(fields) == 1:
        return str(data[title_field]), str(data[fields[0]])
    parts = []
    for name in fields:
        value = data[name]
        if isinstance(value, list):
            value = "\n".join(str(v) for v in value)
        parts.append(f"{name}:\n{value}")
    return str(data[title_field]), "\n\n".join(parts)


def source_files(sources_dir: Path) -> list[Path]:
    files = []
    for root, dirs, names in os.walk(sources_dir):
        dirs.sort()
        files.extend(Path(root) / n for n in sorted(names) if n.endswith(".json"))
    return files


def ensure_repo(repo: Path, commit: str, clone: bool) -> None:
    if not repo.exists():
        if not clone:
            raise SystemExit(f"{repo} not found; pass --clone to download it")
        subprocess.run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                CORPUS_REPO_URL,
                str(repo),
            ],
            check=True,
        )
        subprocess.run(["git", "-C", str(repo), "checkout", commit], check=True)
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != commit:
        raise SystemExit(f"Corpus repo is at {head}, expected pinned {commit}")


def build_corpus(repo: Path, output: Path, batch_size: int = 5000) -> dict:
    sources_dir = repo / "generated_data" / "sources"
    files = source_files(sources_dir)
    # A few IDs are reused by different files; upstream's index names the canonical one.
    canonical = json.loads((repo / "generated_data" / "uuid_index.json").read_text())
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".tmp")
    digest, counts, skipped, seen = sha256(), Counter(), [], set()
    batch = {name: [] for name in SCHEMA.names}
    with pq.ParquetWriter(tmp, SCHEMA, compression="zstd") as writer:

        def flush():
            if batch["doc_id"]:
                writer.write_table(pa.table(batch, schema=SCHEMA))
                for values in batch.values():
                    values.clear()

        for path in files:
            relative = path.relative_to(sources_dir).as_posix()
            data = json.loads(path.read_text())
            try:
                title, content = extract_document(data)
                doc_id = data["dataset_doc_uuid"]
            except (KeyError, ValueError) as error:
                skipped.append({"path": relative, "reason": str(error)})
                continue
            if canonical.get(doc_id) != relative:
                skipped.append(
                    {"path": relative, "reason": f"not canonical for {doc_id}"}
                )
                continue
            if doc_id in seen:
                raise ValueError(f"Duplicate document ID: {doc_id}")
            seen.add(doc_id)
            source = relative.split("/", 1)[0]
            counts[source] += 1
            for value in (doc_id, source, title, content):
                digest.update(value.encode() + b"\0")
            for name, value in zip(
                SCHEMA.names, (doc_id, source, title, content, relative)
            ):
                batch[name].append(value)
            if len(batch["doc_id"]) >= batch_size:
                flush()
        flush()
    tmp.replace(output)
    if missing := set(canonical) - seen:
        raise ValueError(f"{len(missing)} indexed documents were not exported")
    return {
        "documents": len(seen),
        "files": len(files),
        "skipped": skipped,
        "source_counts": dict(sorted(counts.items())),
        "content_sha256": digest.hexdigest(),
        "doc_ids": seen,
    }


def question_coverage(questions: list[dict], doc_ids: set[str]) -> dict:
    referenced = {d for q in questions for d in q["expected_doc_ids"]}
    return {
        "questions": len(questions),
        "questions_without_references": sum(
            not q["expected_doc_ids"] for q in questions
        ),
        "referenced_documents": len(referenced),
        "referenced_documents_missing": sorted(referenced - doc_ids),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=CORPUS_DIR / "EnterpriseRAG-Bench")
    parser.add_argument("--output", type=Path, default=CORPUS_DIR / "documents.parquet")
    parser.add_argument("--clone", action="store_true", help="Clone if missing")
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "evaluation" / "corpus_manifest.json"
    )
    args = parser.parse_args()
    ensure_repo(args.repo, CORPUS_COMMIT, args.clone)
    print("Converting source documents to Parquet...", flush=True)
    report = build_corpus(args.repo, args.output)
    shutil.copyfile(
        args.repo / "questions.jsonl", args.output.parent / "questions.jsonl"
    )
    questions = [
        json.loads(line)
        for line in (args.repo / "questions.jsonl").read_text().splitlines()
        if line.strip()
    ]
    manifest = {
        "version": 1,
        "repo": CORPUS_REPO_URL,
        "commit": CORPUS_COMMIT,
        "extraction": "upstream title_field_name/content_field_names labels",
        **{k: v for k, v in report.items() if k != "doc_ids"},
        "coverage": question_coverage(questions, report["doc_ids"]),
        "questions_sha256": sha256(
            (args.repo / "questions.jsonl").read_bytes()
        ).hexdigest(),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "skipped"}, indent=2))


if __name__ == "__main__":
    main()
