"""Loading the MultiHop-RAG corpus and queries, and the fixed dev/test splits."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
SPLITS_DIR = REPO_ROOT / "data" / "splits"

CORPUS_FILE = RAW_DIR / "corpus.json"
QUERIES_FILE = RAW_DIR / "MultiHopRAG.json"

# Document metadata kept on every chunk. `url` is the unique document id used
# to match retrieved chunks against gold evidence.
DOC_METADATA_KEYS = ("title", "source", "author", "category", "published_at", "url")


@dataclass
class Query:
    qid: str
    query: str
    answer: str
    question_type: str
    evidence_urls: list[str] = field(default_factory=list)
    evidence_sources: list[str] = field(default_factory=list)


def _require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` first."
        )


def load_corpus() -> list[dict]:
    _require(CORPUS_FILE)
    with open(CORPUS_FILE, encoding="utf-8") as f:
        return json.load(f)


def load_queries() -> list[Query]:
    """Load all 2,556 queries. `qid` is the zero-padded position in the source file."""
    _require(QUERIES_FILE)
    with open(QUERIES_FILE, encoding="utf-8") as f:
        raw = json.load(f)
    queries = []
    for i, item in enumerate(raw):
        urls, sources = [], []
        for ev in item["evidence_list"]:
            if ev["url"] not in urls:
                urls.append(ev["url"])
            sources.append(ev["source"])
        queries.append(
            Query(
                qid=f"q{i:04d}",
                query=item["query"],
                answer=item["answer"],
                question_type=item["question_type"],
                evidence_urls=urls,
                evidence_sources=sources,
            )
        )
    return queries


def load_split(name: str) -> list[Query]:
    """Return the queries in a named split (`dev`, `test`, or `all`)."""
    queries = load_queries()
    if name == "all":
        return queries
    split_file = SPLITS_DIR / f"{name}.json"
    _require(split_file)
    with open(split_file, encoding="utf-8") as f:
        ids = set(json.load(f)["qids"])
    return [q for q in queries if q.qid in ids]


def load_documents():
    """Convert the corpus into LlamaIndex `Document`s (one per news article)."""
    from llama_index.core import Document

    docs = []
    for article in load_corpus():
        metadata = {k: (article.get(k) or "") for k in DOC_METADATA_KEYS}
        docs.append(Document(text=article["body"], metadata=metadata, id_=article["url"]))
    return docs
