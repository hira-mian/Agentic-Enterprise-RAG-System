import numpy as np
import pytest

from src.contracts import Document, SearchRequest, UserContext
from src.data.preprocessing import ChunkingConfig, chunk_document, chunk_documents
from src.retrieval.bm25 import BM25Index
from src.retrieval.dense import DenseIndex


class TinyEncoder:
    identity = {"name": "synthetic-unit-test", "revision": "1", "normalize": True}

    def encode(self, texts, *, query=False):
        return np.array(
            [
                [1, 0]
                if any(t in text.lower() for t in ("car", "vehicle", "automobile"))
                else [0, 1]
                for text in texts
            ],
            dtype="float32",
        )


@pytest.fixture
def chunks():
    return chunk_documents(
        [
            Document(
                doc_id="public",
                source_type="docs",
                title="Transport",
                content="A car is a vehicle.",
            ),
            Document(
                doc_id="secret",
                source_type="email",
                title="Restricted",
                content="A car automobile vehicle secret.",
            ),
            Document(
                doc_id="fruit",
                source_type="docs",
                title="Food",
                content="Apples and bananas are fruit.",
            ),
        ]
    )


def request(ids=("public", "fruit"), query="car", **kwargs):
    return SearchRequest(
        query=query,
        user=UserContext(user_id="test", allowed_doc_ids=set(ids)),
        **kwargs,
    )


def test_chunk_coverage_offsets_metadata_and_determinism():
    doc = Document(
        doc_id="d", source_type="docs", title="Title", content="abcdefghijklmnop"
    )
    config = ChunkingConfig(size=6, overlap=2)
    rows = chunk_document(doc, config)
    assert [(r.start_char, r.end_char) for r in rows] == [
        (0, 6),
        (4, 10),
        (8, 14),
        (12, 16),
    ]
    assert all(
        r.text == doc.content[r.start_char : r.end_char] and r.title == "Title"
        for r in rows
    )
    assert rows == chunk_document(doc, config)
    assert (
        rows[0].chunk_id
        != chunk_document(
            doc.model_copy(update={"content": "Abcdefghijklmnop"}), config
        )[0].chunk_id
    )
    assert not chunk_document(doc.model_copy(update={"content": "  "}))
    with pytest.raises(ValueError):
        ChunkingConfig(size=5, overlap=5)
    with pytest.raises(ValueError):
        chunk_documents([doc, doc])


@pytest.mark.parametrize("kind", ["bm25", "dense"])
def test_scope_filters_and_roundtrip(kind, chunks, tmp_path):
    index = BM25Index(chunks) if kind == "bm25" else DenseIndex(chunks, TinyEncoder())
    hits = index.search(request())
    assert hits[0].chunk.doc_id == "public"
    assert all(h.chunk.doc_id != "secret" for h in hits)
    assert not index.search(SearchRequest(query="car", user=UserContext()))
    assert not index.search(request(source_types=("email",)))
    assert not index.search(
        request(after="2026-01-01T00:00:00Z")
    )  # Unknown freshness excluded.
    index.save(tmp_path)
    loaded = (
        BM25Index.load(tmp_path)
        if kind == "bm25"
        else DenseIndex.load(tmp_path, TinyEncoder())
    )
    assert loaded.search(request()) == hits
    # A larger, denied corpus does not change authorized scores or rank.
    small = (
        BM25Index([c for c in chunks if c.doc_id != "secret"])
        if kind == "bm25"
        else DenseIndex([c for c in chunks if c.doc_id != "secret"], TinyEncoder())
    )
    assert small.search(request()) == hits


def test_bm25_no_match_or_empty_corpus(chunks):
    assert not BM25Index(chunks).search(request(query="unfindable"))
    assert not BM25Index([]).search(request())
    assert not BM25Index(chunks).search(request(query="?!"))


def test_dense_rejects_wrong_revision_and_zero_vectors(chunks, tmp_path):
    index = DenseIndex(chunks, TinyEncoder())
    index.save(tmp_path)
    encoder = TinyEncoder()
    encoder.identity = {**encoder.identity, "revision": "different"}
    with pytest.raises(ValueError, match="revision"):
        DenseIndex.load(tmp_path, encoder)
    with pytest.raises(ValueError, match="Zero"):
        DenseIndex(chunks, TinyEncoder(), np.zeros((len(chunks), 2)))


def test_corrupted_metadata_rejected(chunks, tmp_path):
    BM25Index(chunks).save(tmp_path)
    with (tmp_path / "chunks.jsonl").open("a") as f:
        f.write("corruption")
    with pytest.raises(ValueError, match="checksum"):
        BM25Index.load(tmp_path)
