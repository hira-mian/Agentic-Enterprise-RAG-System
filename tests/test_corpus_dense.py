import json
import re
from hashlib import md5

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.retrieval.corpus_dense import CorpusDense, embed_corpus, windows


class FakeModel:
    """Whitespace tokens and hashed bag-of-words vectors; no model download."""

    identity = {"name": "fake", "fp16": False}

    def __init__(self):
        self.encoded = 0

    def offsets(self, texts):
        return [[m.span() for m in re.finditer(r"\S+", t)] for t in texts]

    def encode(self, texts, query=False):
        self.encoded += len(texts)
        out = np.zeros((len(texts), 32), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in text.lower().split():
                out[row, int(md5(word.encode()).hexdigest(), 16) % 32] += 1
        out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
        return out


def test_windows_cover_document_with_overlap():
    offsets = [(i * 2, i * 2 + 1) for i in range(7)]  # Seven one-character tokens.
    assert windows(offsets, window=3, overlap=1) == [(0, 5), (4, 9), (8, 13)]
    assert windows(offsets[:2], window=3, overlap=1) == [(0, 3)]
    assert windows([], window=3, overlap=1) == []
    with pytest.raises(ValueError):
        windows(offsets, window=3, overlap=3)


@pytest.fixture
def corpus(tmp_path):
    table = {
        "doc_id": ["d0", "d1", "d2", "d3", "d4"],
        "source_type": ["slack"] * 5,
        "title": ["Deploy", "", "Sales", "Grid", ""],
        "content": [
            "canary rollback steps for the deployment",
            "lunch menu " * 6 + "canary rollback",
            "quarterly forecast review",
            "elevation tokens for grid density",
            "",
        ],
        "path": ["p"] * 5,
    }
    path = tmp_path / "docs.parquet"
    pq.write_table(pa.table(table), path)
    return path


def test_embed_is_resumable_and_rank_uses_best_passage(corpus, tmp_path):
    out = tmp_path / "dense"
    model = FakeModel()
    manifest = embed_corpus(corpus, out, model, shard_docs=2, window=4, overlap=1)
    assert manifest["complete"] and manifest["shards"] == 3
    passages = pq.read_table(out / "shard_00000.parquet").to_pydict()
    # d1 is long enough for several windows; every window maps back to d1.
    assert passages["doc_index"].count(1) > 1
    first = model.encoded
    embed_corpus(corpus, out, model, shard_docs=2, window=4, overlap=1)
    assert model.encoded == first  # Completed shards are not recomputed.
    with pytest.raises(ValueError, match="different settings"):
        embed_corpus(corpus, out, model, shard_docs=2, window=8, overlap=1)

    index = CorpusDense.load(out, model, corpus_path=corpus)
    hits = index.rank("canary rollback", top_k=2)
    assert {d for d, _ in hits} == {"d0", "d1"}
    # d4 has no text, so it has no passages and can never be returned.
    assert {d for d, _ in index.rank("canary rollback", top_k=10)} == {
        "d0",
        "d1",
        "d2",
        "d3",
    }
    mask = np.array([False, True, True, True, True])
    assert "d0" not in [d for d, _ in index.rank("canary rollback", 2, mask)]
    scores = [s for _, s in index.rank("grid elevation", top_k=4)]
    assert scores == sorted(scores, reverse=True)


def test_load_rejects_incomplete_or_mismatched_embeddings(corpus, tmp_path):
    out = tmp_path / "dense"
    embed_corpus(corpus, out, FakeModel(), shard_docs=5, window=4, overlap=1)
    other = FakeModel()
    other.identity = {"name": "other", "fp16": False}
    with pytest.raises(ValueError, match="Query model"):
        CorpusDense.load(out, other)
    manifest = json.loads((out / "manifest.json").read_text())
    manifest["complete"] = False
    (out / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="incomplete"):
        CorpusDense.load(out, FakeModel())
