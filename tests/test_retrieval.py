import json
import bm25s
import numpy as np
import pyarrow.parquet as pq
import pytest
import re
import pyarrow as pa
from src.contracts import SearchRequest, UserContext
from src.data.corpus import build_corpus, extract_document, question_coverage
from src.evaluation.run_retrieval import load_questions, run
from src.retrieval.corpus_bm25 import BM25Config, CorpusBM25, build_bm25, tokenize
from hashlib import md5
from src.retrieval.corpus_dense import CorpusDense, embed_corpus, windows


# Corpus and BM25

DOCS = [
    "kappa elevation scale for data grids",
    "rollback the canary deployment safely",
    "canary canary rollback checklist for on call",
    "quarterly sales forecast and pipeline review",
    "grid density tokens and elevation review",
]


def write_doc(root, relative, doc_id, title, content, **extra):
    path = root / "generated_data" / "sources" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "dataset_doc_uuid": doc_id,
        "title_field_name": "title",
        "content_field_names": ["body"],
        "title": title,
        "body": content,
        **extra,
    }
    path.write_text(json.dumps(data))


def test_extract_document_matches_upstream_labels():
    single = {"title_field_name": "t", "content_field_names": ["c"], "t": "T", "c": "C"}
    assert extract_document(single) == ("T", "C")
    multi = {
        "title_field_name": "t",
        "content_field_names": ["a", "b"],
        "t": 1,
        "a": ["x", "y"],
        "b": "z",
    }
    assert extract_document(multi) == ("1", "a:\nx\ny\n\nb:\nz")
    with pytest.raises(ValueError):
        extract_document({"title_field_name": "t", "t": "T"})


def test_build_corpus_uses_canonical_file_for_reused_ids(tmp_path):
    repo = tmp_path / "repo"
    write_doc(repo, "slack/a.json", "d1", "A", "alpha")
    write_doc(repo, "jira/b.json", "d2", "B", "beta")
    write_doc(repo, "confluence/dup.json", "d2", "B2", "beta copy")
    index = {"d1": "slack/a.json", "d2": "jira/b.json"}
    (repo / "generated_data" / "uuid_index.json").write_text(json.dumps(index))
    output = tmp_path / "docs.parquet"
    report = build_corpus(repo, output, batch_size=1)
    rows = pq.read_table(output).to_pylist()
    assert [r["doc_id"] for r in rows] == ["d2", "d1"]  # Sorted source directories.
    assert rows[0]["source_type"] == "jira" and rows[0]["content"] == "beta"
    assert report["source_counts"] == {"jira": 1, "slack": 1}
    assert report["skipped"] == [
        {"path": "confluence/dup.json", "reason": "not canonical for d2"}
    ]
    coverage = question_coverage(
        [{"expected_doc_ids": ["d1", "missing"]}, {"expected_doc_ids": []}],
        report["doc_ids"],
    )
    assert coverage["referenced_documents_missing"] == ["missing"]
    assert coverage["questions_without_references"] == 1


def test_build_corpus_fails_when_indexed_document_is_missing(tmp_path):
    repo = tmp_path / "repo"
    write_doc(repo, "slack/a.json", "d1", "A", "alpha")
    index = {"d1": "slack/a.json", "d9": "slack/gone.json"}
    (repo / "generated_data" / "uuid_index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="not exported"):
        build_corpus(repo, tmp_path / "docs.parquet")


def test_vectorized_index_matches_bm25s():
    rng = np.random.default_rng(0)
    words = [f"w{i}" for i in range(200)]
    docs = [" ".join(rng.choice(words, rng.integers(1, 50))) for _ in range(300)]
    model, stats = build_bm25(docs, BM25Config())
    tokens = [tokenize(d) for d in docs]
    reference = bm25s.BM25(k1=1.5, b=0.75, delta=1, method="bm25+")
    reference.index(tokens, show_progress=False)
    assert stats["documents"] == 300
    for query in (["w1", "w2"], ["w7", "w7", "w99"], ["w150"]):
        ours = model.get_scores_from_ids([model.vocab_dict[t] for t in query])
        expected = reference.get_scores(query)
        # float32 rounding differs slightly; rankings must not.
        np.testing.assert_allclose(ours, expected, rtol=1e-6)
        np.testing.assert_array_equal(
            np.round(ours, 4).argsort(kind="stable"),
            np.round(expected, 4).argsort(kind="stable"),
        )


@pytest.fixture
def corpus_index(tmp_path):
    table = {
        "doc_id": [f"d{i}" for i in range(len(DOCS))],
        "source_type": ["linear", "slack", "confluence", "gmail", "linear"],
        "title": ["Grid", "", "Runbook", "Sales", "Tokens"],
        "content": DOCS,
        "path": [f"p{i}" for i in range(len(DOCS))],
    }
    import pyarrow as pa

    path = tmp_path / "docs.parquet"
    pq.write_table(pa.table(table), path)
    index = CorpusBM25.build(path)
    index.save(tmp_path / "index")
    return CorpusBM25.load(tmp_path / "index", corpus_path=path), path


def test_rank_skips_nonmatches_and_respects_masks(corpus_index):
    index, _ = corpus_index
    hits = index.rank("canary rollback", top_k=10)
    assert [d for d, _ in hits] == ["d2", "d1"]
    assert hits[0][1] > hits[1][1]
    assert index.rank("unknownword", top_k=5) == []
    mask = np.array([True, True, False, True, True])
    assert [d for d, _ in index.rank("canary rollback", 10, mask)] == ["d1"]
    assert len(index.rank("elevation grid review", top_k=1)) == 1


def test_search_enforces_permissions_and_sources(corpus_index):
    index, _ = corpus_index
    user = UserContext(user_id="u", allowed_doc_ids=frozenset({"d1", "d4"}))
    results = index.search(
        SearchRequest(query="canary elevation review", user=user, top_k=5)
    )
    assert {e.chunk.doc_id for e in results} == {"d1", "d4"}
    only = index.search(
        SearchRequest(query="canary elevation", user=user, source_types=("linear",))
    )
    assert [e.chunk.doc_id for e in only] == ["d4"]
    assert only[0].chunk.text == "Tokens\ngrid density tokens and elevation review"
    anonymous = UserContext(allowed_doc_ids=frozenset({"d1"}))
    assert index.search(SearchRequest(query="canary", user=anonymous)) == ()


def test_load_rejects_different_corpus(corpus_index, tmp_path):
    _, path = corpus_index
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="Corpus differs"):
        CorpusBM25.load(tmp_path / "index", corpus_path=path)


def test_runner_records_rankings_and_errors(tmp_path):
    questions = tmp_path / "q.jsonl"
    rows = [
        {
            "question_id": "q1",
            "question_type": "basic",
            "source_types": ["slack"],
            "question": "canary",
            "expected_doc_ids": ["d1"],
        },
        {
            "question_id": "q2",
            "question_type": "basic",
            "source_types": [],
            "question": "boom",
            "expected_doc_ids": ["d2"],
        },
    ]
    questions.write_text("".join(json.dumps(r) + "\n" for r in rows))
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"splits": {"development": ["q1", "q2"]}}))

    class Fake:
        def rank(self, query, top_k):
            if query == "boom":
                raise RuntimeError("failed")
            return [("d1", 2.0), ("d3", 1.0)]

    selected = load_questions(questions, splits, "development")
    output, latencies = run(Fake(), selected, top_k=20)
    assert output[0]["retrieved_doc_ids"] == ["d1", "d3"] and output[0]["error"] is None
    assert output[1]["retrieved_doc_ids"] == [] and "failed" in output[1]["error"]
    assert len(latencies) == 2
    splits.write_text(json.dumps({"splits": {"development": ["q1", "q9"]}}))
    with pytest.raises(ValueError):
        load_questions(questions, splits, "development")


# Dense retrieval

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
