import json

import bm25s
import numpy as np
import pyarrow.parquet as pq
import pytest

from src.contracts import SearchRequest, UserContext
from src.data.corpus import build_corpus, extract_document, question_coverage
from src.evaluation.run_retrieval import load_questions, run
from src.retrieval.corpus_bm25 import BM25Config, CorpusBM25, build_bm25, tokenize

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
