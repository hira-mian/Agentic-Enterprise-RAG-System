from math import log2

import pytest

from src.evaluation.retrieval_metrics import evaluate_retrieval, ndcg_at_k, recall_at_k


def test_hand_calculated_ranking_and_deduplication():
    # Unique ranking is a, irrelevant, b. Both a and b are relevant.
    hits = ["a", "a", "x", "b"]
    assert recall_at_k(hits, ["a", "b"], 2) == 0.5
    assert recall_at_k(hits, ["a", "b"], 3) == 1
    assert ndcg_at_k(hits, ["a", "b"], 3) == pytest.approx(1.5 / (1 + 1 / log2(3)))
    assert ndcg_at_k(["b", "a"], ["a", "b"], 2) == 1
    assert recall_at_k(["a"], ["a", "a"], 1) == 1


def test_empty_predictions_and_unlabeled():
    assert recall_at_k([], ["a"], 10) == 0
    assert ndcg_at_k([], ["a"], 10) == 0
    assert recall_at_k(["a"], [], 10) is None
    assert ndcg_at_k([], [], 10) is None


@pytest.mark.parametrize("k", [0, -1, 1.5, True])
def test_invalid_k(k):
    with pytest.raises(ValueError):
        recall_at_k([], [], k)


def test_aggregation_includes_errors_and_reports_exclusions():
    rows = [
        {
            "question_id": "1",
            "question_type": "basic",
            "retrieved_doc_ids": ["a"],
            "relevant_doc_ids": ["a"],
        },
        {
            "question_id": "2",
            "question_type": "basic",
            "retrieved_doc_ids": ["a"],
            "relevant_doc_ids": ["a"],
            "error": "timeout",
        },
        {
            "question_id": "3",
            "question_type": "unknown",
            "retrieved_doc_ids": [],
            "relevant_doc_ids": [],
        },
    ]
    report = evaluate_retrieval(rows, ks=(10,))
    assert report["overall"]["recall@10"] == {"mean": 0.5, "scored": 2, "excluded": 1}
    assert report["overall"]["errors"] == 1
    assert report["slices"]["unknown"]["recall@10"]["mean"] is None
    with pytest.raises(ValueError):
        evaluate_retrieval(rows + rows)
