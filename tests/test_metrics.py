import pytest

from enterprise_rag.metrics import (
    answer_correct,
    exact_match,
    is_abstention,
    retrieval_scores,
    token_f1,
)


@pytest.mark.parametrize(
    "pred,gold,qtype,expected",
    [
        ("Sam Bankman-Fried", "Sam Bankman-Fried", "inference_query", True),
        ("The person is Sam Bankman-Fried.", "Sam Bankman-Fried", "inference_query", True),
        ("Sam Altman", "Sam Bankman-Fried", "inference_query", False),
        ("Yes, both articles agree.", "Yes", "comparison_query", True),
        ("No. Although yes ...", "no", "comparison_query", True),
        ("No, they differ.", "Yes", "comparison_query", False),
        ("Nothing definite", "Yes", "comparison_query", False),
        ("The context does not provide this.", "Insufficient information.", "null_query", True),
        ("Insufficient information.", "Insufficient information.", "null_query", True),
        ("It was Google.", "Insufficient information.", "null_query", False),
    ],
)
def test_answer_correct(pred, gold, qtype, expected):
    assert answer_correct(pred, gold, qtype) is expected


def test_contains_is_token_level():
    # "no" must not match inside "Google"/"not"; entity spans must align to tokens.
    assert not answer_correct("It is not stated", "no", "comparison_query")
    assert not answer_correct("Googleplex", "Google", "inference_query")


def test_exact_match_and_f1():
    assert exact_match("The Google.", "google")
    assert token_f1("Sam Bankman-Fried", "Sam Bankman-Fried") == 1.0
    assert token_f1("", "x") == 0.0
    assert 0 < token_f1("Sam Altman", "Sam Bankman-Fried") < 1


def test_abstention():
    assert is_abstention("I don't know.")
    assert not is_abstention("Yes.")


def test_retrieval_scores_dedupes_chunks_by_document():
    retrieved = ["a", "a", "x", "b"]
    s = retrieval_scores(retrieved, ["a", "b"], k=2)
    assert s == {"hit@2": 1.0, "recall@2": 0.5, "all@2": 0.0, "mrr@2": 1.0}
    s = retrieval_scores(retrieved, ["a", "b"], k=4)
    assert s["all@4"] == 1.0
    s = retrieval_scores(["x", "b"], ["a", "b"], k=2)
    assert s["mrr@2"] == 0.5
