from collections import defaultdict

import pytest

from src.data.audit import audit
from src.evaluation.splits import build_splits


def questions():
    return [{"question_id":f"q{i:03}", "question_type": f"type{i % 3}",
             "question":f"Question {i}", "expected_doc_ids":[f"d{i//2}"]}
            for i in range(60)]


def test_split_reproducible_disjoint_complete_and_grouped():
    rows = questions()
    rows[4]["expected_doc_ids"] = ["d1", "d2"]  # Transitive overlap.
    result = build_splits(rows)
    assert result == build_splits(list(reversed(rows)))
    splits = result["splits"]
    all_ids = [q for ids in splits.values() for q in ids]
    assert len(all_ids) == len(set(all_ids)) == len(rows)
    assignments = {q:name for name,ids in splits.items() for q in ids}
    doc_splits = defaultdict(set)
    for q in rows:
        for doc in q["expected_doc_ids"]:
            doc_splits[doc].add(assignments[q["question_id"]])
    assert all(len(names) == 1 for names in doc_splits.values())
    assert set(result["golden_development_ids"]) <= set(splits["development"])
    assert all(splits.values())


def test_duplicate_text_is_grouped_without_reference_docs():
    rows = questions()
    rows[0].update(question="same question", expected_doc_ids=[])
    rows[20].update(question=" SAME  question ", expected_doc_ids=[])
    result = build_splits(rows)
    assert any({"q000", "q020"} <= set(ids) for ids in result["splits"].values())


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError):
        build_splits([questions()[0], questions()[0]])


def test_audit_reports_missing_evidence_not_false_coverage():
    report, _ = audit([{"doc_id":"d0", "source_type":"demo", "content":"x", "title":"x"}],
                      questions(), "test-revision")
    coverage = report["development_reference_coverage"]
    assert any(row["missing_doc_ids"] for row in coverage)
    assert all(row["expected"] == row["present"] + len(row["missing_doc_ids"]) for row in coverage)


def test_committed_split_manifest_and_examples_are_consistent():
    import json
    from pathlib import Path
    base = Path(__file__).resolve().parents[1] / "evaluation"
    manifest = json.loads((base / "splits.json").read_text())
    examples = json.loads((base / "development_examples.json").read_text())
    parts = manifest["splits"]
    ids = [qid for values in parts.values() for qid in values]
    assert len(ids) == len(set(ids)) == 500
    assert [len(parts[k]) for k in ("development", "calibration", "final")] == [300, 100, 100]
    assert len(manifest["golden_development_ids"]) == 60
    assert set(manifest["golden_development_ids"]) <= set(parts["development"])
    assert len({row["question_type"] for row in examples}) == 10
    assert all(row["question_id"] in parts["development"] for row in examples)
    assert all("gold_answer" not in row for row in examples)
