import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config import Settings
from src.contracts import (Answer, Chunk, CriticDecision, Document, GenerationRequest,
                           QueryRequest, QueryResponse, UserContext)

FIXTURE = json.loads((Path(__file__).parent / "fixtures/contracts.json").read_text())


def test_serialized_boundaries_round_trip():
    for cls, key in [(Document, "document"), (QueryRequest, "request"),
                     (UserContext, "user"), (QueryResponse, "response")]:
        obj = cls.model_validate(FIXTURE[key])
        assert cls.model_validate_json(obj.model_dump_json()) == obj


def test_default_and_unknown_identity_deny_access():
    assert not UserContext().permits("demo-1")
    assert not UserContext(allowed_doc_ids={"demo-1"}).permits("demo-1")
    assert not UserContext(user_id="known", role="admin").permits("demo-1")


def test_client_cannot_supply_authorization():
    with pytest.raises(ValidationError):
        QueryRequest(question="hello", allowed_doc_ids=["demo-1"])


def test_generation_rejects_denied_evidence():
    response = QueryResponse.model_validate(FIXTURE["response"])
    with pytest.raises(ValidationError):
        GenerationRequest(question="limit?", user=UserContext(), evidence=response.sources)
    request = GenerationRequest(question="limit?", user=UserContext.model_validate(FIXTURE["user"]),
                                evidence=response.sources)
    assert request.evidence == response.sources


def test_unknown_citation_rejected():
    data = json.loads(json.dumps(FIXTURE["response"]))
    data["answer"]["citation_ids"] = ["invented"]
    with pytest.raises(ValidationError):
        QueryResponse.model_validate(data)


def test_invalid_offsets_and_confidence():
    with pytest.raises(ValidationError):
        Chunk(chunk_id="x", doc_id="d", source_type="slack", text="abc", start_char=5, end_char=7)
    with pytest.raises(ValidationError):
        CriticDecision(sufficient=True, confidence=1.1, reason="enough")
    with pytest.raises(ValidationError):
        Settings(max_cost_usd=float("nan"))


def test_abstention_needs_no_citation_but_answer_does():
    Answer(status="insufficient_evidence", text="No supporting evidence found.")
    with pytest.raises(ValidationError):
        Answer(status="answered", text="I guessed.")


def test_timestamps_require_timezone():
    with pytest.raises(ValidationError):
        Document(doc_id='d', source_type='docs', title='t', content='text', timestamp='2026-01-01T00:00:00')
