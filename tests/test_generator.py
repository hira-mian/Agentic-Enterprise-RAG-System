import json
from types import SimpleNamespace

import pytest

from src.contracts import Document, GenerationRequest, SearchRequest, Usage, UserContext
from src.data.preprocessing import chunk_documents
from src.generation.generator import (
    AnthropicConfig,
    AnthropicProvider,
    Completion,
    GenerationConfig,
    GroundedGenerator,
    ProviderError,
)
from src.retrieval.bm25 import BM25Index
from tests.fakes import FakeProvider


def request():
    user = UserContext(user_id="demo", allowed_doc_ids={"d"})
    chunks = chunk_documents(
        [
            Document(
                doc_id="d",
                source_type="docs",
                title="Limit",
                content="Upload limit is 10 MB.",
            )
        ]
    )
    evidence = BM25Index(chunks).search(SearchRequest(query="upload", user=user))
    return GenerationRequest(
        question="What is the upload limit?", user=user, evidence=evidence
    )


def completion(req, **changes):
    output = {
        "status": "answered",
        "text": "The upload limit is 10 MB.",
        "citation_ids": [req.evidence[0].citation_id],
    }
    output.update(changes)
    return Completion(
        json.dumps(output),
        Usage(input_tokens=20, output_tokens=10, api_calls=1, estimated_cost_usd=0.001),
    )


def test_retrieval_to_generation_uses_only_supplied_evidence():
    req = request()
    provider = FakeProvider([completion(req)])
    answer, usage = GroundedGenerator(provider).generate(req)
    assert answer.status == "answered" and usage.api_calls == 1
    payload = json.loads(provider.calls[0][1])
    assert payload["evidence"][0]["text"] == "Upload limit is 10 MB."
    assert set(payload) == {"question", "evidence"}


def test_empty_or_over_budget_context_never_calls_provider():
    req = request()
    provider = FakeProvider([])
    answer, usage = GroundedGenerator(
        provider, GenerationConfig(max_context_chars=1)
    ).generate(req)
    assert answer.status == "insufficient_evidence" and usage.api_calls == 0
    empty = GenerationRequest(question="Unknown?", user=UserContext())
    assert (
        GroundedGenerator(provider).generate(empty)[0].status == "insufficient_evidence"
    )
    assert not provider.calls


@pytest.mark.parametrize("kind", ["invented", "malformed", "truncated"])
def test_invalid_responses_are_not_answers(kind):
    req = request()
    value = {
        "invented": completion(req, citation_ids=["made-up"]),
        "malformed": Completion("not JSON", Usage(api_calls=1)),
        "truncated": Completion(
            completion(req).text, Usage(api_calls=1), truncated=True
        ),
    }[kind]
    answer, usage = GroundedGenerator(FakeProvider([value])).generate(req)
    assert answer.status == "error" and usage.api_calls == 1


def test_retries_bounded_and_usage_retained():
    req = request()
    provider = FakeProvider([ProviderError(retryable=True), completion(req)])
    answer, usage = GroundedGenerator(
        provider, GenerationConfig(retry_delay_seconds=0)
    ).generate(req)
    assert answer.status == "answered" and usage.api_calls == 2
    assert usage.estimated_cost_usd is None  # Failed call's billing is unknown.
    provider = FakeProvider(
        [ProviderError(retryable=True), ProviderError(retryable=True)]
    )
    answer, usage = GroundedGenerator(
        provider, GenerationConfig(retry_delay_seconds=0)
    ).generate(req)
    assert answer.status == "error" and usage.api_calls == 2


def test_paid_calls_disabled_by_default():
    provider = AnthropicProvider(
        AnthropicConfig(model="explicit-model"), client=object()
    )
    with pytest.raises(ProviderError) as result:
        provider.complete("system", "question")
    assert result.value.usage.api_calls == 0


class Messages:
    def __init__(self):
        self.created = 0

    def count_tokens(self, **kwargs):
        return SimpleNamespace(input_tokens=100)

    def create(self, **kwargs):
        self.created += 1
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="{}")],
            usage=SimpleNamespace(input_tokens=100, output_tokens=10),
            stop_reason="end_turn",
        )


def test_anthropic_budget_reservation_and_sdk_mapping():
    messages = Messages()
    config = AnthropicConfig(
        model="explicit-model",
        allow_paid=True,
        budget_usd=0.0003,
        input_usd_per_million=1,
        output_usd_per_million=2,
        max_output_tokens=100,
    )
    provider = AnthropicProvider(config, client=SimpleNamespace(messages=messages))
    result = provider.complete("system", "prompt")
    assert (
        result.usage.api_calls == 2
        and result.usage.estimated_cost_usd == pytest.approx(0.00012)
    )
    with pytest.raises(ProviderError):
        provider.complete("system", "prompt")
    assert messages.created == 1


def test_citation_to_evidence_excluded_by_context_budget_is_rejected():
    req = request()
    other = req.evidence[0].model_copy(
        update={
            "citation_id": "excluded",
            "chunk": req.evidence[0].chunk.model_copy(update={"chunk_id": "other"}),
        }
    )
    request_with_two = GenerationRequest(
        question=req.question, user=req.user, evidence=(*req.evidence, other)
    )
    provider = FakeProvider([completion(req, citation_ids=["excluded"])])
    answer, _ = GroundedGenerator(
        provider, GenerationConfig(max_context_chars=len(req.evidence[0].chunk.text))
    ).generate(request_with_two)
    assert answer.status == "error"
    assert len(json.loads(provider.calls[0][1])["evidence"]) == 1


def test_authentication_error_is_not_retried():
    provider = FakeProvider([ProviderError(retryable=False)])
    answer, usage = GroundedGenerator(provider).generate(request())
    assert answer.status == "error" and usage.api_calls == 1
    assert len(provider.calls) == 1
