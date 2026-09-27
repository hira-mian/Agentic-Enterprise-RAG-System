"""Opt-in real-model integration; default CI stays offline."""

import os
from pathlib import Path

import pytest

from src.config import EMBEDDING_REVISION
from src.contracts import Document, SearchRequest, UserContext
from src.data.preprocessing import chunk_documents


@pytest.mark.skipif(
    os.environ.get("RUN_MODEL_TESTS") != "1",
    reason="Set RUN_MODEL_TESTS=1 after downloading BGE",
)
def test_real_bge_semantic_search_reload_and_scope(tmp_path):
    from src.retrieval.dense import BGEEncoder, DenseIndex

    fixture = Path(__file__).parent / "fixtures/retrieval_documents.jsonl"
    docs = [
        Document.model_validate_json(line) for line in fixture.read_text().splitlines()
    ]
    encoder = BGEEncoder(EMBEDDING_REVISION)
    index = DenseIndex(chunk_documents(docs), encoder)
    user = UserContext(user_id="test", allowed_doc_ids={"demo-transport", "demo-food"})
    request = SearchRequest(query="Can I use an automobile for a work trip?", user=user)
    result = index.search(request)
    assert result[0].chunk.doc_id == "demo-transport"
    assert all(hit.chunk.doc_id != "demo-secret" for hit in result)
    index.save(tmp_path)
    loaded = DenseIndex.load(tmp_path, encoder)
    restored = loaded.search(request)
    assert [hit.citation_id for hit in restored] == [hit.citation_id for hit in result]
    assert [hit.score for hit in restored] == pytest.approx(
        [hit.score for hit in result]
    )
