"""Deterministic character chunks that preserve original evidence and offsets."""

from hashlib import sha256
from typing import Iterable

from pydantic import Field, model_validator

from src.contracts import Chunk, Contract, Document


class ChunkingConfig(Contract):
    size: int = Field(default=1000, ge=1)
    overlap: int = Field(default=100, ge=0)
    version: str = "characters-v1"

    @model_validator(mode="after")
    def valid_overlap(self):
        if self.overlap >= self.size:
            raise ValueError("overlap must be smaller than size")
        return self


def chunk_document(
    document: Document, config: ChunkingConfig | None = None
) -> list[Chunk]:
    config = config or ChunkingConfig()
    output = []
    start = 0
    while start < len(document.content):
        end = min(start + config.size, len(document.content))
        text = document.content[start:end]
        if text.strip():
            signature = f"{config.version}:{config.size}:{config.overlap}:{text}"
            digest = sha256(signature.encode()).hexdigest()[:16]
            output.append(
                Chunk(
                    chunk_id=f"{document.doc_id}:{start}:{end}:{digest}",
                    doc_id=document.doc_id,
                    source_type=document.source_type,
                    title=document.title,
                    metadata_origin=document.metadata_origin,
                    text=text,
                    start_char=start,
                    end_char=end,
                    timestamp=document.timestamp,
                )
            )
        if end == len(document.content):
            break
        start = end - config.overlap
    return output


def chunk_documents(
    documents: Iterable[Document], config: ChunkingConfig | None = None
) -> list[Chunk]:
    seen = set()
    result = []
    for document in documents:
        if document.doc_id in seen:
            raise ValueError(f"Duplicate document ID: {document.doc_id}")
        seen.add(document.doc_id)
        result.extend(chunk_document(document, config))
    return result
