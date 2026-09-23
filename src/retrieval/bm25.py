"""BM25+ baseline over authorized chunks; scoped corpus statistics avoid leaks."""

import re
from pathlib import Path

from pydantic import Field
from rank_bm25 import BM25Plus

from src.contracts import Contract, SearchRequest
from src.retrieval.common import (
    eligible,
    evidence_for,
    load_metadata,
    save_metadata,
    validate_chunks,
)


class BM25Config(Contract):
    k1: float = Field(default=1.5, gt=0)
    b: float = Field(default=0.75, ge=0, le=1)
    delta: float = Field(default=1, ge=0)
    tokenizer: str = "unicode-words-lower-v1"


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class BM25Index:
    def __init__(
        self, chunks, config: BM25Config | None = None, provenance: dict | None = None
    ):
        self.chunks = tuple(chunks)
        validate_chunks(self.chunks)
        self.config = config or BM25Config()
        if self.config.tokenizer != "unicode-words-lower-v1":
            raise ValueError("Unsupported tokenizer")
        self.provenance = provenance or {}
        self.tokens = [tokenize(c.text) for c in self.chunks]

    def search(self, request: SearchRequest):
        query = tokenize(request.query)
        rows = [
            (c, t)
            for c, t in zip(self.chunks, self.tokens)
            if t and eligible(c, request)
        ]
        if not query or not rows:
            return ()
        chunks, tokens = zip(*rows)
        model = BM25Plus(
            tokens, k1=self.config.k1, b=self.config.b, delta=self.config.delta
        )
        scores = model.get_scores(query)
        # BM25+ gives a floor score to nonmatches; don't return those as evidence.
        matches = [
            (c, score)
            for c, t, score in zip(chunks, tokens, scores)
            if set(query).intersection(t)
        ]
        if not matches:
            return ()
        matched_chunks, matched_scores = zip(*matches)
        return evidence_for(matched_chunks, matched_scores, request, "bm25")

    def save(self, directory):
        save_metadata(
            Path(directory),
            self.chunks,
            "bm25",
            self.config.model_dump(),
            self.provenance,
        )

    @classmethod
    def load(cls, directory):
        chunks, manifest = load_metadata(Path(directory), "bm25")
        return cls(
            chunks,
            BM25Config.model_validate(manifest["config"]),
            manifest["provenance"],
        )
