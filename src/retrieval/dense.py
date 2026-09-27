"""Normalized BGE embeddings and scoped exact FAISS cosine search."""

import json
import os
import sys
from hashlib import sha256
from pathlib import Path
from typing import Protocol

# macOS PyTorch/FAISS OpenMP runtimes can crash with parallel CPU workers.
# Set the default before either native runtime loads; users may override it.
if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import faiss
import numpy as np

from src.config import ROOT
from src.contracts import SearchRequest
from src.retrieval.common import (
    eligible,
    evidence_for,
    load_metadata,
    save_metadata,
    validate_chunks,
)


class Encoder(Protocol):
    @property
    def identity(self) -> dict: ...
    def encode(self, texts: list[str], *, query: bool = False) -> np.ndarray: ...


class BGEEncoder:
    def __init__(
        self, revision: str, model_name="BAAI/bge-small-en-v1.5", batch_size=32
    ):
        from sentence_transformers import SentenceTransformer

        if not revision or revision == "main":
            raise ValueError("Use a pinned embedding model revision")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size
        self.model_name, self.revision = model_name, revision
        self.model = SentenceTransformer(
            model_name,
            revision=revision,
            device="cpu",
            cache_folder=str(ROOT / ".cache" / "models"),
            trust_remote_code=False,
        )
        self.prefix = "Represent this sentence for searching relevant passages: "

    @property
    def identity(self):
        return {
            "name": self.model_name,
            "revision": self.revision,
            "query_prefix": self.prefix,
            "normalize": True,
            "max_seq_length": self.model.max_seq_length,
        }

    def encode(self, texts, *, query=False):
        inputs = [self.prefix + t if query else t for t in texts]
        lengths = self.model.tokenizer(inputs, truncation=False, return_length=True)[
            "length"
        ]
        if any(length > self.model.max_seq_length for length in lengths):
            raise ValueError(
                "Input exceeds embedding token limit; reduce chunk/query length"
            )
        return self.model.encode(
            inputs,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )


def normalized(vectors):
    array = np.array(vectors, dtype="float32", copy=True, order="C")
    if array.ndim != 2 or array.shape[1] == 0 or not np.isfinite(array).all():
        raise ValueError("Embeddings must be a finite 2D matrix")
    if (np.linalg.norm(array, axis=1) == 0).any():
        raise ValueError("Zero embedding vector")
    faiss.normalize_L2(array)
    return array


class DenseIndex:
    def __init__(self, chunks, encoder: Encoder, vectors=None, provenance=None):
        self.chunks = tuple(chunks)
        validate_chunks(self.chunks)
        if not self.chunks:
            raise ValueError("Dense index requires at least one nonempty chunk")
        self.encoder = encoder
        self.provenance = provenance or {}
        if vectors is None:
            vectors = encoder.encode([chunk.text for chunk in self.chunks])
        self.vectors = normalized(vectors)
        if len(self.vectors) != len(self.chunks):
            raise ValueError("Embedding/chunk count mismatch")
        self.index = faiss.IndexFlatIP(self.vectors.shape[1])
        self.index.add(self.vectors)

    def search(self, request: SearchRequest):
        positions = [i for i, c in enumerate(self.chunks) if eligible(c, request)]
        if not positions:
            return ()
        query = normalized(self.encoder.encode([request.query], query=True))
        if query.shape != (1, self.index.d):
            raise ValueError("Query encoder dimension mismatch")
        scoped = faiss.IndexFlatIP(self.index.d)
        scoped.add(self.vectors[positions])
        scores, indices = scoped.search(query, len(positions))
        chunks = [self.chunks[positions[i]] for i in indices[0]]
        return evidence_for(chunks, scores[0], request, "dense")

    def save(self, directory):
        path = Path(directory)
        manifest = save_metadata(
            path, self.chunks, "dense", self.encoder.identity, self.provenance
        )
        faiss.write_index(self.index, str(path / "index.faiss"))
        manifest["index_sha256"] = sha256(
            (path / "index.faiss").read_bytes()
        ).hexdigest()
        (path / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )

    @classmethod
    def load(cls, directory, encoder):
        path = Path(directory)
        chunks, manifest = load_metadata(path, "dense")
        if manifest["config"] != encoder.identity:
            raise ValueError("Encoder settings/revision differ from saved index")
        if (
            sha256((path / "index.faiss").read_bytes()).hexdigest()
            != manifest["index_sha256"]
        ):
            raise ValueError("Vector artifact checksum mismatch")
        index = faiss.read_index(str(path / "index.faiss"))
        if type(index) is not faiss.IndexFlatIP or index.ntotal != len(chunks):
            raise ValueError("Invalid dense index type/count")
        return cls(
            chunks,
            encoder,
            index.reconstruct_n(0, index.ntotal),
            manifest["provenance"],
        )
