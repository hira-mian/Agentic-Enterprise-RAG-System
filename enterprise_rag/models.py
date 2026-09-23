"""Build LlamaIndex LLM / embedding objects from short spec strings.

Specs are `<provider>:<model>`, e.g.

    --embed fastembed:BAAI/bge-base-en-v1.5   (open-source, runs locally on CPU)
    --embed openai:text-embedding-3-small
    --llm   ollama:llama3.1:8b                 (open-source, local Ollama server)
    --llm   openai:gpt-4o-mini
    --llm   none                               (retrieval-only evaluation)
    --llm   mock / --embed mock                (plumbing smoke tests; meaningless scores)
"""

from __future__ import annotations

import os

from .data import REPO_ROOT

LOCAL_MODELS_DIR = REPO_ROOT / "models"

# Deterministic decoding so repeated runs are comparable. This is the only
# deviation from LlamaIndex defaults in the reference baseline.
TEMPERATURE = 0.0


def _split(spec: str) -> tuple[str, str]:
    provider, _, model = spec.partition(":")
    return provider.lower(), model


def slug(spec: str) -> str:
    return spec.replace("/", "-").replace(":", "-")


def build_embed_model(spec: str):
    provider, model = _split(spec)
    if provider == "fastembed":
        from llama_index.embeddings.fastembed import FastEmbedEmbedding

        kwargs = {}
        # scripts/download_embed_model.py places the ONNX export here for machines
        # that cannot reach the Hugging Face Hub.
        local = LOCAL_MODELS_DIR / f"fast-{model.split('/')[-1]}"
        if local.exists():
            kwargs["specific_model_path"] = str(local)
        return FastEmbedEmbedding(model_name=model, embed_batch_size=64, **kwargs)
    if provider == "openai":
        from llama_index.embeddings.openai import OpenAIEmbedding

        return OpenAIEmbedding(model=model or "text-embedding-3-small")
    if provider == "mock":
        from llama_index.core.embeddings import MockEmbedding

        return MockEmbedding(embed_dim=8)
    raise ValueError(f"Unknown embedding spec: {spec!r}")


def build_llm(spec: str):
    """Return a LlamaIndex LLM, or None for retrieval-only runs."""
    provider, model = _split(spec)
    if provider == "none":
        return None
    if provider == "ollama":
        from llama_index.llms.ollama import Ollama

        return Ollama(
            model=model or "llama3.1:8b",
            temperature=TEMPERATURE,
            request_timeout=300.0,
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
    if provider == "openai":
        from llama_index.llms.openai import OpenAI

        return OpenAI(model=model or "gpt-4o-mini", temperature=TEMPERATURE)
    if provider == "mock":
        from llama_index.core.llms import MockLLM

        return MockLLM(max_tokens=32)
    raise ValueError(f"Unknown LLM spec: {spec!r}")
