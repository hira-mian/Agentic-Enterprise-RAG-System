"""Shared defaults. Model and budget choices must be recorded with each run."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "onyx-dot-app/EnterpriseRAG-Bench"
DATASET_REVISION = "69916e31c68aa5963c00248fd7f0bc12d04fd235"


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    dataset_revision: str = DATASET_REVISION
    seed: int = 5980
    top_k: int = Field(default=10, ge=1, le=100)
    max_search_rounds: int = Field(default=3, ge=1, le=10)
    max_context_chars: int = Field(default=24000, ge=1)
    timeout_seconds: float = Field(default=60, gt=0)
    max_tokens: int = Field(default=8000, ge=1)
    max_cost_usd: float = Field(default=0, ge=0)  # Paid calls disabled by default.


EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
