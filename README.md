# Agentic-Enterprise-RAG-System

CIS 5980 AI Capstone, **AI Engineering track**. The goal is an agentic retrieval-augmented generation (RAG) assistant that answers questions needing evidence from several documents in an enterprise-style knowledge base, and declines when the answer isn't there.

This repository has the Milestone 2 deliverables:

| Deliverable | Where |
|---|---|
| Dataset + Data Card (provenance, licensing, splits, limitations) | [`docs/DATA_CARD.md`](docs/DATA_CARD.md), `scripts/download_data.py`, `data/splits/` |
| Runnable evaluation harness (metrics, test design) | `enterprise_rag/evaluate.py`, [`docs/EVALUATION.md`](docs/EVALUATION.md) |
| Qualitative evaluation rubric | [`docs/QUALITATIVE_RUBRIC.md`](docs/QUALITATIVE_RUBRIC.md) |
| Simple baselines (majority answer, BM25 retrieval) | `enterprise_rag/baselines/majority.py`, `enterprise_rag/baselines/rag.py` |
| Open-source reference baseline (**LlamaIndex default RAG**) | `enterprise_rag/baselines/rag.py` |
| Results table + initial error analysis | [`docs/BASELINE_RESULTS.md`](docs/BASELINE_RESULTS.md), `results/` |

## The LlamaIndex default RAG baseline

This is the standard LlamaIndex "5-line" pipeline with **no tuning**:

```python
index = VectorStoreIndex.from_documents(documents)   # SentenceSplitter(chunk_size=1024, chunk_overlap=200)
answer = index.as_query_engine().query(question)     # similarity_top_k=2, "compact" synthesizer, default QA prompt
```

The only change is that the OpenAI defaults are replaced with open-source models, so the whole baseline is open source and runs locally:

* embeddings: `BAAI/bge-base-en-v1.5` through FastEmbed (ONNX, CPU)
* LLM: any Ollama model (default `llama3.1:8b`) at temperature 0

Both can be switched back to OpenAI (`--embed openai:text-embedding-3-small --llm openai:gpt-4o-mini`) to get the fully default LlamaIndex stack.

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/download_data.py      # MultiHop-RAG corpus + queries -> data/raw/ (SHA-256 verified)
python scripts/make_splits.py        # optional: recreates data/splits/{dev,test}.json (already committed)
```

The embedding model downloads from the Hugging Face Hub the first time it is used. If the Hub is blocked on your network, get it from FastEmbed's GCS mirror instead:

```bash
python scripts/download_embed_model.py   # -> models/fast-bge-base-en-v1.5
```

For generation, use one of these:

* **Open source (local):** install [Ollama](https://ollama.com), then run `ollama pull llama3.1:8b`. If the server is not on `localhost:11434`, set `OLLAMA_BASE_URL`.
* **OpenAI:** `export OPENAI_API_KEY=...`

## Running the evaluation harness

```bash
# 1. Simple baselines (no model needed, run in seconds)
python -m enterprise_rag.evaluate --system majority_global
python -m enterprise_rag.evaluate --system majority_per_type

# 2. Retrieval-only evaluation (no LLM needed; embeds the corpus once, about 10 minutes on 4 CPU cores,
#    then the index is reused from storage/)
python -m enterprise_rag.evaluate --system bm25               --llm none
python -m enterprise_rag.evaluate --system llamaindex_default --llm none

# 3. End-to-end RAG (retrieval + generation) with an open-source LLM
python -m enterprise_rag.evaluate --system llamaindex_default --llm ollama:llama3.1:8b
python -m enterprise_rag.evaluate --system bm25               --llm ollama:llama3.1:8b
#    ...or with OpenAI
python -m enterprise_rag.evaluate --system llamaindex_default --llm openai:gpt-4o-mini

# Quick checks: --limit N evaluates the first N queries; --split dev uses the dev split;
# --resume continues an interrupted run.
python -m enterprise_rag.evaluate --system llamaindex_default --llm ollama:llama3.1:8b --limit 50

# 4. Build the results table from every run in results/
python scripts/make_results_table.py        # -> results/RESULTS.md

# 5. Export a sample for manual scoring with the qualitative rubric
python scripts/export_review_sample.py results/<run>/predictions.jsonl --per-type 10

# Unit tests for the scorer
pytest tests/
```

Useful flags: `--top-k` (chunks passed to the LLM, default 2 = LlamaIndex default), `--eval-ks` (retrieval diagnostic depths, default `2 5 10`), `--embed`, `--rebuild-index`, `--name`.

### Outputs

Each run writes `results/<run-name>/`:

* `predictions.jsonl`: one row per query, with gold answer and gold URLs, retrieved article URLs (ranked), the top-k chunk ids and scores, the prediction, per-query metrics, and latencies
* `summary.json`: run configuration, overall metrics, and metrics per question type

Example: `python -m enterprise_rag.evaluate --system llamaindex_default --llm none` prints

```
Loading persisted index from storage/vector-fastembed-BAAI-bge-base-en-v1.5-c1024-o200
Evaluating llamaindex_default on 2045 test queries -> results/llamaindex_default__test__fastembed-BAAI-bge-base-en-v1.5__retrieval-only
  100/2045
  ...
{
  "n": 2045,
  "all@2": ...,
  "hit@2": ...,
  "recall@2": ...,
  ...
}
```

The full numbers are in [`docs/BASELINE_RESULTS.md`](docs/BASELINE_RESULTS.md).

## Repository layout

```
enterprise_rag/
  data.py            corpus/query loading, splits, conversion to LlamaIndex Documents
  metrics.py         retrieval (hit/recall/all-evidence/MRR@k) and answer (accuracy/EM/F1) metrics
  models.py          embedding / LLM factories (fastembed, openai, ollama, mock)
  evaluate.py        evaluation harness CLI
  baselines/
    majority.py      answer-prior baselines
    rag.py           LlamaIndex default RAG + BM25 RAG
scripts/             data download, split creation, embedding-model download, results table, review sample export
data/splits/         committed dev/test query-id lists
docs/                Data Card, evaluation design, qualitative rubric, baseline results + error analysis
results/             committed run outputs (summary.json + predictions.jsonl)
tests/               scorer unit tests
```

## Citation

Dataset: Yixuan Tang and Yi Yang. *MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop Queries.* COLM 2024. arXiv:2401.15391. Licensed ODC-BY 1.0.
