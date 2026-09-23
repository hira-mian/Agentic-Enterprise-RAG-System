# Retrieval and generation

## Indexes

Activate `.venv`, install `requirements.txt`, and run `python -m src.data.audit`
if the sample corpus is missing.

```sh
python -m src.retrieval.index_cli build --input .cache/audit/documents.jsonl --output .cache/indexes/bm25 --method bm25
python -m src.retrieval.index_cli search --index .cache/indexes/bm25 --query 'perf-canary rollback' --benchmark-all-docs --top-k 5
```

Defaults: 256 documents, 1,000-character chunks, 100-character overlap. Chunks
preserve text, offsets, title, source, timestamp, and provenance. IDs include
content and chunking settings. Blank chunks are skipped; duplicate IDs fail.
BM25+ uses lowercase word tokens, k1=1.5, b=.75, delta=1; nonmatches are omitted.

Dense retrieval downloads pinned BGE-small weights and runs locally on CPU:

```sh
python -m src.retrieval.index_cli build --input .cache/audit/documents.jsonl --output .cache/indexes/dense --method dense --limit 8 --chunk-size 600 --overlap 60
python -m src.retrieval.index_cli search --index .cache/indexes/dense --query 'How can I undo a deployment?' --benchmark-all-docs
```

BGE uses normalized embeddings and its [query prefix](https://huggingface.co/BAAI/bge-small-en-v1.5);
FAISS inner product gives cosine similarity. Oversize inputs fail; reduce chunk
size rather than silently truncating. On macOS, `OMP_NUM_THREADS` defaults to 1
to avoid the native-library crash found during testing. Set it before starting
Python if you import PyTorch/FAISS before this module.

Both retrievers filter permissions and sources before ranking. Date filters
exclude unknown timestamps. Use `--allow-doc ID` for restricted scope;
`--benchmark-all-docs` grants access to the whole local index for benchmark runs.
Without either flag, queries return no hits.

Indexes and manifests stay in `.cache/`. Manifests record settings and checksums;
load only trusted artifacts. Each query rebuilds statistics or an index over its
authorized subset. This works for small corpora but needs optimization for scale.

## Generation

`GroundedGenerator.generate(request)` returns `(Answer, Usage)`. It uses authorized
chunks that fit the context budget. Empty context abstains without a provider call.
Malformed/truncated output and unknown citations produce errors. Only transient
provider failures are retried. Valid citations do not guarantee a correct answer.

For Anthropic, set `ANTHROPIC_API_KEY` and supply `AnthropicConfig` with a model,
`allow_paid=True`, budget, input/output token prices, timeout, and output-token cap.
Paid calls are disabled by default. The provider reserves worst-case cost before
each attempt and retains reservations after failures. Prices are caller-supplied;
this is a conservative estimate, not a billing guarantee. Use one provider per
sequential job. API call counts include token counting; lost response usage is
unknown. SDK retries are disabled so retries are counted in one place.

## Metrics and tests

```sh
python -m src.evaluation.retrieval_metrics --input tests/fixtures/retrieval_predictions.jsonl
python -m pytest -q
```

Retriever `top_k` counts chunks; metric k counts unique documents. Pass parent
document IDs in ranked order, with enough hits for the cutoff. Metrics report
Recall@k, binary nDCG@k, category averages, errors, and exclusions. Unlabeled
questions are N/A; eligible error rows score zero. Fixtures are invented examples.

After downloading BGE, run the real-model test offline:

```sh
HF_HUB_OFFLINE=1 RUN_MODEL_TESTS=1 python -m pytest tests/test_dense_model.py -q
```

Local checks passed for 2,736 BM25 chunks, 169 BGE chunks, persistence, semantic
retrieval, and permissions. Anthropic is mock-tested; no paid calls were made.
The sample lacks most reference evidence, so these checks are not baseline results.
