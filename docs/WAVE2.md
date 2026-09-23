# Wave 2 components

Implemented: deterministic chunking, BM25+ and dense retrieval, a grounded
generation adapter, and document-level retrieval metrics. Hybrid/reranking,
Search Critic, adaptive search, and the full evaluation runner are later work.

## Build and query

Activate `.venv` and install `requirements.txt`. Generate the inspection data with
`python -m src.data.audit` if `.cache/audit/documents.jsonl` is missing.

```sh
python -m src.retrieval.index_cli build --input .cache/audit/documents.jsonl --output .cache/indexes/bm25 --method bm25
python -m src.retrieval.index_cli search --index .cache/indexes/bm25 --query 'perf-canary rollback' --benchmark-all-docs --top-k 5
```

The build defaults to 256 documents, 1,000-character chunks, and 100-character
overlap. Offsets refer to unchanged source text. Chunk IDs include document ID,
offsets, content hash, and chunking settings. Title, source, timestamp, and metadata
origin are preserved. Empty/whitespace chunks are skipped; duplicate IDs fail.
The baseline uses BM25+ (`rank-bm25`), k1=1.5, b=.75, delta=1, lowercase Unicode
word tokens. This variant is recorded explicitly rather than called BM25 Okapi.
Nonmatching chunks are not returned.

Dense build (downloads the pinned BGE-small model once; local CPU inference):

```sh
python -m src.retrieval.index_cli build --input .cache/audit/documents.jsonl --output .cache/indexes/dense --method dense --limit 8 --chunk-size 600 --overlap 60
python -m src.retrieval.index_cli search --index .cache/indexes/dense --query 'How can I undo a deployment?' --benchmark-all-docs
```

Dense retrieval uses normalized embeddings, FAISS inner product (cosine), and the
[BGE query prefix](https://huggingface.co/BAAI/bge-small-en-v1.5). On macOS, the
module defaults `OMP_NUM_THREADS` to 1 before native libraries load, avoiding the
threading crash observed during integration. Set this before starting Python if
your application imports PyTorch/FAISS first. Oversize inputs fail instead of silently truncating evidence;
reduce chunk size if needed. Model revision, normalization, input hash, chunking,
and checksums are recorded in local index artifacts. Only load trusted local
FAISS artifacts; checksums detect corruption, not a maliciously rewritten manifest.

Both retrievers restrict candidates to allowed document IDs and requested sources
before ranking. Unknown timestamps are excluded when a date filter is requested;
provided timestamps must include a timezone. The CLI returns no hits without
`--allow-doc ID` or the explicit `--benchmark-all-docs` override. That override is
for local synthetic benchmark work, not an application authentication mechanism.

For now BM25 statistics and dense search indexes are rebuilt over each authorized
subset per query. This favors simple, auditable filtering on small corpora and is
not a full-corpus performance solution. Indexes stay in ignored `.cache/`.

## Generation

`GroundedGenerator.generate(GenerationRequest(...))` returns `(Answer, Usage)`.
It sends only authorized chunks that fit the context budget. No usable evidence
returns an insufficient-evidence answer without a provider call. Unknown citations,
invalid JSON, and truncated responses return errors; only transient provider errors
get bounded retries. Citation validation does not prove semantic correctness.

Use `FakeProvider` for offline tests. For real calls, instantiate `AnthropicProvider`
with `AnthropicConfig`: explicit model, `allow_paid=True`, positive USD budget,
current input/output token prices, timeout, and output-token cap. Set the key through
`ANTHROPIC_API_KEY`; never commit it. The default is disabled and no paid calls were
made during development verification.

The single-worker provider counts tokens and reserves worst-case generation cost
before each attempt, including retries. Unused reservations are retained, so the
budget is conservative. Pricing is caller-supplied; this estimate is not a billing
guarantee. API call counts include token-count requests. Lost response usage is
unknown, not zero. Use one provider per sequential job; distributed/concurrent
budget enforcement is not implemented. The SDK's own retries are disabled.

## Metrics and checks

```sh
python -m src.evaluation.retrieval_metrics --input tests/fixtures/retrieval_predictions.jsonl
python -m pytest -q
```

The example predictions and retrieval documents are invented fixtures, not benchmark
results. Retriever `top_k` counts chunks; metric k counts unique documents. Supply enough
ranked chunk hits for the requested document cutoff and retain their order.
Metrics deduplicate document IDs before applying k; compute Recall@k and
binary nDCG@k; macro-average; report category slices, errors, and exclusions.
No relevance labels means N/A, not zero. Eligible error rows score zero. Adapters
must supply parent document IDs, not chunk IDs. Tests cover hand-calculated ranks,
empty inputs, duplicate chunks, unauthorized evidence, persistence, and generator
failure paths. Dense unit tests use a deterministic fake encoder to stay offline;
real BGE verification is a separate local integration check:

```sh
HF_HUB_OFFLINE=1 RUN_MODEL_TESTS=1 python -m pytest tests/test_dense_model.py -q
```

Verified locally: BM25 saved/reloaded 2,736 chunks from 256 audit documents;
BGE/FAISS saved/reloaded 169 chunks from eight audit documents. The real-model
semantic fixture and permission-scope test passed. These are smoke checks only.

The 256-document audit prefix is not a representative evaluation corpus. These
commands validate components, not answer-quality gains or Milestone 2 results.
