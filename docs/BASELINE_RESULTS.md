# Baseline results and initial error analysis

All numbers are on the **test split (2,045 queries)**. Retrieval metrics are computed on the 1,804 test queries that have gold evidence; the 241 null queries have none. Scores are regenerated with:

```bash
python scripts/make_results_table.py                     # -> results/RESULTS.md
python scripts/analyze_errors.py \
  results/llamaindex_default__test__fastembed-BAAI-bge-base-en-v1.5__retrieval-only \
  results/bm25__test__retrieval-only --chunk-stats       # error-analysis numbers below
```

## Baselines

| Baseline | Type | What it is |
|---|---|---|
| `majority_global` | Simple (prior only) | Always gives the most common dev-split answer ("Yes") |
| `majority_per_type` | Simple (prior only, uses the gold question-type label) | Most common dev answer per question type: comparison/temporal → "Yes", inference → "Sam Bankman-Fried", null → "Insufficient information." |
| `bm25` | Simple minimal retrieval | BM25 (bm25s, English stemming) over the **same** 2,078 chunks, top-2, with the same synthesizer and LLM as below |
| `llamaindex_default` | **Open-source reference** | `VectorStoreIndex.from_documents` + `as_query_engine()` with LlamaIndex defaults: 1024/200 sentence chunks, top-2 dense retrieval, `compact` synthesizer, default QA prompt. Embeddings: `BAAI/bge-base-en-v1.5` (FastEmbed). LLM: Ollama `llama3.1:8b` or OpenAI |

## Results table (test split)

| System | Hit@2 | Recall@2 | All-evidence@2 | MRR@2 | Recall@10 | Answer acc. | EM | Token F1 |
|---|---|---|---|---|---|---|---|---|
| Majority (global) | — | — | — | — | — | 0.306 | 0.306 | 0.306 |
| Majority (per type, oracle type) | — | — | — | — | — | 0.530 | 0.530 | 0.539 |
| BM25 (top-2) | **0.850** | **0.442** | **0.120** | **0.793** | **0.786** | *pending LLM run* | | |
| LlamaIndex default (bge-base, top-2) | 0.715 | 0.336 | 0.050 | 0.660 | 0.642 | *pending LLM run* | | |
| *Ceiling for any top-2 retriever* | *1.000* | *0.817* | *0.518* | *1.000* | — | | | |

Recall@2 by question type:

| System | inference | comparison | temporal |
|---|---|---|---|
| BM25 | 0.350 | 0.509 | 0.470 |
| LlamaIndex default | 0.254 | 0.373 | 0.395 |

Answer accuracy by question type (prior-only baselines):

| System | inference | comparison | temporal | null |
|---|---|---|---|---|
| Majority (global) | 0.000 | 0.599 | 0.461 | 0.000 |
| Majority (per type) | 0.334 | 0.599 | 0.461 | 1.000 |

**Answer metrics for the retrieval baselines are still pending.** The sandbox these results were produced in could not reach an LLM endpoint (neither OpenAI nor Ollama). The generation code path is implemented and smoke-tested with LlamaIndex's `MockLLM`. To complete the table, run:

```bash
python -m enterprise_rag.evaluate --system llamaindex_default --llm ollama:llama3.1:8b
python -m enterprise_rag.evaluate --system bm25               --llm ollama:llama3.1:8b
python scripts/make_results_table.py
```

Any RAG system must beat **0.530 answer accuracy** (the per-type prior) to show it is using the retrieved evidence.

## Initial error analysis

### 1. The default top-2 cannot hold multi-hop evidence (structural failure)

* 48.2% of the test queries with evidence need **3 or 4** articles, but two chunks can come from at most two articles. That caps Recall@2 at 0.817 for *any* retriever, and All-evidence@2 at 0.518.
* All-evidence@2 is **0.000** for every 3- and 4-article query, for both retrievers. Even for 2-article queries, dense retrieval gets both articles only 9.7% of the time (BM25: 23.1%).
* **48.2% of the dense top-2 lists (BM25: 36.4%) contain two chunks from the same article.** Half the context budget goes to redundant text. Default retrieval has no diversity step and no per-document deduplication.
* Recall rises steeply with depth (dense: 0.336 → 0.642 at k=10; BM25: 0.442 → 0.786). **76% of the dense Hit@2 misses have a gold article in the top-10** (BM25: 93%). Most of the evidence is findable; it is ranked or budgeted out.

*Why:* `similarity_top_k=2` was chosen for single-hop QA over small document sets. A multi-hop query needs at least one slot per hop.

### 2. Dense retrieval underperforms BM25 (unexpected)

The off-the-shelf dense pipeline is **worse than BM25 on every metric** (Recall@2 0.336 vs 0.442). BM25 alone gets a hit on 380 queries that dense misses; the reverse happens on only 136. We see two likely causes:

* **Chunks are longer than the embedder can read.** The LlamaIndex default chunk (1024 LlamaIndex tokens plus a metadata header) is 904 bge tokens on average. `bge-base-en-v1.5` truncates its input at 512 tokens, so **93.7% of chunks are truncated and on average 40.9% of each chunk's text is never embedded**. Evidence sentences in the second half of a chunk are invisible to dense retrieval but not to BM25. (With OpenAI's `text-embedding-3-small`, which accepts 8k tokens, this failure would not occur. The mismatch appears as soon as the default embedder is replaced with an open-source one.)
* **A compound query becomes one blended vector.** Multi-hop questions ask about two unrelated sub-topics ("…Davis Cup team… while… South Africa rugby…", "…Cowboys vs. 49ers stream… in contrast to Polygon's film availability…"). A single query embedding lands between the topics and retrieves articles on the *blend* ("Twitch is the best way to watch sports", "How the OpenAI fiasco could bolster Meta"). BM25 still matches the rare, distinctive terms of each hop.

### 3. Metadata constraints are ignored

99.4% of the queries with evidence name the publisher ("according to Fortune", "as reported by TechCrunch"). Temporal queries depend on publication dates. Neither retriever uses metadata as a filter: 15.4% of the dense top-2 articles (BM25: 10.9%) come from a publisher the query never mentions. The default metadata template also embeds the URL and timestamps, which adds noise rather than a usable constraint.

### 4. Inference queries are hardest to retrieve

Inference queries have the lowest Recall@2 (dense 0.254, BM25 0.350). They describe an entity only indirectly ("the individual associated with the cryptocurrency industry facing trial…") and never name it, so neither exact terms nor a single embedding reliably connects the two descriptions.

### 5. What the answer-prior baselines show

The global majority gets 30.6% by always answering "Yes". The per-type prior gets 53.0%, including 100% on null queries by always abstaining and 33% on inference queries by always answering "Sam Bankman-Fried". This means that (a) a RAG system's gains must be measured against 0.530, not 0; (b) yes/no questions are partly guessable, so we must also check with the rubric that yes/no answers are *justified*; and (c) null-query accuracy must be reported next to answerable-query accuracy, so a system cannot score well by abstaining too often.

### Expected generation-stage failures (to confirm once the LLM run is done)

* **Under-abstention** on null queries: the default LlamaIndex QA prompt never tells the model it may say the information is missing.
* **Verbose answers**: the default prompt asks for no specific format, so EM will be far below containment accuracy.
* **Partial-evidence reasoning** (`R-partial` in the rubric): with 0–5% complete evidence, most comparison and temporal answers will be decided on half the evidence.

These will be measured with the rubric failure categories on a 40-query review sample (`scripts/export_review_sample.py`).

## What we will change next (toward the agentic system)

| Finding | Planned change | Metric that should move |
|---|---|---|
| top-2 cannot cover 3–4 hops; half the slots are duplicates | Retrieve deeper (k≈10) with per-article dedup, then rerank (cross-encoder) down to the context budget | Recall@k, All-evidence@k |
| A compound query becomes one blended embedding | **Agentic query decomposition**: an LLM planner splits the query into one sub-query per hop and retrieves for each | Recall@2 per hop, All-evidence |
| Dense < BM25; truncation | Hybrid BM25 + dense (reciprocal-rank fusion); chunk at 512 bge tokens (or use a long-context embedder) | Recall@2, Hit@2 |
| Publisher and date constraints ignored | Extract metadata filters from the query (source, date range) as retrieval tool arguments; exclude the URL and timestamp from embedded text | Off-constraint rate, temporal-query accuracy |
| No abstention instruction | Answer prompt with an explicit "insufficient information" option and a short-answer format; a verification step checks that evidence covers every hop before answering | Null-query accuracy without losing answerable-query accuracy; EM |

Every change will be ablated with this harness on **dev** and reported once on **test**.
