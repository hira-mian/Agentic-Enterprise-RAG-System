# Evaluation design

The harness (`python -m enterprise_rag.evaluate`) uses an existing benchmark, **MultiHop-RAG** (see [DATA_CARD.md](DATA_CARD.md)). It adds fixed dev/test splits, document-level retrieval metrics, and a deterministic answer scorer. Every metric below is written for every query to `results/<run>/predictions.jsonl`, and the per-query rows are averaged into `summary.json`. Averages are reported overall and per question type.

## What we measure and why

The target system is an *agentic* RAG assistant for enterprise knowledge bases. It succeeds when it (1) **finds all the documents** a question depends on, (2) **answers correctly from them**, and (3) **declines to answer** when the knowledge base does not contain the answer. We measure each of these separately, so we can tell whether a failure came from retrieval or from generation.

### Retrieval metrics (document level, queries with evidence only)

A retrieved chunk counts toward the article it came from, matched by article URL. Several chunks from the same article count once. `k` is the number of retrieved **chunks**.

| Metric | Definition | Why it matters |
|---|---|---|
| **Hit@k** | At least one gold article is in the top-k | Usual single-hop retrieval measure; shows whether retrieval is on topic |
| **Recall@k** | Share of the gold articles found in the top-k | **Main retrieval metric.** Multi-hop answers need every piece of evidence |
| **All-evidence@k** | Every gold article is in the top-k | Strict: an upper bound on how often the generator even has what it needs |
| **MRR@k** | 1 / rank of the first gold article | Ranking quality |

`k = 2` is what the default LlamaIndex pipeline actually gives the LLM. We also report `k = 5, 10` as **diagnostics**: they show whether evidence is missing entirely or just ranked too low, which decides the fix (better retrieval vs. retrieving more / reranking / query decomposition).

### Answer metrics (all queries)

| Metric | Definition |
|---|---|
| **Answer accuracy** (primary) | *Null queries:* correct if the system declines (e.g. "insufficient information", "the context does not provide…"; the patterns are in `enterprise_rag/metrics.py`). *Yes/no gold:* the first `yes`/`no` token in the prediction must match. *Other gold:* the normalised gold answer must appear as a token span in the normalised prediction. This follows the MultiHop-RAG evaluation, which allows verbose answers |
| **Exact match** | Normalised prediction == normalised gold (lowercase, no punctuation or articles). Strict; penalises verbose but correct answers |
| **Token F1** | SQuAD-style token overlap F1 |

Scoring is deterministic and free: there is no LLM judge in the automatic metrics, so any run can be reproduced exactly. Answer quality beyond these string matches is covered by the manual [qualitative rubric](QUALITATIVE_RUBRIC.md).

### Efficiency

Average `latency_retrieval_s` and `latency_generation_s` per query. These are recorded because agentic pipelines trade extra LLM calls for accuracy, and that cost needs to be tracked.

## Test design

* **Split:** results are reported on the 2,045-query **test** split. The 511-query **dev** split is used for tuning and for fitting the majority baselines.
* **Question types**, each reported separately:
  * *inference*: combine facts about one entity from different articles ("Which company…, as reported by X and Y?")
  * *comparison*: compare claims across sources ("Does X report … while Y reports …?")
  * *temporal*: order events by publication date or claim time ("Did X report … before Y reported …?")
  * *null*: the answer is not in the corpus; the right behaviour is to decline
* **Sanity checks built into the baselines:** a prior-only **majority** baseline shows how much can be scored without reading anything. A **BM25** baseline shows how much a plain keyword retriever achieves. The LlamaIndex default must beat both to be worth its cost.
* **Unit tests** for the scorer: `pytest tests/`.

## Known caveats of the metrics

* Containment accuracy can reward an answer that lists several candidates ("either Google or Apple"). The rubric's *correctness* criterion catches this during manual review.
* The abstention patterns are heuristic. An answer that hedges and then answers anyway can be counted as an abstention. We check the null-query rows in the review sample.
* Article-level recall does not guarantee that the retrieved chunk contains the supporting sentence (see Data Card, limitation 3).
