# Full-corpus baselines

Baselines run over all 511,958 EnterpriseRAG-Bench documents, not the 256-document
inspection sample.

## Corpus

The corpus comes from the upstream generator repository at a pinned commit
(`CORPUS_COMMIT` in `src/config.py`), which stores each document as a JSON file.

```sh
python -m src.data.corpus --clone
```

This clones the repository into `.cache/corpus/` (about 5 GB), converts every
document to `.cache/corpus/documents.parquet` (about 1 GB), copies `questions.jsonl`,
and writes `evaluation/corpus_manifest.json`. It takes about a minute after cloning.

- Title and content use upstream's own field labels (`title_field_name`,
  `content_field_names`), matching the Hugging Face export.
- Four document IDs are reused by two different files each. We keep the file
  upstream's `uuid_index.json` points to and list the others under `skipped`.
- All 722 reference documents for the 500 questions are present. The questions
  match the Hugging Face revision used for `evaluation/splits.json`.

## BM25+ (simple baseline)

```sh
python -m src.retrieval.corpus_bm25
python -m src.evaluation.run_retrieval --method bm25 --split development
```

- Unit: one entry per document (title + content), like upstream's baselines.
  Scoring is per document, so no chunk deduplication is needed.
- Scoring: BM25+ via [`bm25s`](https://github.com/xhluca/bm25s) with the same
  tokenizer and settings as `src/retrieval/bm25.py` (k1=1.5, b=0.75, delta=1).
  Tests check the scores match `bm25s` and `rank_bm25`.
- The index is built with NumPy in one streaming pass: about 4.5 minutes and 7 GB
  of peak memory on 4 CPUs. It is stored in `.cache/corpus/indexes/bm25` (about 2 GB).
- Queries take about 21 ms each. Benchmark runs search every document. The
  contract-compatible `search()` applies permissions and source filters.

The runner writes `predictions.jsonl`, `summary.json`, and `upstream_answers.jsonl`
(upstream's answer format, top 10 documents, for their scorer) to
`evaluation/results/<method>_full_<split>/`.

## Results: development split (300 questions)

Recall@k and nDCG@k count unique documents. 18 questions have no reference
documents (`high_level`, `info_not_found`) and are excluded from retrieval scores.

| Question type | Scored | Recall@5 | Recall@10 | Recall@20 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Overall** | 282 | 0.625 | 0.692 | 0.747 | 0.604 |
| basic | 105 | 0.743 | 0.771 | 0.800 | 0.674 |
| semantic | 75 | 0.307 | 0.413 | 0.480 | 0.281 |
| intra_document_reasoning | 24 | 0.958 | 1.000 | 1.000 | 0.911 |
| project_related | 24 | 0.523 | 0.676 | 0.816 | 0.634 |
| constrained | 18 | 0.806 | 0.917 | 0.972 | 0.820 |
| completeness | 12 | 0.393 | 0.497 | 0.622 | 0.516 |
| conflicting_info | 12 | 0.792 | 0.792 | 0.917 | 0.810 |
| miscellaneous | 12 | 0.917 | 0.917 | 0.917 | 0.886 |

BM25+ is strong when questions reuse document wording and weak on `semantic`
questions, which avoid keyword overlap. Source and single/multiple-reference slices
are in `summary.json`. Small categories have wide uncertainty.

The final split stays untouched until settings are frozen.

## BGE-small + FAISS (open-source reference baseline)

Uses the pinned `BAAI/bge-small-en-v1.5` model from Hugging Face with the same
corpus, runner, and metrics as BM25+.

- Unit: the median document is about 1,000 BGE tokens, and only 6% fit in the
  model's 512-token limit. Each document (title + content) is split into
  500-token windows with 50-token overlap, about 1.5 million passages in all.
- Scoring: exact FAISS inner product over normalized embeddings (cosine). A
  document's score is its best passage's score. Queries use BGE's retrieval prefix.
- Embeddings are stored as float16 shards, one per 20,000 documents. Completed
  shards are skipped on rerun, so interrupted runs resume. fp16 inference is the
  default on GPU and is recorded in the manifest.

Embedding on a CPU would take more than a day, so run it on a GPU. The easiest
way is [`notebooks/dense_baseline_colab.ipynb`](../notebooks/dense_baseline_colab.ipynb)
on a free Colab T4. It builds the corpus, embeds, scores the development split,
and saves everything to Google Drive. Equivalent commands:

```sh
python -m src.retrieval.corpus_dense --device cuda --batch-size 256
python -m src.evaluation.run_retrieval --method dense --device cuda --split development
```

Results are in `evaluation/results/dense_full_development/`. They were run on a
Colab T4 from commit `e924855` with no local changes. The embeddings cover all
511,958 documents in 26 shards. The index's `corpus_sha256` differs from the local
BM25+ index because the Parquet file was rebuilt on Colab. The document content
hash (`content_sha256`) is identical.

## Comparison: development split (282 scored questions)

| Question type | Scored | BM25+ R@10 | BGE R@10 | BM25+ nDCG@10 | BGE nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Overall** | 282 | **0.692** | 0.468 | **0.604** | 0.367 |
| basic | 105 | 0.771 | 0.505 | 0.674 | 0.375 |
| semantic | 75 | 0.413 | 0.160 | 0.281 | 0.084 |
| intra_document_reasoning | 24 | 1.000 | 0.750 | 0.911 | 0.641 |
| project_related | 24 | 0.676 | 0.543 | 0.634 | 0.475 |
| constrained | 18 | 0.917 | 0.861 | 0.820 | 0.705 |
| completeness | 12 | 0.497 | 0.334 | 0.516 | 0.369 |
| conflicting_info | 12 | 0.792 | 0.625 | 0.810 | 0.512 |
| miscellaneous | 12 | 0.917 | 0.750 | 0.886 | 0.641 |

Recall@5 is 0.625 for BM25+ and 0.382 for BGE; Recall@20 is 0.747 and 0.525.
Queries take about 20 ms for BM25+ (CPU) and 130 ms for BGE (T4 GPU, including
query encoding).

Findings:

- BM25+ beats BGE-small in every category. Per question, BM25+ has higher
  Recall@10 on 88 questions, BGE on 14, and 180 tie. The mean paired difference is
  -0.224 (95% bootstrap CI -0.280 to -0.169).
- The BGE result is not a pipeline error. For 10 missed questions we re-embedded
  the retrieved and reference documents independently and reproduced the Colab
  scores to within 0.001. Reference documents simply score below the top 20.
- Scores are compressed: top-1 and 20th-ranked scores are often only 0.01–0.05
  apart, so near-duplicate topical documents crowd out the answer. The corpus uses
  internal codenames and identifiers that exact word matching handles well and a
  small general-purpose embedding model does not.
- `semantic` questions were expected to favor dense retrieval but do not
  (0.160 vs 0.413). We have not yet examined why; that belongs in the error
  analysis.
- The methods are partly complementary. Counting a reference document as found
  if either method's top 10 contains it gives Recall@10 of 0.731, above BM25+
  alone. That supports the planned hybrid fusion.
- Possible dense improvements, not tried: shorter passages (for example 128–256
  tokens) to reduce dilution of specific facts, a larger embedding model, or
  reranking.
