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

## Next: BGE-small + FAISS

Dense retrieval will use the same corpus, runner, and metrics. Embedding about
500k documents is slow on CPU, so the embedding step should be resumable and able
to run on a GPU.
