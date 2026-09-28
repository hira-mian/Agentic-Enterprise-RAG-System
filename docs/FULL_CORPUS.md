# Baselines on the full dataset

We compare two standard ways of finding documents, run over all 511,958 documents
of EnterpriseRAG-Bench. They are the starting point our agentic system must beat.

## At a glance

| | BM25+ | BGE-small + FAISS |
| --- | --- | --- |
| **Idea** | Keyword search: find documents that share the question's words | Meaning search: find documents with similar meaning, even with different words |
| **Assignment role** | Simple baseline | Open-source reference baseline |
| **Recall@10** | **0.692** | 0.468 |
| **nDCG@10** | **0.604** | 0.367 |
| **Time per question** | ~20 ms (CPU) | ~130 ms (GPU) |
| **Setup time** | ~5 minutes to build the index (CPU) | ~2 hours to embed the documents (GPU) |

**Bottom line:** keyword search is clearly better on this dataset, but the two
methods find different documents, so combining them looks promising.

## How we measure

We use the 300 **development** questions. 18 of them (high-level and
"information not found" questions) have no correct documents to find, so they are
not scored, leaving **282 scored questions**.

For each question, a method returns its 20 best documents. We then check:

- **Recall@k**: what share of the correct documents appear in the top *k*
  results. Example: a question with 2 correct documents, and 1 of them in the
  top 10, scores 0.5 Recall@10.
- **nDCG@k**: like recall, but finding a correct document at rank 1 counts more
  than at rank 10. 1.0 means every correct document is ranked at the very top.

Scores are averaged over all scored questions.

## The data

We build the dataset from the benchmark's GitHub repository, pinned to one version
so results are reproducible:

```sh
python -m src.data.corpus --clone
```

This downloads the repository (about 5 GB) and saves all documents into one file,
`.cache/corpus/documents.parquet`. All 722 documents that the 500 questions point
to are included.

## Baseline 1: BM25+ (keyword search)

**How it works:** BM25+ scores each document by how many of the question's words
it contains. Rare words (like a project codename) count more than common words
(like "the" or "update"), and very long documents are not unfairly favored. We
index each document's title and text as one unit and use the
[`bm25s`](https://github.com/xhluca/bm25s) library.

**Run it:**

```sh
python -m src.retrieval.corpus_bm25          # build the index (~5 min, ~8 GB memory)
python -m src.evaluation.run_retrieval --method bm25 --split development
```

## Baseline 2: BGE-small + FAISS (meaning search)

**How it works:**

1. The pretrained model [BGE-small](https://huggingface.co/BAAI/bge-small-en-v1.5)
   turns text into a list of numbers (an "embedding"). Texts with similar meaning
   get similar numbers.
2. The model reads at most 512 tokens (word pieces) at a time, and most documents
   are longer, so each document is split into overlapping passages. The full
   dataset becomes 1,724,015 passages.
3. [FAISS](https://github.com/facebookresearch/faiss) finds the passages most
   similar to the question. Each document gets the score of its best passage.

**Run it:** embedding 1.7 million passages takes more than a day on a normal
computer, so we run it on a GPU. Open
[`notebooks/dense_baseline_colab.ipynb`](../notebooks/dense_baseline_colab.ipynb)
in Google Colab, select a T4 GPU, and run all cells. On a free T4 the embedding
took 1 hour 50 minutes. Progress is saved to Google Drive, so if Colab
disconnects, run the cells again and it continues where it stopped.

## Results

Development questions, 282 scored. **Bold** marks the better method.

| Question type | Questions | BM25+ Recall@10 | BGE Recall@10 | BM25+ nDCG@10 | BGE nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Overall** | 282 | **0.692** | 0.468 | **0.604** | 0.367 |
| Basic | 105 | **0.771** | 0.505 | **0.674** | 0.375 |
| Semantic (reworded) | 75 | **0.413** | 0.160 | **0.281** | 0.084 |
| Intra-document reasoning | 24 | **1.000** | 0.750 | **0.911** | 0.641 |
| Project related | 24 | **0.676** | 0.543 | **0.634** | 0.475 |
| Constrained | 18 | **0.917** | 0.861 | **0.820** | 0.705 |
| Completeness | 12 | **0.497** | 0.334 | **0.516** | 0.369 |
| Conflicting info | 12 | **0.792** | 0.625 | **0.810** | 0.512 |
| Miscellaneous | 12 | **0.917** | 0.750 | **0.886** | 0.641 |

Recall at other cutoffs:

| | Recall@5 | Recall@10 | Recall@20 |
| --- | ---: | ---: | ---: |
| BM25+ | **0.625** | **0.692** | **0.747** |
| BGE-small | 0.382 | 0.468 | 0.525 |

Categories with 12–24 questions are small, so treat their scores as rough.

## What we learned

1. **Keyword search wins everywhere.** BM25+ is better in every question type.
   Question by question, BM25+ does better on 88, BGE on 14, and they tie on 180.
   The average gap in Recall@10 is 0.22 (95% confidence interval 0.17–0.28), so
   this is not luck.
2. **The BGE result is real, not a bug.** We independently recomputed BGE's
   scores for 10 questions it missed and got the same numbers. The correct
   documents simply score lower than the ones BGE returned.
3. **Why BGE struggles (our current explanation).** BGE gives many related
   documents nearly the same score (the 1st and 20th results often differ by only
   0.01–0.05), so similar documents crowd out the right one. The dataset is also
   full of internal codenames and IDs, which exact word matching handles well and
   a small general-purpose model does not.
4. **Reworded questions are hard for both.** "Semantic" questions avoid the
   document's wording, which should help meaning search, yet BGE scores lower
   (0.160 vs 0.413). We have not investigated why yet; this belongs in the error
   analysis.
5. **The methods complement each other.** If we count a document as found when
   either method has it in its top 10, Recall@10 rises to 0.731, higher than
   BM25+ alone. This supports combining both in a hybrid search.

**Ideas to improve meaning search (not tried yet):** shorter passages, a larger
embedding model, or re-ranking the top results with a stronger model.

**Not run yet:** the final split (100 questions) stays untouched until our
settings are final.

## Where the results are

| Folder | Contents |
| --- | --- |
| `evaluation/results/bm25_full_development/` | BM25+ results |
| `evaluation/results/dense_full_development/` | BGE results |

Each folder has:

- `summary.json`: all scores, including breakdowns by source and by number of
  correct documents
- `predictions.jsonl`: the top 20 documents returned for each question
- `upstream_answers.jsonl`: the top 10 in the benchmark's own format

## Reproducibility details

- **Dataset version:** upstream commit `d36685e273713975ee20299bbf1ab64165575b3c`.
  Counts and a content fingerprint are in `evaluation/corpus_manifest.json`.
- **Duplicate IDs:** the upstream repository has 511,962 files, but four document
  IDs are each used by two different files. We keep the file that upstream's own
  index (`uuid_index.json`) points to.
- **BM25+ settings:** k1=1.5, b=0.75, delta=1, lowercase word tokens. Our index
  builder is tested to rank documents the same way as the `bm25s` and `rank_bm25`
  libraries.
- **BGE settings:** model `BAAI/bge-small-en-v1.5` at revision
  `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`; 500-token passages with 50-token
  overlap; half precision (fp16) on the GPU; BGE's standard query prefix; exact
  cosine search.
- **Runs:** BM25+ ran on 4 CPUs. BGE ran on a Colab T4 from commit `e924855`. The
  BGE index's corpus file hash differs from the local one because the file was
  rebuilt on Colab; the document content fingerprint is identical.
