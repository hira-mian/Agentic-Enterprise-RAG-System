# Data Card — MultiHop-RAG (as used in this project)

## Summary

| | |
|---|---|
| **Dataset** | MultiHop-RAG (Tang & Yang, COLM 2024, [arXiv:2401.15391](https://arxiv.org/abs/2401.15391)) |
| **Task** | Retrieval-augmented question answering where the evidence for each query is spread across 2–4 documents |
| **Corpus** | 609 English news articles, 49 publishers, published 26 Sep – 25 Dec 2023 |
| **Queries** | 2,556 queries with gold answer, question type, and gold evidence list (article URL + supporting sentence) |
| **Licence** | ODC-BY 1.0 (Open Data Commons Attribution) for the dataset |
| **Access** | `python scripts/download_data.py`. It downloads the two JSON files from the upstream GitHub repository, pinned to commit `89d5b19`, and checks their SHA-256 hashes |
| **Splits** | Queries only: dev 511 / test 2,045 (20/80, stratified by question type, seed 42). The corpus is the same for both splits |

## Why this dataset

The project is building an **agentic** RAG system for enterprise knowledge bases. Enterprise questions often need facts from several documents at once. They also often refer to document metadata ("according to the Q3 report from Finance…", "what changed after the October memo?"). MultiHop-RAG is one of the few public benchmarks that tests exactly this:

* every non-null query needs evidence from **2–4 separate articles**
* queries refer to **metadata**: 99.4% of the queries that have evidence name the publisher, and temporal queries depend on publication dates
* there are **null queries** that cannot be answered from the corpus, which tests whether the system abstains instead of making up an answer
* gold evidence is given at the article level, so we can **score retrieval separately from generation**

The corpus was also published after the knowledge cutoff of the LLMs used to build the dataset. That lowers, though does not remove, the chance that an LLM can answer from memory without retrieval.

## Sourcing and access

* **Upstream:** <https://github.com/yixuantt/MultiHop-RAG> (files `dataset/corpus.json` and `dataset/MultiHopRAG.json`). The files are also mirrored on Hugging Face as `yixuantt/MultiHopRAG`.
* **Pinned version:** Git-LFS objects at commit `89d5b19`

  | File | Size | SHA-256 |
  |---|---|---|
  | `corpus.json` | 6,785,567 B | `20b61b5ab84de84a927420c5d265b7ec8d859ae49980699958a787ade9e4d28f` |
  | `MultiHopRAG.json` | 5,171,312 B | `03cfb4926461f868684903aadc8024447bdda5bb3f6804741424cce338515bff` |
* Raw files go to `data/raw/` and are **not committed**; the download script is the source of truth. The split definitions (`data/splits/{dev,test}.json`) are only lists of query IDs and **are committed**.
* Query IDs `qNNNN` are the zero-padded position of the query in `MultiHopRAG.json`.

## Provenance

* **Articles:** collected by the dataset authors from a news API (the mediastack API, per the paper) covering Sep–Dec 2023. Each record has `title, author, source, published_at, category, url, body`. The body is scraped text, so it can include page leftovers ("Table of Contents…", deal lists, bylines).
* **Queries and answers:** made by a **GPT-4-based pipeline** in the paper: extract claims from articles → link claims across articles through shared entities or topics → generate the query and answer with an LLM → filter with an LLM and by hand. Answers are therefore partly machine-generated labels.
* **Corpus composition:** sports 211, technology 172, entertainment 114, business 81, science 21, health 10 articles. The largest publishers are Sporting News (101), TechCrunch (97), The Verge (45), Polygon (44), The Guardian (29) and Fortune (25).
* **Query composition** (all 2,556): comparison 856, inference 816, temporal 583, null 301. Distinct evidence articles per query: 2 (1,079), 3 (778), 4 (398), 0 (301 null queries).

## Licensing and usage constraints

* The dataset is released under **ODC-BY 1.0**: use, sharing and adaptation are allowed with attribution. Cite Tang & Yang (2024) in any report or derived artifact.
* The **article text is third-party copyrighted news content**. We use it only for non-commercial academic evaluation. We do not redistribute it: it is not committed to this repository, and prediction logs store article URLs, not article text.
* Using OpenAI models means sending article snippets and queries to a third-party API under that provider's terms. The open-source path (FastEmbed + Ollama) runs fully locally.
* There is no personal data beyond the names of public figures and journalist bylines that appear in published news.

## Splits

| Split | Queries | comparison | inference | temporal | null | Use |
|---|---|---|---|---|---|---|
| dev | 511 | 171 | 163 | 117 | 60 | Prompt and parameter tuning, error analysis, fitting the majority baselines |
| test | 2,045 | 685 | 653 | 466 | 241 | Reported numbers only; never used for tuning |

* **No training split.** Every system in this project is zero-shot (pretrained embedders and LLMs with prompting and orchestration), so no data is needed for fitting weights. If fine-tuning is added later (for example a reranker), it must be trained on dev or external data only.
* **Stratified by question type** so that per-type scores are comparable across the two splits. `scripts/make_splits.py` recreates the split exactly.
* **The corpus is shared across splits.** This is the retrieval setting of the original benchmark and of a real deployment: the knowledge base is fixed and the questions vary. Dev and test queries can use the same articles as evidence, so dev tuning can overfit to specific articles. We accept this and report results on test only.

## Known limitations

1. **Label noise and inconsistent answer formats.** Yes/no answers appear as `Yes`, `no` and `No`. Some comparison and temporal answers use other words (`True`, `Consistent`, …). The scorer normalises case and punctuation, but paraphrased labels such as "Consistent" and "Yes" are not treated as the same answer. The LLM-generated labels were not fully checked by people.
2. **Skewed answer distribution.** 53% of all answers are yes/no. Among inference queries, "Sam Bankman-Fried" (271) and "Google" (211) together account for 59% of the answers. A prior-only baseline therefore scores well above zero (see the majority baselines), and an LLM can sometimes guess from the question wording alone.
3. **Evidence granularity.** Gold evidence is one sentence per article, but we score retrieval at the article level. A retrieved chunk from the right article may not contain the supporting sentence (the corpus has 2,078 chunks for 609 articles, about 3.4 per article). 168 queries cite two facts from the same article.
4. **Domain mismatch with enterprise data.** The corpus is public news, not internal enterprise documents. It has no access-control boundaries, tables, PDFs/slides, internal jargon, or versioned duplicate documents. Topic coverage is uneven (sports and tech make up 63%, health 2%). Conclusions about enterprise use cannot be taken directly from these results. We plan to add an enterprise-style corpus as a secondary evaluation.
5. **Scraping artifacts.** Some bodies contain boilerplate, navigation text or long deal lists. Article length ranges from 839 to 12,387 words (median 1,298). 68 articles have no author.
6. **Short time window and publisher bias.** All articles are from Sep–Dec 2023 and from mostly US/UK English-language outlets, so there are temporal and geographic coverage gaps.
7. **Possible model memorisation.** The events are recent enough to be mostly outside older LLMs' training data, but newer LLMs may have seen them. Comparing answer accuracy with and without retrieval is therefore informative.
