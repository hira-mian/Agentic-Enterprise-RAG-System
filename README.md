# Agentic RAG for Enterprise Knowledge Search

A course project that answers questions about a company's internal documents
(Slack, email, tickets, wikis, and more) and cites the documents it used.

**Status:** the two retrieval baselines are done and evaluated on the full
dataset. The agentic search (Search Agent and Search Critic) is next.

## How it will work

```text
Question → Search Agent → Retrieve documents → Search Critic ("is this enough?")
         → search again if needed → Generate answer with sources
```

## Dataset

[EnterpriseRAG-Bench](https://github.com/onyx-dot-app/EnterpriseRAG-Bench) is a
synthetic company, "Redwood Inference", with:

- **511,958 documents** from 9 sources (Slack, Gmail, Linear, Google Drive,
  HubSpot, Fireflies, GitHub, Jira, Confluence)
- **500 questions**, each labeled with the documents that answer it

We split the questions into **development** (300, for building and comparing),
**calibration** (100, for tuning later), and **final** (100, untouched until the
end). The split is in `evaluation/splits.json`.

## Results so far

Two baselines, scored on the 300 development questions (282 have labeled answer
documents):

| Baseline | What it does | Recall@10 | nDCG@10 |
| --- | --- | ---: | ---: |
| **BM25+** | Keyword search: matches the question's words | **0.692** | **0.604** |
| **BGE-small + FAISS** | Meaning search: matches similar meaning | 0.468 | 0.367 |

- **Recall@10**: share of the correct documents found in the top 10 results.
- **nDCG@10**: also rewards ranking the correct documents near the top (1.0 is perfect).

What this means:

- Keyword search wins in every question type, likely because this dataset is
  full of project codenames and IDs, which exact word matching handles well.
- The two methods find different documents. Together, their top 10 results contain
  about 73% of the correct documents on average, so a hybrid of both is a promising next step.

Full results by question type: [docs/FULL_CORPUS.md](docs/FULL_CORPUS.md).

## Quick start

Set up Python 3.11:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest
```

**1. Build the dataset** (downloads about 5 GB, then takes about a minute):

```sh
python -m src.data.corpus --clone
```

**2. Run the BM25+ baseline** (about 5 minutes; needs about 8 GB of memory):

```sh
python -m src.retrieval.corpus_bm25
python -m src.evaluation.run_retrieval --method bm25 --split development
```

**3. Run the BGE baseline** (needs a GPU; about 2 hours on a free Colab T4):
open [`notebooks/dense_baseline_colab.ipynb`](notebooks/dense_baseline_colab.ipynb)
in Colab and run all cells.

Results are saved to `evaluation/results/`.

## Repository layout

| Folder | Contents |
| --- | --- |
| `src/data/` | Building the dataset |
| `src/retrieval/` | Search methods (BM25+, BGE) |
| `src/evaluation/` | Scoring and the evaluation runner |
| `src/generation/` | Writing answers with an LLM (paid calls are off by default) |
| `src/agents/`, `src/api/` | Planned: Search Agent, Search Critic, web API |
| `evaluation/` | Question splits, dataset record, and results |
| `docs/` | Documentation |
| `notebooks/` | Colab notebook for the BGE baseline |
| `tests/` | Automated tests (run on every push) |

## Documentation

- [Baselines on the full dataset](docs/FULL_CORPUS.md): methods, results, findings
- [Data card](docs/DATA_CARD.md): the dataset in detail
- [Evaluation plan](docs/EVALUATION.md): metrics and how answers will be graded
- [Answer generation and metrics](docs/GENERATION.md): the LLM answer writer and search metrics
- [Shared contracts](docs/CONTRACTS.md): data formats used across the code

## Contributing

Workflow: personal branch → `staging` → `main`.
See [Contributing](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md).

## Team

Hira Mian, Jane Choi, and Alexander Xykis.

## License

[MIT](LICENSE).
