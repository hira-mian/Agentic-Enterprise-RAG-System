# Agentic RAG for Enterprise Knowledge Search

A course project for searching enterprise knowledge and generating answers with
sources. Includes a dataset audit, shared contracts, BM25/dense retrieval, a generation
adapter, and retrieval metrics. The full evaluation runner and agents are not
implemented yet.

[GitHub repository](https://github.com/hira-mian/Agentic-Enterprise-RAG-System)

## Planned Architecture

```text
Search Agent → Retrieval/Tools → Search Critic
→ Adaptive Search → Generate → Answer + Sources
```

## Dataset

[EnterpriseRAG-Bench](https://huggingface.co/datasets/onyx-dot-app/EnterpriseRAG-Bench).
The loader streams the `documents` and `questions` test subsets.

## Planned Tech Stack

Python, FastAPI, BM25, dense retrieval, RRF, cross-encoder reranking,
hosted LLM API, and React.

## Setup

Use Python 3.11:

```sh
python3 -m venv .venv
source .venv/bin/activate
python --version
python -m pip install -r requirements.txt
python -m pytest
```

CI runs offline component tests on every push and PR. These do not measure
benchmark answer quality.

Inspect dataset schemas and one sample per subset (requires internet):

```sh
python -m src.data.loader
```

Starter modules live in `src/{data,retrieval,agents,generation,evaluation,api}`.
Use `notebooks/exploration.ipynb` for exploration with the project environment.

## Data and evaluation setup

Create a pinned 256-document inspection sample and question split artifacts:

```sh
python -m src.data.audit
```

Expected summary for the pinned revision:

```text
documents: 256; questions: 500
development: 300; calibration: 100; final: 100; golden examples: 60
```

Raw data stays in ignored `.cache/audit/`. The sample is not a complete evaluation
corpus. This command prepares data; it does not run baseline evaluation.

See [Data Card](docs/DATA_CARD.md), [shared contracts](docs/CONTRACTS.md), and
[evaluation protocol](docs/EVALUATION.md). Frozen split IDs and audit summaries
are in `evaluation/`. Do not regenerate them with different settings silently.

## Retrieval and generation

See [Wave 2 usage](docs/WAVE2.md) for index build/search commands, generator
configuration, and retrieval metrics. Paid generation is disabled by default.

## Contributing

PR flow: personal branch → `staging` → `main`.
See [Contributing](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md).

## Team

Jane Choi, Hira Mian. Additional members: TBD.

## License

[MIT](LICENSE).
