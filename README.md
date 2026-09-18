# Agentic RAG for Enterprise Knowledge Search

A course project for searching enterprise knowledge and generating answers with
sources. Currently includes a dataset loader, starter modules, and a smoke test.

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

CI runs the smoke test on every push and PR. It checks the test runner only.

Inspect dataset schemas and one sample per subset (requires internet):

```sh
python -m src.data.loader
```

Starter modules live in `src/{data,retrieval,agents,generation,evaluation,api}`.
Use `notebooks/exploration.ipynb` for exploration with the project environment.

## Contributing

PR flow: personal branch → `staging` → `main`.
See [Contributing](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md).

## Team

Jane Choi, Hira Mian, Alexander Xykis. 

## License

[MIT](LICENSE).
