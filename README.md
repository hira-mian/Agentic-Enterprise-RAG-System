# Agentic RAG for Enterprise Knowledge Search

A course project for searching enterprise knowledge and generating answers with
sources. Currently a scaffold with a basic smoke test.

[GitHub repository](https://github.com/hira-mian/Agentic-Enterprise-RAG-System)

## Planned Architecture

```text
Search Agent → Retrieval/Tools → Search Critic
→ Adaptive Search → Generate → Answer + Sources
```

## Dataset

EnterpriseRAG-Bench (planned).

## Planned Tech Stack

Python, FastAPI, BM25, dense retrieval, RRF, cross-encoder reranking,
hosted LLM API, and React.

## Setup

Use Python 3.11:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest
```

CI runs the smoke test on every push and PR. It checks the test runner only.

## Contributing

PR flow: personal branch → `staging` → `main`.
See [Contributing](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md).

## Team

Jane Choi, Hira Mian. Additional members: TBD.

## License

[MIT](LICENSE).
