# Evaluation artifacts

- `splits.json`: frozen question IDs, dataset revision, seed, category counts,
  and development regression IDs. Do not tune on the final partition.
- `data_audit.json`: bounded document/question profiles and missing reference IDs
  for development questions. This is corpus coverage, not retrieval performance.
- `development_examples.json`: one real development question per category, with
  original IDs and relevance labels. From Onyx's EnterpriseRAG-Bench (MIT declared
  in its dataset card), revision recorded in `splits.json`:
  https://huggingface.co/datasets/onyx-dot-app/EnterpriseRAG-Bench

Reproduce with `python -m src.data.audit` from the project root. Full cached
question records and the 256-document prefix live in ignored `.cache/audit/`.
See `docs/DATA_CARD.md` and `docs/EVALUATION.md` for limitations and protocol.

These are data-preparation artifacts. No baseline results exist yet.
