# Search agents

The agent searches for documents, checks whether they answer the question, and
searches again if needed. It generates an answer once the evidence is sufficient.

- **Critic:** identifies missing facts and suggests follow-up searches.
- **Search loop:** stops when evidence is sufficient, no new evidence appears,
  a limit is reached, or an error occurs.
- **Routing:** chooses text search, registered structured tools, or both.
  Real database/API tools are not connected yet.

Offline tests pass. Live LLM quality and the critic's confidence threshold have
not been evaluated. Dense/hybrid retrieval still needs an adapter that returns
text and citations.

## Run the pilot

Follow the [README setup](../README.md#quick-start), then obtain these matching
files from Hira or build them using the README's dataset and BM25 commands:

```text
.cache/corpus/
├── documents.parquet
├── questions.jsonl
└── indexes/bm25/     # Entire index folder
```

Check that the files load correctly. This makes **no paid calls**:

```sh
python -m src.agents.pilot
```

For a live run, set `ANTHROPIC_API_KEY` in your terminal environment. Replace the
placeholders below with your model, total dollar budget, and current prices per
million tokens:

```sh
python -m src.agents.pilot --live \
  --model MODEL_ID --budget-usd TOTAL_DOLLARS \
  --input-price INPUT_DOLLARS_PER_MILLION \
  --output-price OUTPUT_DOLLARS_PER_MILLION
```

The pilot uses BM25 and text routing for five development questions, with up to
two search rounds. The dollar budget covers all model calls in that run; starting
another run starts a new budget. Time/token limits are checked between calls.
Reference answers never go to the agent. This benchmark pilot allows access to
all indexed documents; it does not test production permissions.

## Review the results

Results stay local in `.cache/agent-pilot/<timestamp>/`:

- `case_*.json`: answers, retrieved text, searches, model responses, and usage.
- `review.json`: reference answers and blank fields for your review.
- `summary.json`: stop reasons, tokens, cost, and time.
- `manifest.json`: model, prompts, settings, and data/code versions.

Check whether the critic was right, follow-up searches helped, and answers were
correct, complete, and supported by citations. This is a small debugging pilot,
not a completed answer-quality evaluation.

## Run tests

```sh
python -m pytest tests/test_agents.py -q
```

Tests use scripted responses and make no paid calls.
