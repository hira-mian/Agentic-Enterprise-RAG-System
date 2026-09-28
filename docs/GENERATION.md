# Answer generation and metrics

How the answer generator and the search metrics work. To build the dataset and
run the search baselines, see [baselines on the full dataset](FULL_CORPUS.md).

## Answer generation

`GroundedGenerator.generate(request)` writes an answer from the documents found by
search and returns `(Answer, Usage)`.

- It only uses documents the user is allowed to see, up to a size limit.
- If no documents are found, it says it cannot answer instead of calling the LLM.
- Every citation must point to a document it was given. A malformed answer or an
  unknown citation is an error. (A valid citation does not guarantee the answer is
  correct.)
- Only temporary provider errors are retried.

**Using Claude:** set `ANTHROPIC_API_KEY` and pass an `AnthropicConfig` with the
model, `allow_paid=True`, a budget, token prices, a timeout, and a maximum answer
length.

- **Paid calls are off by default.** You must opt in with `allow_paid=True`.
- Before each call, the worst-case cost is reserved against the budget, so
  estimated spending stays within it. Prices are supplied by you, so costs are
  estimates, not a bill.
- Use one provider per job, run one call at a time, and count retries in one
  place (the SDK's own retries are turned off).

## Search metrics

For each question, the metrics compare the ranked documents a method returned with
the question's correct documents:

- **Recall@k:** share of the correct documents that appear in the top *k*.
- **nDCG@k:** like recall, but a correct document ranked higher counts more.
- Results are averaged overall and per question type. Questions with no correct
  documents are left out and counted separately. A question whose search failed
  scores zero.

The evaluation runner computes these for the full dataset (see
[baselines](FULL_CORPUS.md)). To try the metrics on a small made-up example:

```sh
python -m src.evaluation.retrieval_metrics --input tests/fixtures/retrieval_predictions.jsonl
```

## Tests

```sh
python -m pytest -q
```

The tests use small invented examples and run without internet. The Claude
integration is tested with a fake provider; the tests make no paid calls.
