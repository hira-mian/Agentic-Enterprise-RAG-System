# Evaluation protocol

Retrievers, generation adapter, splits, and retrieval metrics are implemented.
The full runner and answer scoring remain to be built. No benchmark results yet.

## Data splits

Use the revision in `src/config.py` and frozen IDs in `evaluation/splits.json`.
These are our partitions of the upstream `test` split:

| Set | Questions | Use |
| --- | --- | --- |
| Development | 300 | Baselines, debugging, and model selection |
| Calibration | 100 | Critic thresholds and evidence-sufficiency labels |
| Final | 100 | Evaluation after settings are frozen |

Seed: 5980. Questions sharing reference documents or identical normalized text
stay together, including transitive overlap. Assignment balances question types
at a target ratio of 60/20/20; keeping groups intact takes priority. Other shared
projects and near-duplicates may still cross splits.

The development regression set contains six questions per category (60 total).
`evaluation/development_examples.json` includes one per category. Use the full
development set for baseline tables. Version changes to splits; do not tune on
final outcomes. Documents form a shared search corpus. Reference answers belong
only in the evaluator. We do not train a model or use benchmark data for training.

## Baselines

Compare BM25+ and pretrained BGE-small dense retrieval with the same corpus,
questions, generator/prompt, top-k, context budget, and access policy. Add hybrid
retrieval and agentic search later. Record code commit, model/dependency versions,
seed, corpus hash, and settings. These are project comparisons, not official
leaderboard reproductions.

## Metrics

| Metric | Definition |
| --- | --- |
| Recall@5/10/20 | Relevant documents retrieved / all labeled relevant documents, averaged per question |
| nDCG@10 | Binary relevance: sum(rel_i / log2(i+1)), divided by ideal DCG |
| Correctness | Fraction of answers without substantive factual errors; unjustified abstention fails |
| Completeness | Fraction of reference answer facts correctly covered, averaged per question |
| Citation support | Supported factual claims / factual claims |
| Abstention | Report correct abstention on unanswerable questions and unnecessary abstention separately |
| Efficiency | Median/p95 latency, API calls, tokens, USD/query, and search rounds, including retries |

Scoring rules:

- Deduplicate chunks by document ID, preserving first rank, before applying k.
- No relevant-document IDs means N/A for retrieval metrics. No answer facts means
  N/A for completeness; no factual claims means N/A for citation support. Report
  exclusion counts. Empty relevance labels alone do not establish unanswerability.
- Keep missing corpus documents in recall denominators and report corpus coverage.
- Eligible error rows score zero; count errors/timeouts instead of dropping them.
- Separate judge cost from serving cost, state prices, and mark unknown usage.
- Report category scores and counts, plus single/multiple-document and source
  slices. Source slices may overlap. Use paired differences and confidence
  intervals for later comparisons; avoid strong conclusions from small slices.

## Answer rubric

Score each applicable dimension separately from 0 to 2. A usable answer needs 2
on each; an average should not conceal factual errors.

| Dimension | 0 | 1 | 2 |
| --- | --- | --- | --- |
| Correctness | Wrong central claim | Main answer right, substantive detail wrong or unverifiable | All substantive claims correct |
| Completeness | Main information missing | Required details missing | Required facts, constraints, and conflicts covered |
| Citation support | Central claims unsupported | Some claims unsupported or ambiguously cited | Every factual claim supported by its citation |
| Abstention/limits | Unsupported answer or unjustified refusal | Uncertainty noted, key limitation missed | Answers when justified; explains missing/conflicting evidence otherwise |

Two reviewers independently score 20 development examples, including successes,
failures, and boundary cases. Record agreement by dimension, resolve differences,
and keep the adjudicated examples as an anchor set. Compare any LLM judge against
these labels; record its model, prompt, and reasons. Proposed acceptance: 80%
exact agreement per dimension before scaling up. This small-sample threshold is
for calibration, not proof of reliability. The Search Critic must not be its own
sole evaluator.

## Error analysis

Check schema validity, citation references, and permissions separately from answer
quality. Any invalid citation or access leak fails its check. Access tests use
synthetic permissions, separate from benchmark scores.

Inspect 20–30 failed development examples, or all if fewer. Record question/run
IDs, ranked evidence, selected context, answer, failed dimension, suspected cause,
proposed fix, and owner. Identify three to five recurring patterns when supported:
missing corpus evidence, retrieval miss, wrong chunk, generation error, or bad
abstention. Note uncertainty; a selected failure sample cannot estimate overall
failure rates. Later add critic confusion counts and useful/redundant search
rounds. Evaluate fixes on the same development set.

## Milestone 2

Deliver the Data Card, runnable harness, two baseline tables, validated rubric,
failure analysis, README commands/output, and TA check-in. The audit command only
prepares data. Hybrid reranking and the critic are not required for this checkpoint.
