# Evaluation protocol (v1)

Status: protocol, splits, baseline retrievers, generation adapter, and retrieval
metrics are implemented. The full evaluation runner and answer-quality scoring
remain later tickets; no benchmark scores are claimed.

## Question sets

Use the pinned revision in `src/config.py` and IDs in `evaluation/splits.json`.
The upstream split is named `test`; our partitions are project-specific, not
additional official benchmark splits. We are not training a model, so there is
no training set. Do not put this benchmark into model-training data.

Allocate approximately 60% development, 20% critic calibration, and 20% final.
Seed 5980. Group shared reference-document IDs (including transitive overlap) and
identical normalized question text before assignment. Greedily balance question
categories across splits; group integrity takes priority over exact percentages.
This reduces known leakage but cannot detect every shared project or near-duplicate.

The generated split contains 300 development, 100 calibration, and 100 final
questions. The development golden set takes up to six questions per category
(60 total); `evaluation/development_examples.json` shows one per category. Use it for
quick regression checks and the full development set for baseline tables.
Calibration is for critic thresholds and manually labeled sufficiency examples.
Keep final answers and outcomes out of iterative review; inspect final results
only after settings are frozen. Version new splits explicitly rather than silently
replacing the committed manifest. Documents are the shared search corpus, not
model-training examples; question labels are only available to the evaluator.

## Comparisons

Initial systems: BM25+-only and pretrained BGE-small dense-only retrieval, using
identical corpus, question IDs, generator/prompt, top-k, context budget, and access
policy. Hybrid/reranking and agentic search follow later. Record model revisions,
code commit, dependency versions, seed, corpus hash, and configuration per run.
Our scores are project comparisons, not official leaderboard reproductions.

## Metrics

| Metric | Definition and interpretation |
| --- | --- |
| Recall@5/10/20 | Per question, relevant unique documents in the first k unique document hits divided by all labeled relevant documents; then macro-average. |
| nDCG@10 | Binary relevance; DCG = sum(rel_i / log2(i+1)), divided by ideal DCG for min(10, number of relevant documents). Measures ranking quality. |
| Correctness | Fraction judged free of substantive factual errors against the reference and evidence; inappropriate abstention on answerable questions fails. |
| Completeness | Fraction of reference answer facts correctly covered, macro-averaged over questions with answer facts. Report empty-fact exclusions. |
| Citation support | Supported factual claims / factual claims, assessed against cited passages. No factual claims means N/A, not perfect support. |
| Abstention | Separately report abstention rate on labeled unanswerable questions and unnecessary abstention rate on answerable questions. |
| Efficiency | Median/p95 end-to-end latency, API calls, input/output tokens, estimated USD/query, and search rounds. Include retries; report errors/timeouts separately. |

Deduplicate chunks by parent document, preserving the first rank before applying k.
Questions with no relevant-document IDs are N/A for retrieval metrics; report their
count. Missing labels do not by themselves mean unanswerable: use the benchmark's
question category and manual review. Never remove missing corpus documents from
the recall denominator. Report corpus coverage alongside reduced-corpus scores.
Run errors get zero on eligible quality metrics, with counts visible; do not drop
them from averages. Cost reports identify pricing assumptions and separate judge
cost from serving cost. Unknown token/cost data stays missing, not zero.

Report overall and per-question-category scores with sample counts. Also slice by
one versus multiple reference documents and one versus multiple reference sources;
source slices can overlap. Small slices support examples, not strong significance
claims. Later comparisons should use paired per-question differences and confidence
intervals; do not claim gains from a few selected examples.

## Qualitative rubric

Score each dimension 0, 1, or 2 independently. A usable answer needs 2 on every
applicable dimension. Report dimensions separately; do not hide factual errors
behind an average score.

| Dimension | 0 | 1 | 2 |
| --- | --- | --- | --- |
| Correctness | Wrong central claim or contradiction | Main answer right, but a substantive detail is wrong or unverifiable | All substantive claims correct |
| Completeness | Misses the main requested information | Addresses the question but omits required facts/qualifications | Covers required facts, constraints, and material conflicts |
| Citation support | No valid support for central claims | Some factual claims unsupported or cited ambiguously | Every factual claim supported by identifiable cited evidence |
| Abstention/limits | Confident unsupported answer or unwarranted refusal | Signals uncertainty but misses the key evidence limitation | Answers when justified; otherwise clearly identifies missing/conflicting evidence |

Two reviewers independently score 20 development examples, including successes,
failures, and boundary cases. Record initial exact agreement per dimension, resolve
disagreements, and preserve the adjudicated anchor set. Compare any LLM judge to
both human ratings; log judge model/prompt versions and reasons. Proposed gate:
at least 80% exact agreement per dimension on this small anchor set before using
judge scores at scale; otherwise refine the rubric. This is a process threshold,
not statistical proof of reliability. Never use the online Search Critic as the
sole evaluator of its own outputs.

## Behavior checks and error analysis

Invariants: valid response schema, citations resolve, and no out-of-scope evidence.
Any observed access leak or invalid citation fails that check. Access tests use
explicit synthetic permission fixtures, separate from benchmark quality scores.
Schema/citation checks do not establish semantic faithfulness.

Inspect 20–30 failed development examples (or all if fewer). Record question ID,
baseline/run ID, retrieved IDs/ranks, selected context, answer, failed dimension,
evidence for the suspected cause, proposed fix, and owner. Group three to five
recurring patterns only when observed: missing corpus evidence, retrieval miss,
wrong chunk, generation error, or inappropriate abstention. Keep competing causes
when uncertain. This selected sample is not a population failure-rate estimate.
Later add critic false-sufficient/false-insufficient confusion counts and useful
versus redundant search rounds. Evaluate each fix on the same development set.

## Milestone 2 gate

Data Card, working baseline harness, two measured baseline tables, rubric and
human validation, initial failure analysis, README commands/output, and TA check-in.
The current audit command is not the evaluation harness. No quality threshold is
claimed before measurements; report both positive and negative comparisons.
A3 reranking and B3 critic are not prerequisites for this checkpoint.
