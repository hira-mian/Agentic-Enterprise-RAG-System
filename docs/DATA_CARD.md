# Data Card: EnterpriseRAG-Bench

## Source and access

We use Onyx's synthetic Redwood Inference enterprise benchmark, accessed through
Hugging Face `datasets`. The upstream card lists 500 questions and over 500,000
documents across enterprise-like sources. These are simulated records, not our
team's private company data. See the [upstream card](https://huggingface.co/datasets/onyx-dot-app/EnterpriseRAG-Bench)
and [release repository](https://github.com/onyx-dot-app/EnterpriseRAG-Bench).

Pinned revision: `69916e31c68aa5963c00248fd7f0bc12d04fd235`.
The card metadata declares `mit`; the inspected dataset tree contains no separate
LICENSE file. Preserve attribution and applicable license notices when distributing
data. The card asks that benchmark data not enter training corpora. We use it for
retrieval/evaluation only. Public reads succeeded without a token; network access
and Hub rate limits still apply. No hosted LLM key is required for the audit.

## Fields and provenance

| Field group | Origin and constraints |
| --- | --- |
| Document ID, source type, title, content | Upstream document columns; preserve IDs and original text |
| Question ID/type, source types, question | Upstream question columns |
| Expected document IDs, reference answer, answer facts | Upstream evaluation labels; never runtime agent inputs |
| Chunk IDs and offsets | Derived locally during future preprocessing |
| Timestamp, employee role, project membership, ACL, Jira/CRM fields | Not structured columns in the inspected schema; any extraction needs validation and provenance |

Text may mention dates or organizational details; we have not verified a complete
employee directory, permission policy, or structured-record schema. Missing ACLs
must not be interpreted as public access. Invented authorization fixtures must
remain labeled synthetic and separate from benchmark measurements.

## Sampling, splits, and limits

`python -m src.data.audit` streams all question rows for split assignment and
profiles the first 256 documents at the pinned revision. It saves raw data under
ignored `.cache/audit/` and versioned summaries in `evaluation/`. This prefix is a
repeatable inspection fixture, not a representative corpus or a ready-to-score
benchmark. Per-development-question missing reference IDs are explicitly reported.
Do not change relevance labels to hide missing evidence.

Question partitions, grouping, rationale, and scoring rules are in
[EVALUATION.md](EVALUATION.md); actual IDs and slice counts are in
`evaluation/splits.json`. No parameter-training partition is created. Full-text
labels remain in the ignored cache; development examples can be exported without
opening final answers. The broader corpus must be built before credible baseline
scores, or reduced-corpus results must be clearly labeled.

Known limits: synthetic-to-real domain mismatch, source/category imbalance,
possible incomplete relevance labels, near-duplicate documents, conflicting text,
and unknown freshness/access metadata. Our sample cannot estimate corpus-wide
missingness or establish that every reference document exists in the full corpus.
Use the audit's observed counts rather than treating reported upstream totals as
an independent full-corpus verification.

## Observed audit

- 256 document rows, all Confluence; no empty core fields or duplicate IDs.
- 500 question rows; no duplicate question IDs. Thirty rows have empty reference
  document/source lists. These include high-level and information-not-found cases;
  empty relevance lists do not automatically mean an unanswerable question.
- Development sample coverage: 10 of 442 reference-document links present;
  only two questions with nonempty relevance labels have all references present.
- Splits: 300 development / 100 calibration / 100 final, with 60 development
  golden examples. Counts and IDs are in the committed evaluation artifacts.

These findings describe the bounded audit, not full-corpus quality.
