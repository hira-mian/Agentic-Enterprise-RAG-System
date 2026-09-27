# Data Card: EnterpriseRAG-Bench

## Source and license

[EnterpriseRAG-Bench](https://huggingface.co/datasets/onyx-dot-app/EnterpriseRAG-Bench)
is Onyx's synthetic enterprise dataset about a fictional company, Redwood
Inference. The upstream card reports over 500,000 documents and 500 questions.
We access its `documents` and `questions` subsets through Hugging Face `datasets`.

Pinned revision: `69916e31c68aa5963c00248fd7f0bc12d04fd235`.
The card declares MIT licensing; the inspected dataset tree has no separate
LICENSE file. Preserve attribution and applicable notices when distributing data.
The card asks that benchmark data stay out of training corpora. Our use is
retrieval and evaluation. Public reads worked without a token; network access
and Hub rate limits apply.

## Fields

| Data | Upstream fields |
| --- | --- |
| Documents | `doc_id`, `source_type`, `title`, `content` |
| Questions | `question_id`, `question_type`, `source_types`, `question` |
| Evaluation labels | `expected_doc_ids`, `gold_answer`, `answer_facts` |

Chunk IDs and offsets are generated locally. Timestamps, employee roles, project
memberships, permissions, and Jira/CRM record fields are not structured columns
in this schema. Extracted values need validation; missing permissions do not mean
public access. Test permissions are explicitly synthetic.

## Audit and splits

Run `python -m src.data.audit` to inspect the first 256 documents and all question
rows. Raw data goes to ignored `.cache/audit/`; summaries and split IDs go to
`evaluation/`.

Observed at the pinned revision:

- All 256 sampled documents are Confluence records, with no empty core fields or
  duplicate IDs. The prefix is not representative of the corpus.
- All 500 question IDs are unique. Thirty questions have no reference document or
  source IDs, including high-level and information-not-found questions.
- The sample contains 10 of 442 development reference-document links. Only two
  questions with relevance labels have all their reference documents present.
- Question splits: 300 development, 100 calibration, 100 final; 60 development
  questions form the regression set. See [evaluation protocol](EVALUATION.md)
  for grouping and split rules. We do not train a model, so no training set exists.

## Full corpus

Baselines use the full corpus from the upstream generator repository at commit
`d36685e273713975ee20299bbf1ab64165575b3c`, built by `python -m src.data.corpus`.
See `evaluation/corpus_manifest.json` for counts and hashes.

| Source | Documents |
| --- | ---: |
| Slack | 285,605 |
| Gmail | 121,390 |
| Linear | 35,308 |
| Google Drive | 25,108 |
| HubSpot | 15,016 |
| Fireflies | 10,173 |
| GitHub | 8,052 |
| Jira | 6,118 |
| Confluence | 5,188 |
| **Total** | **511,958** |

The repository has 511,962 document files. Four IDs are each reused by two
different files; we keep the file named in upstream's `uuid_index.json`. One of
these IDs is a reference document. All 722 reference documents are present.
Documents average 767 word tokens (title and content).

## Limitations

Synthetic data may not reflect real employee questions. Sources and question
categories are imbalanced; documents may be duplicated or conflicting; relevance
labels may be incomplete. Freshness and access metadata remain unverified.

The sample is for inspection. Build a larger corpus before baseline evaluation,
or label results as reduced-corpus scores and report reference coverage. Keep
missing reference documents in recall denominators. Sample findings do not
establish corpus-wide missingness or quality.
