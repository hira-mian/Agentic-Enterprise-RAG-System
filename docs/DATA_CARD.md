# Data card: EnterpriseRAG-Bench

## What it is

[EnterpriseRAG-Bench](https://github.com/onyx-dot-app/EnterpriseRAG-Bench), made by
Onyx, is a synthetic set of internal documents for a fictional AI company,
"Redwood Inference", plus questions about them. It is designed to test search
and question answering over company data.

- **License:** MIT (declared by the dataset authors). Keep attribution when
  sharing the data.
- **Use:** we use it only for search and evaluation. The authors ask that it never
  be used to train models, and we do not.
- **Version:** we build it from the authors' GitHub repository at commit
  `d36685e273713975ee20299bbf1ab64165575b3c`, so everyone gets identical data.
  The same data is also published on
  [Hugging Face](https://huggingface.co/datasets/onyx-dot-app/EnterpriseRAG-Bench).

Build it with:

```sh
python -m src.data.corpus --clone
```

This writes all documents to `.cache/corpus/documents.parquet` and records counts
and a content fingerprint in `evaluation/corpus_manifest.json`.

## Documents

**511,958 documents** from 9 sources:

| Source | What it contains | Documents |
| --- | --- | ---: |
| Slack | Team chat threads | 285,605 |
| Gmail | Email threads | 121,390 |
| Linear | Project tickets | 35,308 |
| Google Drive | Shared documents | 25,108 |
| HubSpot | Sales CRM records | 15,016 |
| Fireflies | Meeting transcripts | 10,173 |
| GitHub | Pull requests | 8,052 |
| Jira | Support tickets | 6,118 |
| Confluence | Wiki pages and runbooks | 5,188 |
| **Total** | | **511,958** |

Each document has an ID, a source, a title, and text. An average document is
about 770 words long.

## Questions

**500 questions**, each labeled with a reference answer and the documents that
contain it.

| Type | Count | What it tests |
| --- | ---: | --- |
| Basic | 175 | One document holds the answer |
| Semantic | 125 | Like basic, but reworded to avoid the document's key words |
| Intra-document reasoning | 40 | Combining distant parts of one long document |
| Project related | 40 | Combining several documents about one project |
| Constrained | 30 | Several documents look relevant; details rule out all but one |
| Conflicting info | 20 | Documents contradict each other |
| Completeness | 20 | Every relevant document (up to 10) is needed |
| Miscellaneous | 20 | Informal or loosely organized documents |
| Info not found | 20 | The answer is not in the data |
| High level | 10 | The answer is spread across the company, not in one document |

- 30 questions (info not found and high level) have no reference documents, so
  they cannot be scored on search.
- 93 questions have more than one reference document.
- All 722 reference documents are in the dataset.

## How we split the questions

| Split | Questions | Use |
| --- | ---: | --- |
| Development | 300 | Building and comparing methods |
| Calibration | 100 | Tuning the Search Critic later |
| Final | 100 | Final evaluation only, after all settings are frozen |

Each split has a similar mix of question types. Questions that share a reference
document, or have identical wording, stay in the same split, which limits overlap
between development and final (related questions about the same project can still
land in different splits). We do not train a model, so there is no training split. The split is
fixed in `evaluation/splits.json`; see the [evaluation plan](EVALUATION.md) for
the exact rules.

## Data quality notes

- **Duplicate IDs:** the repository has 511,962 document files, but four IDs are
  each used by two different files. We keep the file that the authors' own index
  (`uuid_index.json`) points to. One of these four is a reference document.
- **No structured metadata:** dates, authors' roles, project membership, and
  access permissions are not provided as fields. Any values we extract from the
  text must be checked. Permissions used in our tests are made up for testing.

## Limitations

- The data is synthetic, so questions and documents may not match how real
  employees write.
- Sources are imbalanced (over half is Slack), and question types are imbalanced
  (basic and semantic are 60% of questions).
- The authors deliberately added noise: near-duplicate documents, outdated
  information, and misfiled documents. Some reference labels may be incomplete.
