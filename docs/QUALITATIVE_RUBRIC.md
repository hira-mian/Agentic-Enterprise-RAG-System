# Qualitative evaluation rubric

The automatic metrics check whether the right documents were retrieved and whether the gold string appears in the answer. They do not check whether the answer is *trustworthy*. For that, reviewers score a stratified sample of predictions (10 per question type = 40 per system by default) using the rubric below.

```bash
python scripts/export_review_sample.py results/<run>/predictions.jsonl --per-type 10
# -> results/<run>/review_sample.csv with empty rubric columns to fill in
```

Give each criterion a score of **0 / 1 / 2**. A response is **"good"** if it scores 2 on Correctness and Grounding and at least 1 on everything else.

## Criteria

### 1. Correctness: is the final answer right?
| Score | Description |
|---|---|
| 2 | States the gold answer (or a clearly equivalent one) without contradicting itself |
| 1 | Partly right: the right entity with a wrong detail, or the right answer among several hedged alternatives |
| 0 | Wrong, missing, or contradicts itself ("Yes… therefore no") |

### 2. Grounding / faithfulness: is every claim supported by the retrieved context?
| Score | Description |
|---|---|
| 2 | Every factual claim can be traced to the retrieved chunks |
| 1 | Mostly grounded, but with small unsupported additions (for example background knowledge that happens to be true) |
| 0 | Key claims are not in the context (hallucinated, or answered from the LLM's memory against or without the context) |

### 3. Multi-hop coverage: does the answer use evidence from every required source?
| Score | Description |
|---|---|
| 2 | Draws on all the sources or articles the question refers to ("as reported by X and Y") |
| 1 | Uses only some of the required sources, but the answer is still justified |
| 0 | Relies on a single source for a question that needs several, or confuses which source said what |

### 4. Abstention / calibration: does the system know when not to answer?
| Score | Description |
|---|---|
| 2 | Null query: clearly says the information is not available. Answerable query: answers directly |
| 1 | Hedges without need, but still gives the answer; or declines only partly on a null query |
| 0 | Makes up an answer to a null query, or refuses an answerable query whose evidence was retrieved |

### 5. Clarity and usefulness: would an enterprise user be satisfied?
| Score | Description |
|---|---|
| 2 | Direct answer first, brief justification, and names the sources or dates used |
| 1 | Correct but wordy, poorly organised, or missing source attribution |
| 0 | Hard to find the answer, or padded with irrelevant text |

## Failure categories (pick one per incorrect or low-scoring row)

| Code | Category | Typical cause |
|---|---|---|
| `R-miss` | **Retrieval miss:** a required article is not in the top-k | Too few chunks (top-k), weak lexical/semantic match, publisher or date constraint ignored |
| `R-partial` | **Partial evidence:** some but not all hops were retrieved | k smaller than the number of hops; one article crowds out the others |
| `G-reason` | **Reasoning error:** the evidence is present but the comparison or inference is wrong | Multi-step reasoning, temporal ordering |
| `G-halluc` | **Hallucination:** the answer is not supported by the context | LLM prior or memorisation |
| `G-overabstain` | **Over-abstention:** declines although the evidence was retrieved | Conservative prompt; evidence spread across chunks |
| `G-underabstain` | **Under-abstention:** answers a null query | No abstention instruction in the default prompt |
| `F-format` | **Format or scoring mismatch:** right in substance, scored wrong | Paraphrased gold label ("Consistent" vs "Yes"), verbose answer |
| `D-label` | **Dataset label issue:** the gold answer looks wrong or ambiguous | LLM-generated labels (Data Card, limitation 1) |

Each baseline's category counts go in the error-analysis section of [BASELINE_RESULTS.md](BASELINE_RESULTS.md) once the sample is reviewed. They guide what the agentic system should fix first.

## Example judgements

| Query (abridged) | Gold | Prediction | Scores (C/G/M/A/Cl) | Category |
|---|---|---|---|---|
| Who is in a fraud trial, as reported by The Verge and TechCrunch? | Sam Bankman-Fried | "Sam Bankman-Fried, the FTX founder, per both The Verge and TechCrunch." | 2/2/2/2/2 | — |
| Does TechCrunch report X while The Verge reports Y? | Yes | "The context only discusses TechCrunch's article; it does not mention The Verge." | 0/2/0/0/1 | `R-partial` |
| Which company …, as reported by Fortune and The Age? (not in corpus) | Insufficient information. | "Google." | 0/0/0/0/1 | `G-underabstain` |
