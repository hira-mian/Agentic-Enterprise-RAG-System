"""Retrieval and answer metrics.

Retrieval is scored at the *document* level: a retrieved chunk counts as a hit
for the article (URL) it came from, and repeated chunks from one article count
once. Answer scoring follows the MultiHop-RAG convention that gold answers are
short (an entity, "Yes"/"No", ...) while systems may answer verbosely.
"""

from __future__ import annotations

import re
import string
from collections import Counter

# Phrases treated as the system declining to answer. Null queries (whose gold
# answer is "Insufficient information.") are correct only if the system abstains.
ABSTAIN_PATTERNS = [
    r"insufficient information",
    r"not enough information",
    r"(does not|doesn't|do not|don't) (provide|contain|include|mention|specify|say|state)",
    r"(is|are) not (mentioned|provided|specified|stated|available)",
    r"no (information|mention|details?)",
    r"cannot (be )?(determine|determined|answer|answered|say)",
    r"can't (be )?(determine|determined|answer|say)",
    r"unable to (determine|answer)",
    r"not possible to (determine|answer)",
    r"i (do not|don't) know",
    r"^empty response$",
]
_ABSTAIN_RE = re.compile("|".join(ABSTAIN_PATTERNS))

_YES_NO = {"yes", "no"}


def normalize(text: str) -> str:
    """Lowercase, drop punctuation and articles, collapse whitespace (SQuAD-style)."""
    text = text.lower()
    text = "".join(ch if ch not in string.punctuation else " " for ch in text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def is_abstention(prediction: str) -> bool:
    return bool(_ABSTAIN_RE.search(prediction.lower().strip()))


def _contains_span(haystack: list[str], needle: list[str]) -> bool:
    n = len(needle)
    return n > 0 and any(haystack[i : i + n] == needle for i in range(len(haystack) - n + 1))


def answer_correct(prediction: str, gold: str, question_type: str) -> bool:
    """Primary answer metric ("accuracy").

    * null queries: correct iff the prediction abstains.
    * yes/no gold answers: the first "yes"/"no" token in the prediction must match.
    * otherwise: the normalized gold answer must appear as a contiguous token span
      in the normalized prediction (lenient to verbose answers).
    """
    if question_type == "null_query":
        return is_abstention(prediction)
    gold_n, pred_tokens = normalize(gold), normalize(prediction).split()
    if gold_n in _YES_NO:
        first = next((t for t in pred_tokens if t in _YES_NO), None)
        return first == gold_n
    return _contains_span(pred_tokens, gold_n.split())


def exact_match(prediction: str, gold: str) -> bool:
    return normalize(prediction) == normalize(gold)


def token_f1(prediction: str, gold: str) -> float:
    pred, ref = normalize(prediction).split(), normalize(gold).split()
    if not pred or not ref:
        return float(pred == ref)
    common = sum((Counter(pred) & Counter(ref)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(pred), common / len(ref)
    return 2 * precision * recall / (precision + recall)


def dedupe(items: list[str]) -> list[str]:
    seen, out = set(), []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def retrieval_scores(retrieved_chunk_urls: list[str], gold_urls: list[str], k: int) -> dict:
    """Document-level metrics using the top-k retrieved *chunks*.

    hit@k      at least one gold article was retrieved
    recall@k   fraction of gold articles retrieved
    all@k      every gold article was retrieved (complete evidence for a multi-hop query)
    mrr@k      reciprocal rank (by distinct article) of the first gold article
    """
    docs = dedupe(retrieved_chunk_urls[:k])
    gold = set(gold_urls)
    found = [d for d in docs if d in gold]
    first_rank = next((i + 1 for i, d in enumerate(docs) if d in gold), None)
    return {
        f"hit@{k}": float(bool(found)),
        f"recall@{k}": len(found) / len(gold),
        f"all@{k}": float(len(found) == len(gold)),
        f"mrr@{k}": 1.0 / first_rank if first_rank else 0.0,
    }
