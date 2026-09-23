"""Answer-prior baselines that never look at the corpus.

MultiHop-RAG answers are heavily skewed (most comparison/temporal questions are
yes/no), so these give the floor any retrieval system has to beat.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from ..data import Query
from ..metrics import normalize
from .base import System


def _most_common(answers: list[str]) -> str:
    # Count on normalized answers so "No" and "no" are one class, but return a
    # surface form that actually appears in the data.
    counts = Counter(normalize(a) for a in answers)
    top = counts.most_common(1)[0][0]
    return next(a for a in answers if normalize(a) == top)


class MajorityBaseline(System):
    """`global`: always predict the most frequent dev answer.
    `per_type`: predict the most frequent dev answer for the query's question type.
    (The per-type variant reads the gold question_type label, so it is an upper
    bound for any prior-only heuristic rather than a deployable system.)
    """

    def __init__(self, fit_queries: list[Query], mode: str = "global"):
        self.mode = mode
        self.name = f"majority_{mode}"
        self.global_answer = _most_common([q.answer for q in fit_queries])
        by_type = defaultdict(list)
        for q in fit_queries:
            by_type[q.question_type].append(q.answer)
        self.type_answer = {t: _most_common(a) for t, a in by_type.items()}

    def answer(self, query: str, nodes: list, question_type: str) -> str:
        if self.mode == "per_type":
            return self.type_answer.get(question_type, self.global_answer)
        return self.global_answer

    def config(self) -> dict:
        return {
            **super().config(),
            "fit_split": "dev",
            "global_answer": self.global_answer,
            "type_answer": self.type_answer if self.mode == "per_type" else None,
        }
