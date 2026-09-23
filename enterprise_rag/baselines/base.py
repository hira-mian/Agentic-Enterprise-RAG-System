from __future__ import annotations

from abc import ABC, abstractmethod


class System(ABC):
    """A QA system under evaluation.

    `retrieve` returns the ranked candidate chunks (at least `top_k` of them, more
    when the harness asks for deeper retrieval diagnostics); `answer` produces the
    final answer from the query and the top-`top_k` chunks the system would
    actually use.
    """

    name: str
    top_k: int = 0

    def retrieve(self, query: str, k: int) -> list:
        return []

    @abstractmethod
    def answer(self, query: str, nodes: list, question_type: str) -> str | None:
        ...

    def config(self) -> dict:
        return {"system": self.name, "top_k": self.top_k}
