"""The reranking port: score (query, text) pairs precisely."""

from collections.abc import Sequence
from typing import Protocol


class Reranker(Protocol):
    @property
    def model_id(self) -> str: ...

    async def score(
        self, query: str, texts: Sequence[str]
    ) -> list[float]:
        """One relevance score per text, same order. Higher is
        more relevant. Scores are comparable only within a call."""
        ...
