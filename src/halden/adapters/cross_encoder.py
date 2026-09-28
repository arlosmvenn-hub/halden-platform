"""A local cross-encoder reranker via sentence-transformers."""

import asyncio
from collections.abc import Sequence
from typing import Any


# [start:adapter]
class CrossEncoderReranker:
    """Reads query and text *together*, so every word of one can
    attend to every word of the other. Far more precise than
    comparing two independent embeddings, and far slower: one
    model pass per candidate, at query time."""

    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        from sentence_transformers import CrossEncoder

        self._model: Any = CrossEncoder(model_name, device="cpu")
        self._name = model_name
        self._batch_size = batch_size

    @property
    def model_id(self) -> str:
        return self._name

    def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores = self._model.predict(pairs, batch_size=self._batch_size)
        return [float(s) for s in scores]

    async def score(
        self, query: str, texts: Sequence[str]
    ) -> list[float]:
        pairs = [(query, t) for t in texts]
        return await asyncio.to_thread(self._predict, pairs)


# [end:adapter]
