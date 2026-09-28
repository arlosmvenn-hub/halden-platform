"""The embedding port: text in, unit-length vectors out."""

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

Vector = NDArray[np.float32]
Matrix = NDArray[np.float32]


class Embedder(Protocol):
    @property
    def model_id(self) -> str:
        """Stored beside every vector; never mix models in an index."""
        ...

    @property
    def dimensions(self) -> int: ...

    async def embed_documents(self, texts: Sequence[str]) -> Matrix:
        """Shape (len(texts), dimensions), rows L2-normalized."""
        ...

    async def embed_query(self, text: str) -> Vector:
        """Shape (dimensions,), L2-normalized."""
        ...
