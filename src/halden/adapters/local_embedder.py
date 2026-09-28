"""Embedders that run in-process: a real model and a test double."""

import asyncio
import hashlib
import re
from collections.abc import Sequence
from typing import Any

import numpy as np

from halden.ports.embedder import Matrix, Vector


class SentenceTransformerEmbedder:
    """Wraps an open-weight model via sentence-transformers.

    Some models expect different prefixes for queries and documents
    (asymmetric retrieval); pass them in rather than hard-coding.
    """

    def __init__(
        self,
        model_name: str,
        *,
        query_prefix: str = "",
        doc_prefix: str = "",
        batch_size: int = 32,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self._model: Any = SentenceTransformer(model_name, device="cpu")
        self._name = model_name
        self._query_prefix = query_prefix
        self._doc_prefix = doc_prefix
        self._batch_size = batch_size

    @property
    def model_id(self) -> str:
        return self._name

    @property
    def dimensions(self) -> int:
        # Renamed in newer sentence-transformers; support both.
        get_dim = (
            getattr(
                self._model,
                "get_embedding_dimension",
                None,
            )
            or self._model.get_sentence_embedding_dimension
        )
        return int(get_dim())

    def _encode(self, texts: list[str]) -> Matrix:
        out = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(out, dtype=np.float32)

    async def embed_documents(self, texts: Sequence[str]) -> Matrix:
        batch = [self._doc_prefix + t for t in texts]
        # Model inference is CPU-bound: keep it off the event loop.
        return await asyncio.to_thread(self._encode, batch)

    async def embed_query(self, text: str) -> Vector:
        mat = await asyncio.to_thread(
            self._encode, [self._query_prefix + text]
        )
        vec: Vector = mat[0]
        return vec


TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


# [start:hashing]
class HashingEmbedder:
    """Deterministic, dependency-free stand-in for tests.

    It hashes words into buckets, so it captures *lexical* overlap
    only. That is exactly what makes it useful in tests: results are
    predictable and require no model download.
    """

    def __init__(self, dimensions: int = 2048) -> None:
        self._dims = dimensions

    @property
    def model_id(self) -> str:
        return f"hashing-{self._dims}"

    @property
    def dimensions(self) -> int:
        return self._dims

    def _one(self, text: str) -> Vector:
        vec = np.zeros(self._dims, dtype=np.float32)
        for tok in TOKEN.findall(text.lower()):
            digest = hashlib.blake2b(tok.encode(), digest_size=8)
            h = int.from_bytes(digest.digest(), "little")
            sign = 1.0 if (h >> 63) & 1 else -1.0
            vec[h % self._dims] += sign
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm else vec

    # [end:hashing]
    async def embed_documents(self, texts: Sequence[str]) -> Matrix:
        return np.stack([self._one(t) for t in texts])

    async def embed_query(self, text: str) -> Vector:
        return self._one(text)
