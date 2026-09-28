"""Exact nearest-neighbor search: the baseline every index is
measured against."""

import numpy as np

from halden.ports.embedder import Matrix, Vector


def top_k_cosine(
    query: Vector, docs: Matrix, k: int
) -> list[tuple[int, float]]:
    """Return (row index, cosine similarity) for the k best rows.

    Assumes rows and query are L2-normalized, so the dot product
    *is* the cosine similarity. Cost: O(n * d) per query.
    """
    if docs.shape[0] == 0:
        return []
    scores = docs @ query
    k = min(k, scores.shape[0])
    # argpartition finds the top k in O(n); we then sort only k.
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [(int(i), float(scores[i])) for i in idx]
