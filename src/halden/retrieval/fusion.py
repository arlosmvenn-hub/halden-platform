"""Combine ranked lists from different retrievers."""

from collections import defaultdict
from collections.abc import Sequence


# [start:rrf]
def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse ranked ID lists by rank position, ignoring raw scores.

    Each list contributes weight / (k + rank) for every ID it
    contains (rank starts at 1). Items ranked well by several
    retrievers rise; the constant k damps the influence of any
    single list's top positions.
    """
    weights = weights or [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("one weight per ranking")
    fused: dict[str, float] = defaultdict(float)
    for ranking, w in zip(rankings, weights, strict=True):
        for rank, item in enumerate(ranking, start=1):
            fused[item] += w / (k + rank)
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))


# [end:rrf]
