"""Retrieval and answer metrics. Pure functions, no I/O."""

import math
from collections.abc import Sequence

from halden.eval.dataset import normalize


# [start:covers]
def coverage(
    texts: Sequence[str], evidence: Sequence[str]
) -> list[frozenset[int]]:
    """For each retrieved chunk, in rank order: which evidence
    items (by index) it contains. An empty set means irrelevant."""
    ev = [normalize(e) for e in evidence]
    return [
        frozenset(i for i, e in enumerate(ev) if e in normalize(t))
        for t in texts
    ]


# [end:covers]


# [start:metrics]
def hit_at_k(covers: Sequence[frozenset[int]], k: int) -> float:
    """1.0 if any of the top k chunks is relevant."""
    return float(any(covers[:k]))


def precision_at_k(covers: Sequence[frozenset[int]], k: int) -> float:
    """Share of the top k slots holding a relevant chunk. Empty
    slots count as misses, so returning fewer results is not a
    way to look precise."""
    return sum(1 for c in covers[:k] if c) / k


def recall_at_k(
    covers: Sequence[frozenset[int]], n_evidence: int, k: int
) -> float:
    """Share of the evidence items found anywhere in the top k."""
    found = frozenset().union(*covers[:k]) if covers else frozenset()
    return len(found) / n_evidence


def reciprocal_rank(covers: Sequence[frozenset[int]]) -> float:
    """1 / rank of the first relevant chunk; 0 if there is none.
    Averaged over questions, this is MRR."""
    for rank, c in enumerate(covers, start=1):
        if c:
            return 1 / rank
    return 0.0


def ndcg_at_k(
    covers: Sequence[frozenset[int]], n_evidence: int, k: int
) -> float:
    """Normalized discounted cumulative gain with *novelty* gain:
    a chunk earns 1 only if it adds evidence not already seen.
    A second copy of the same fact is not a second relevant
    result, and the ideal ranking finds one new item per slot."""
    seen: set[int] = set()
    dcg = 0.0
    for i, c in enumerate(covers[:k]):
        if c - seen:
            dcg += 1 / math.log2(i + 2)
            seen |= c
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, n_evidence)))
    return dcg / ideal


# [end:metrics]


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")
