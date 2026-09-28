"""Second-stage ranking of retrieved candidates."""

from collections.abc import Sequence

from halden.ports.reranker import Reranker
from halden.retrieval.pg_retriever import RetrievedChunk


# [start:rerank]
async def rerank(
    reranker: Reranker,
    query: str,
    candidates: Sequence[RetrievedChunk],
    top_n: int,
) -> list[RetrievedChunk]:
    """Re-order candidates by reranker score and keep ``top_n``.

    The reranker sees the chunk's heading path as well as its
    text, for the same reason chunks are embedded with a context
    header (Chapter 8).
    """
    if not candidates:
        return []
    texts = [
        " > ".join([c.title, *c.heading_path]) + "\n" + c.text
        for c in candidates
    ]
    scores = await reranker.score(query, texts)
    order = sorted(
        range(len(candidates)), key=lambda i: scores[i], reverse=True
    )
    return [
        candidates[i].model_copy(update={"score": scores[i]})
        for i in order[:top_n]
    ]


# [end:rerank]
