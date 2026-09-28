"""Chapter 11: does a cross-encoder improve first-stage results?"""

import asyncio
import time

import numpy as np

from examples.ch10_compare_retrievers import DSN, LABELED, PREFIX
from halden.adapters.cross_encoder import CrossEncoderReranker
from halden.adapters.local_embedder import SentenceTransformerEmbedder
from halden.retrieval.pg_retriever import PgRetriever, RetrievedChunk
from halden.retrieval.rerank import rerank

RERANKER = "cross-encoder/ms-marco-MiniLM-L6-v2"


def rank_of(hits: list[RetrievedChunk], answer: str) -> int | None:
    return next(
        (i for i, c in enumerate(hits, 1) if answer in c.text), None
    )


async def main() -> None:
    emb = SentenceTransformerEmbedder(
        "BAAI/bge-small-en-v1.5", query_prefix=PREFIX
    )
    rr = CrossEncoderReranker(RERANKER)
    r = await PgRetriever.connect(DSN, emb)
    g = ["everyone"]
    rows: dict[str, list[int | None]] = {}
    for label, first in (("dense", r.dense), ("hybrid", r.hybrid)):
        base, reranked = [], []
        for q, answer in LABELED:
            cands = await first(q, g, 20)
            base.append(rank_of(cands, answer))
            top = await rerank(rr, q, cands, 20)
            reranked.append(rank_of(top, answer))
        rows[label] = base
        rows[f"{label} + rerank"] = reranked

    n = len(LABELED)
    print(f"{'pipeline':<16}{'hit@1':>7}{'hit@3':>7}{'MRR':>7}")
    for name, rs in rows.items():
        h1 = sum(x == 1 for x in rs)
        h3 = sum(x is not None and x <= 3 for x in rs)
        mrr = sum(1 / x for x in rs if x) / n
        print(f"{name:<16}{h1:>4}/{n}{h3:>4}/{n}{mrr:>7.2f}")

    print("\nreranking latency on CPU (median of 5):")
    q = LABELED[0][0]
    pool = await r.dense(q, g, 60)
    for size in (10, 20, 40, 60):
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            await rr.score(q, [c.text for c in pool[:size]])
            times.append((time.perf_counter() - t0) * 1000)
        print(f"  {size:>3} candidates: {np.median(times):6.0f} ms")
    await r.aclose()


if __name__ == "__main__":
    asyncio.run(main())
