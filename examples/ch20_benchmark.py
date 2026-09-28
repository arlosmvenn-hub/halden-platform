"""Chapter 20: pipeline configurations compared end to end.

Measures retrieval through the real AskService (row-level security,
context building and all) on two question sets:
  design:   Chapter 10's 16 questions (used to design hybrid search)
  held-out: Chapter 13's 12 vague questions (never used for design)
"""

import asyncio
import time

import numpy as np

from examples.ch10_compare_retrievers import LABELED
from examples.ch13_multi_query import VAGUE
from halden.adapters.cross_encoder import CrossEncoderReranker
from halden.adapters.fake_llm import ScriptedLLM
from halden.adapters.local_embedder import SentenceTransformerEmbedder
from halden.config.settings import Settings
from halden.security.identity import Identity
from halden.services.ask import AskService
from halden.store.db import create_pool

READER = Identity(
    user_id="bench", groups=["everyone", "support", "legal"]
)
# [start:configs]
CONFIGS = {
    "dense": dict(retrieval_mode="dense"),
    "hybrid": dict(retrieval_mode="hybrid"),
    "hybrid + rerank": dict(
        retrieval_mode="hybrid",
        reranker_model="cross-encoder/ms-marco-MiniLM-L6-v2",
    ),
}
# [end:configs]


async def run(name: str, overrides: dict[str, str]) -> None:
    settings = Settings(**overrides)  # type: ignore[arg-type]
    pool = create_pool(settings)
    await pool.open()
    emb = SentenceTransformerEmbedder(
        settings.embedding_model,
        query_prefix=settings.embedding_query_prefix,
    )
    rr = (
        CrossEncoderReranker(settings.reranker_model)
        if settings.reranker_model
        else None
    )
    svc = AskService(pool, emb, ScriptedLLM([]), settings, rr)
    await svc.retrieve("warm up", READER)
    cells = []
    for questions in (LABELED, VAGUE):
        top1 = in_ctx = 0
        tokens, ms = [], []
        for q, answer in questions:
            t0 = time.perf_counter()
            hits = await svc.retrieve(q, READER)
            ms.append((time.perf_counter() - t0) * 1000)
            ctx = svc.context(hits)
            top1 += bool(hits) and answer in hits[0].text
            in_ctx += any(answer in s.text for s in ctx.sources)
            tokens.append(ctx.tokens)
        n = len(questions)
        cells.append(
            f"{top1:>3}/{n} {in_ctx:>3}/{n} "
            f"{np.mean(tokens):>5.0f} {np.median(ms):>6.0f}"
        )
    print(f"{name:<16}" + "   ".join(cells))
    await pool.close()


async def main() -> None:
    head = "top1  in-ctx  tokens  ms"
    print(f"{'':16}{'design set (16)':<29}held-out set (12)")
    print(f"{'config':<16}{head}      {head}")
    for name, overrides in CONFIGS.items():
        await run(name, overrides)


if __name__ == "__main__":
    asyncio.run(main())
