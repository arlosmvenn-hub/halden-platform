"""Chapter 25: retrieval latency percentiles, with and without the
reranker, over the golden questions (five passes each)."""

import asyncio
import time
from pathlib import Path

import numpy as np

from halden.adapters.cross_encoder import CrossEncoderReranker
from halden.adapters.fake_llm import ScriptedLLM
from halden.adapters.local_embedder import SentenceTransformerEmbedder
from halden.config.settings import Settings
from halden.eval.dataset import load_cases
from halden.security.identity import Identity
from halden.services.ask import AskService
from halden.store.db import create_pool

GOLDEN = Path(__file__).parent.parent / "eval/datasets/golden.jsonl"
RERANKER = "cross-encoder/ms-marco-MiniLM-L6-v2"


async def main() -> None:
    s = Settings()
    pool = create_pool(s)
    await pool.open()
    emb = SentenceTransformerEmbedder(
        s.embedding_model, query_prefix=s.embedding_query_prefix
    )
    questions = [c.question for c in load_cases(GOLDEN)] * 5
    who = Identity(user_id="bench", groups=["everyone"])
    print(f"{len(questions)} retrievals per configuration")
    print(f"{'config':<16}{'p50':>8}{'p95':>8}{'p99':>8}{'max':>8}  ms")
    for name, rr in [
        ("hybrid", None),
        ("hybrid+rerank", CrossEncoderReranker(RERANKER)),
    ]:
        svc = AskService(pool, emb, ScriptedLLM([]), s, rr)
        await svc.retrieve("warm up", who)
        ms = []
        for q in questions:
            t0 = time.perf_counter()
            await svc.retrieve(q, who)
            ms.append((time.perf_counter() - t0) * 1000)
        p = np.percentile(ms, [50, 95, 99, 100])
        print(f"{name:<16}" + "".join(f"{v:>8.0f}" for v in p))
    await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
