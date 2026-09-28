"""Chapter 25: why a semantic cache is dangerous for Ask Halden.

A semantic cache reuses an answer when a new question's embedding
is close enough to a cached one. These pairs show how close
different questions can be, and how far apart the same one can be.
"""

import asyncio

import numpy as np

from halden.adapters.local_embedder import SentenceTransformerEmbedder
from halden.config.settings import Settings

# [start:pairs]
PAIRS = [  # (question A, question B, same answer?)
    (
        "What does E-17 mean on the PX-200?",
        "What does E-17 mean on the FM-310?",
        False,
    ),
    (
        "How much annual leave do employees in Germany get?",
        "How much annual leave do employees in the US get?",
        False,
    ),
    (
        "What is the accuracy of the PX-200?",
        "What is the accuracy of the HX-4410?",
        False,
    ),
    (
        "How many PTO days do US employees get?",
        "What's my yearly vacation allowance in the States?",
        True,
    ),
    (
        "The VPN says certificate expired. What now?",
        "How do I fix a cert error when connecting remotely?",
        True,
    ),
]
# [end:pairs]


async def main() -> None:
    s = Settings()
    emb = SentenceTransformerEmbedder(
        s.embedding_model, query_prefix=s.embedding_query_prefix
    )
    print(f"{'cosine':>7}  same answer?  question pair")
    for a, b, same in PAIRS:
        va, vb = await emb.embed_query(a), await emb.embed_query(b)
        cos = float(np.dot(va, vb))
        print(f"{cos:>7.3f}  {'yes' if same else 'NO':<12}  {a}")
        print(f"{'':>7}  {'':<12}  {b}")


if __name__ == "__main__":
    asyncio.run(main())
