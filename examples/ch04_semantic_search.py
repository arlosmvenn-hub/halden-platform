"""Chapter 4: brute-force semantic search over Halden snippets.

python examples/ch04_semantic_search.py            # real model
python examples/ch04_semantic_search.py --hashing  # no download
"""

import asyncio
import json
import sys
from pathlib import Path

from halden.adapters.local_embedder import (
    HashingEmbedder,
    SentenceTransformerEmbedder,
)
from halden.ports.embedder import Embedder
from halden.retrieval.brute_force import top_k_cosine

DATA = Path(__file__).parent.parent / "data" / "halden_snippets.jsonl"

QUERIES = [
    "How do I reset the zero point on the pressure transmitter?",
    "How many vacation days do staff in Germany get?",
    "Does the PX-200 need to be recalibrated regularly?",
    "HX-441 accuracy",
]


def make_embedder(use_hashing: bool) -> Embedder:
    if use_hashing:
        return HashingEmbedder()
    return SentenceTransformerEmbedder(
        "BAAI/bge-small-en-v1.5",
        query_prefix=(
            "Represent this sentence for searching relevant passages: "
        ),
    )


# [start:main]
async def main(use_hashing: bool) -> None:
    rows = [json.loads(line) for line in DATA.open()]
    embedder = make_embedder(use_hashing)
    matrix = await embedder.embed_documents([r["text"] for r in rows])
    print(f"model={embedder.model_id} shape={matrix.shape}\n")

    for q in QUERIES:
        qvec = await embedder.embed_query(q)
        print(f"Q: {q}")
        for idx, score in top_k_cosine(qvec, matrix, k=3):
            print(f"  {score:.3f}  {rows[idx]['title']}")
        print()


# [end:main]
if __name__ == "__main__":
    asyncio.run(main("--hashing" in sys.argv))
