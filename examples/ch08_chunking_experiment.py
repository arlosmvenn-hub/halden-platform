"""Chapter 8: one corpus, four chunking strategies, same queries.

For each strategy we embed every chunk, run each query, and report
the rank of the first chunk that contains the answer text.
"""

import asyncio
from pathlib import Path

import numpy as np

from halden.adapters.local_embedder import SentenceTransformerEmbedder
from halden.ingestion.chunking import (
    chunk_document,
    hf_token_counter,
    split_fixed,
    split_recursive,
)
from halden.ingestion.markdown_parser import load_manual

MODEL = "BAAI/bge-small-en-v1.5"
PREFIX = "Represent this sentence for searching relevant passages: "
MANUALS = Path(__file__).parent.parent / "data" / "manuals"

# (query, text that must appear in a chunk for it to count as a hit)
# [start:queries]
QUERIES = [
    ("What does error E-17 mean on the PX-200?", "diaphragm fault"),
    ("What does E-17 mean on the flow meter?", "Empty pipe"),
    (
        "Does the PX-200 need periodic recalibration?",
        "does not require periodic recalibration",
    ),
    ("What loop resistance does HART need?", "250 ohms"),
    (
        "PX-200 reading drifts upward in gas service, why?",
        "condensate accumulating",
    ),
    ("What is the accuracy of the PX-200?", "0.075%"),
    (
        "Can the flow meter measure deionized water?",
        "cannot measure hydrocarbons",
    ),
    (
        "How much straight pipe is needed upstream of the FM-310?",
        "ten pipe diameters",
    ),
]
# [end:queries]


def build_strategies(count: object) -> dict[str, list[tuple[str, str]]]:
    """Return {strategy: [(text_shown, text_embedded), ...]}."""
    assert callable(count)
    docs = [
        load_manual(p, ["everyone"])
        for p in sorted(MANUALS.glob("*.md"))
    ]
    flat = {
        d.doc_id: "\n\n".join(s.text for s in d.sections) for d in docs
    }
    out: dict[str, list[tuple[str, str]]] = {}
    out["A fixed 500 chars"] = [
        (c, c) for t in flat.values() for c in split_fixed(t, 500)
    ]
    out["B recursive 120 tok"] = [
        (c, c)
        for t in flat.values()
        for c in split_recursive(t, 120, count)
    ]
    for label, header in (
        ("C structure 120 tok", False),
        ("D structure + header", True),
    ):
        out[label] = [
            (c.text, c.embed_text)
            for d in docs
            for c in chunk_document(
                d, max_tokens=120, count=count, context_header=header
            )
        ]
    return out


async def main() -> None:
    count = hf_token_counter(MODEL)
    emb = SentenceTransformerEmbedder(MODEL, query_prefix=PREFIX)
    strategies = build_strategies(count)
    print(
        f"{'strategy':<22}{'chunks':>7}{'hit@1':>7}"
        f"{'hit@3':>7}{'tokens to answer':>18}"
    )
    for name, chunks in strategies.items():
        matrix = await emb.embed_documents([e for _, e in chunks])
        sizes = [count(text) for text, _ in chunks]
        ranks, costs = [], []
        for query, answer in QUERIES:
            scores = matrix @ await emb.embed_query(query)
            order = [int(i) for i in np.argsort(-scores)]
            rank = next(
                (
                    r
                    for r, i in enumerate(order)
                    if answer in chunks[i][0]
                ),
                None,
            )
            ranks.append(rank)
            if rank is not None:
                # Tokens the model must read to reach the answer.
                costs.append(sum(sizes[i] for i in order[: rank + 1]))
        n = len(QUERIES)
        h1 = sum(r == 0 for r in ranks)
        h3 = sum(r is not None and r < 3 for r in ranks)
        avg = sum(costs) / len(costs) if costs else float("nan")
        print(
            f"{name:<22}{len(chunks):>7}{h1:>5}/{n}"
            f"{h3:>5}/{n}{avg:>18.0f}"
        )


if __name__ == "__main__":
    asyncio.run(main())
