"""Chapter 14: parent expansion and graph-assisted retrieval."""

import asyncio
import json
from pathlib import Path

from examples.live_common import DSN, embedder, rank_of
from halden.ingestion.chunking import (
    Chunk,
    chunk_document,
    hf_token_counter,
)
from halden.ingestion.markdown_parser import load_manual
from halden.retrieval.brute_force import top_k_cosine
from halden.retrieval.pg_retriever import PgRetriever
from halden.retrieval.structure import (
    EntityGraph,
    expand_query,
    expand_to_parents,
)

ROOT = Path(__file__).parent.parent

# [start:complete]
# Questions whose complete answer spans several small chunks.
NEEDS = [
    (
        "List every PX-200 error code and its meaning",
        ["E-03", "E-11", "E-17", "E-22", "E-30"],
    ),
    (
        "How do I calibrate a PX-200, zero and span?",
        ["ZERO button", "SPAN button", "0.05%"],
    ),
    (
        "Everything about installing the FM-310",
        ["ten pipe diameters", "rising pipe", "grounding rings"],
    ),
]
# [end:complete]


async def parent_experiment() -> None:
    count = hf_token_counter("BAAI/bge-small-en-v1.5")
    emb = embedder()
    docs = [
        load_manual(p, ["everyone"])
        for p in sorted((ROOT / "data" / "manuals").glob("*.md"))
    ]
    small: list[Chunk] = [
        c
        for d in docs
        for c in chunk_document(d, max_tokens=40, count=count)
    ]
    matrix = await emb.embed_documents([c.embed_text for c in small])
    print(f"{len(small)} small chunks (max 40 tokens)\n")
    print(f"{'question':<44}{'chunks':>8}{'depth 2':>9}{'depth 1':>9}")
    for q, facts in NEEDS:
        top = top_k_cosine(await emb.embed_query(q), matrix, k=3)
        hits = [small[i] for i, _ in top]
        views = {
            "chunks": [h.text for h in hits],
            "depth 2": expand_to_parents(hits, small, depth=2),
            "depth 1": expand_to_parents(hits, small, depth=1),
        }
        cells = []
        for texts in views.values():
            joined = "\n".join(texts)
            found = sum(f in joined for f in facts)
            cells.append(f"{found}/{len(facts)} {count(joined):>3}t")
        print(f"{q[:43]:<44}" + "".join(f"{c:>9}" for c in cells))


# [start:multihop]
MULTI_HOP = [
    ("What is the accuracy of the HX-441's replacement?", "±0.1 °C"),
    (
        "Which firmware fix applies to the successor of the HX-441?",
        "drift issue",
    ),
    ("How should the HX-441 replacement be mounted?", "thermowell"),
]
# [end:multihop]


async def graph_experiment() -> None:
    count = hf_token_counter("BAAI/bge-small-en-v1.5")
    docs = [
        load_manual(p, ["everyone"])
        for p in sorted((ROOT / "data" / "manuals").glob("*.md"))
    ]
    chunks = [
        c
        for d in docs
        for c in chunk_document(d, max_tokens=120, count=count)
    ]
    for line in (ROOT / "data" / "halden_snippets.jsonl").open():
        row = json.loads(line)
        chunks.append(
            Chunk(
                chunk_id=row["id"],
                doc_id=row["id"],
                ordinal=0,
                heading_path=[],
                kind="text",
                text=row["text"],
                embed_text=row["text"],
                token_count=0,
            )
        )
    graph = EntityGraph.build(chunks)
    print("\nedges for HX-441:", sorted(graph.edges["HX-441"]))
    r = await PgRetriever.connect(DSN, embedder())
    print(f"\n{'question':<58}{'plain':>6}{'graph':>7}")
    for q, answer in MULTI_HOP:
        plain = rank_of(await r.hybrid(q, ["everyone"], 10), answer)
        expanded = expand_query(q, graph)
        via = rank_of(
            await r.hybrid(expanded, ["everyone"], 10), answer
        )
        print(f"{q[:57]:<58}{plain or '-':>6}{via or '-':>7}")
    await r.aclose()


async def main() -> None:
    await parent_experiment()
    await graph_experiment()


if __name__ == "__main__":
    asyncio.run(main())
