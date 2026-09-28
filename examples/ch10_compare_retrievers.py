"""Chapter 10: dense vs lexical vs hybrid on labeled questions.

Requires the demo index: uv run python scripts/load_demo_index.py
"""

import asyncio

from examples.demo_db import DEMO_DSN as DSN
from halden.adapters.local_embedder import SentenceTransformerEmbedder
from halden.retrieval.pg_retriever import PgRetriever, RetrievedChunk

PREFIX = "Represent this sentence for searching relevant passages: "

# [start:labels]
# (question, text that must appear in a retrieved chunk)
LABELED = [
    ("What does error E-17 mean on the PX-200?", "diaphragm fault"),
    ("E-17 on the FM-310", "Empty pipe"),
    ("PX-200 accuracy", "0.075%"),
    ("Is the HX-441 still sold?", "discontinued legacy"),
    ("How many vacation days do staff in Germany get?", "28 days"),
    (
        "How do I reset the zero point on the pressure transmitter?",
        "ZERO button",
    ),
    (
        "Does the PX-200 need periodic recalibration?",
        "does not require periodic recalibration",
    ),
    ("What resistor does HART communication need?", "250"),
    (
        "Why does my flow meter read zero with liquid flowing?",
        "electrodes",
    ),
    ("ticket 1266", "loop resistor"),
    ("Who approves business class flights?", "VP approval"),
    ("Deadline for submitting expense reports", "30 days"),
    ("Can I work from home?", "three days per week"),
    ("Can the FM-310 measure oil?", "cannot measure hydrocarbons"),
    ("Upstream straight run for the FM-310", "ten pipe diameters"),
    ("What does the HIL lab do?", "simulates process conditions"),
]
# [end:labels]


def rank_of(hits: list[RetrievedChunk], answer: str) -> int | None:
    for i, c in enumerate(hits, start=1):
        if answer in c.text:
            return i
    return None


async def main() -> None:
    emb = SentenceTransformerEmbedder(
        "BAAI/bge-small-en-v1.5", query_prefix=PREFIX
    )
    r = await PgRetriever.connect(DSN, emb)
    groups = ["everyone"]

    async def lexical_all(
        q: str, g: list[str], k: int
    ) -> list[RetrievedChunk]:
        return await r.lexical(q, g, k, any_term=False)

    methods = {
        "dense": r.dense,
        "lex-all": lexical_all,
        "lex-any": r.lexical,
        "hybrid": r.hybrid,
    }
    ranks: dict[str, list[int | None]] = {m: [] for m in methods}
    for q, answer in LABELED:
        for name, fn in methods.items():
            ranks[name].append(rank_of(await fn(q, groups, 10), answer))

    n = len(LABELED)
    print(f"{'method':<8}{'hit@1':>7}{'hit@5':>7}{'MRR':>7}")
    for name, rs in ranks.items():
        h1 = sum(x == 1 for x in rs)
        h5 = sum(x is not None and x <= 5 for x in rs)
        mrr = sum(1 / x for x in rs if x) / n
        print(f"{name:<8}{h1:>4}/{n}{h5:>4}/{n}{mrr:>7.2f}")

    print("\nquestions where the methods disagree (rank, - = miss):")
    for i, (q, _) in enumerate(LABELED):
        row = [ranks[m][i] for m in methods]
        if len(set(row)) > 1:
            cells = "  ".join(
                f"{m}={x or '-'}"
                for m, x in zip(methods, row, strict=True)
            )
            print(f"  {q[:34]:<34} {cells}")

    # Permissions are part of the query, not a post-filter.
    q = "Acme Metals lead time"
    for g in (["everyone"], ["legal"]):
        hits = await r.hybrid(q, g, 5)
        found = any("45-day" in c.text for c in hits)
        print(f"\n'{q}' as {g}: contract found = {found}", end="")
    print()
    await r.aclose()


if __name__ == "__main__":
    asyncio.run(main())
