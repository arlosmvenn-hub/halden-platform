"""Chapter 13: vague questions - single query vs multi-query vs HyDE."""

import asyncio
import time

from examples.live_common import DSN, embedder, rank_of, require_llm
from halden.retrieval.pg_retriever import PgRetriever
from halden.retrieval.query import generate_variants, hyde, multi_query

# [start:vague]
VAGUE = [
    (
        "my pressure sensor threw a code after a big spike and now "
        "reads nonsense",
        "diaphragm",
    ),
    (
        "water is definitely moving but the mag meter says nothing",
        "electrodes",
    ),
    ("how long have I got to file travel costs", "30 days"),
    ("who signs off if I want to fly business", "VP approval"),
    ("what do you put the temperature probe into", "thermowell"),
    ("the HART thing won't talk to my transmitter", "250"),
    ("can the mag meter do diesel", "cannot measure hydrocarbons"),
    (
        "is there a newer version of the old HX switch",
        "replacement is the HX-4410",
    ),
    ("what do I have to wear in the cal lab", "Safety glasses"),
    ("how many days can I WFH", "three days per week"),
    (
        "page came in about a safety thing, who do I escalate to",
        "duty manager",
    ),
    ("what voltage does the pressure transmitter want", "10.5"),
]
# [end:vague]


async def main() -> None:
    llm = require_llm("small")
    emb = embedder()
    r = await PgRetriever.connect(DSN, emb)
    g = ["everyone"]
    ranks: dict[str, list[int | None]] = {
        "single": [],
        "multi-query": [],
        "HyDE": [],
    }
    ms: dict[str, list[float]] = {k: [] for k in ranks}
    for q, answer in VAGUE:
        t0 = time.perf_counter()
        ranks["single"].append(
            rank_of(await r.hybrid(q, g, 10), answer)
        )
        t1 = time.perf_counter()
        variants = [q, *await generate_variants(llm, q)]
        mq = await multi_query(r, variants, g, 10)
        ranks["multi-query"].append(rank_of(mq, answer))
        t2 = time.perf_counter()
        hy = await hyde(llm, emb, r, q, g, 10)
        ranks["HyDE"].append(rank_of(hy, answer))
        t3 = time.perf_counter()
        for k, dt in zip(
            ranks, (t1 - t0, t2 - t1, t3 - t2), strict=True
        ):
            ms[k].append(dt * 1000)

    n = len(VAGUE)
    print(
        f"{'method':<12}{'hit@1':>7}{'hit@3':>7}{'MRR':>6}{'avg ms':>8}"
    )
    for k, rs in ranks.items():
        h1 = sum(x == 1 for x in rs)
        h3 = sum(x is not None and x <= 3 for x in rs)
        mrr = sum(1 / x for x in rs if x) / n
        avg = sum(ms[k]) / n
        print(f"{k:<12}{h1:>4}/{n}{h3:>4}/{n}{mrr:>6.2f}{avg:>8.0f}")
    print("\nper question (rank, - = not in top 10):")
    for i, (q, _) in enumerate(VAGUE):
        cells = "  ".join(f"{k}={ranks[k][i] or '-'}" for k in ranks)
        print(f"  {q[:40]:<40} {cells}")
    await r.aclose()


if __name__ == "__main__":
    asyncio.run(main())
