"""Chapter 13: condensing follow-up questions before retrieval."""

import asyncio

from examples.live_common import DSN, embedder, rank_of, require_llm
from halden.retrieval.pg_retriever import PgRetriever
from halden.retrieval.query import Turn, condense_question

# [start:cases]
# (earlier turns, follow-up question, text the answer chunk contains)
CASES = [
    (
        [
            Turn(
                role="user", text="What does E-17 mean on the PX-200?"
            ),
            Turn(
                role="assistant", text="A sensor diaphragm fault [1]."
            ),
        ],
        "How do I send it back?",
        "decontamination form",
    ),
    (
        [
            Turn(
                role="user", text="What's the accuracy of the FM-310?"
            ),
            Turn(role="assistant", text="±0.3% of reading [1]."),
        ],
        "And its temperature limits?",
        "150 °C",
    ),
    (
        [
            Turn(role="user", text="Is the HX-441 still available?"),
            Turn(
                role="assistant",
                text="No, it is discontinued; the replacement is the "
                "HX-4410 [1].",
            ),
        ],
        "How accurate is the replacement?",
        "±0.1 °C",
    ),
    (
        [
            Turn(role="user", text="My FM-310 shows E-17."),
            Turn(role="assistant", text="That means empty pipe [1]."),
        ],
        "Where should it be installed, then?",
        "rising pipe",
    ),
    (
        [
            Turn(
                role="user",
                text="How much annual leave do I get in Germany?",
            ),
            Turn(role="assistant", text="28 days [1]."),
        ],
        "When are expense reports due?",
        "30 days",
    ),
]
# [end:cases]


async def main() -> None:
    llm = require_llm("small")
    r = await PgRetriever.connect(DSN, embedder())
    for history, follow_up, answer in CASES:
        condensed = await condense_question(llm, history, follow_up)
        raw = rank_of(
            await r.hybrid(follow_up, ["everyone"], 10), answer
        )
        new = rank_of(
            await r.hybrid(condensed, ["everyone"], 10), answer
        )
        print(f"follow-up: {follow_up}")
        print(f"condensed: {condensed}")
        print(
            f"answer rank: raw={raw or '-'}  condensed={new or '-'}\n"
        )
    await r.aclose()


if __name__ == "__main__":
    asyncio.run(main())
