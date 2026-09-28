"""Chapter 12: build a grounded prompt and check model answers.

The three answers below are *scripted* stand-ins for model output,
chosen to show what the citation check accepts and rejects.
"""

import asyncio

from examples.ch10_compare_retrievers import DSN, PREFIX
from halden.adapters.local_embedder import SentenceTransformerEmbedder
from halden.generation.context import build_context
from halden.generation.grounded import (
    SYSTEM_PROMPT,
    build_messages,
    check_citations,
)
from halden.retrieval.pg_retriever import PgRetriever

QUESTION = "What does error E-17 mean on the PX-200, and what do I do?"


async def main() -> None:
    emb = SentenceTransformerEmbedder(
        "BAAI/bge-small-en-v1.5", query_prefix=PREFIX
    )
    r = await PgRetriever.connect(DSN, emb)
    hits = await r.hybrid(QUESTION, ["everyone"], k=12)
    ctx = build_context(hits, budget_tokens=700)
    print(
        f"retrieved {len(hits)}, kept {len(ctx.sources)}, "
        f"dropped {len(ctx.dropped)}, ~{ctx.tokens} tokens\n"
    )
    for s in ctx.sources:
        where = " > ".join([s.title[:40], *s.heading_path])
        print(f"  [{s.label}] {where}")
    msgs = build_messages(QUESTION, ctx)
    prompt_chars = len(SYSTEM_PROMPT) + len(msgs[0].content[0].text)
    print(f"\nprompt size: {prompt_chars:,} characters\n")

    answers = {
        "grounded": (
            "E-17 on the PX-200 indicates a sensor diaphragm "
            "fault [1]. Return the transmitter for repair; do not "
            "attempt a field repair [1]."
        ),
        "bad citation": (
            "E-17 means a diaphragm fault [1]. It is usually "
            "caused by wiring problems [9]."
        ),
        "abstention": (
            "I don't know based on the available documents."
        ),
    }
    for name, text in answers.items():
        c = check_citations(text, ctx)
        print(
            f"{name:<13} ok={c.ok!s:<5} cited={c.cited} "
            f"unknown={c.unknown} uncited={len(c.uncited_sentences)}"
        )
    await r.aclose()


if __name__ == "__main__":
    asyncio.run(main())
