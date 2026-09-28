"""Chapter 15: a whole-corpus question three ways."""

import asyncio
import json
import time
from pathlib import Path

from examples.live_common import DSN, embedder, require_llm
from halden.generation.context import build_context
from halden.generation.grounded import SYSTEM_PROMPT, build_messages
from halden.generation.mapreduce import map_reduce
from halden.ingestion.markdown_parser import load_manual
from halden.ports.llm import Message
from halden.retrieval.pg_retriever import PgRetriever

ROOT = Path(__file__).parent.parent
TASK = "List every safety requirement or safety warning Halden gives."
# [start:facts]
FACTS = [
    "lockout",
    "safety glasses",
    "hearing protection",
    "thermowell",
    "decontamination",
    "burn",
    "depressurize",
]
# [end:facts]


def corpus() -> list[tuple[str, str]]:
    docs = []
    for p in sorted((ROOT / "data" / "manuals").glob("*.md")):
        d = load_manual(p, ["everyone"])
        docs.append((d.title, "\n\n".join(s.text for s in d.sections)))
    snippets = [
        json.loads(x)
        for x in (ROOT / "data" / "halden_snippets.jsonl").open()
    ]
    public = [
        s for s in snippets if not s["doc_id"].startswith("contract")
    ]
    docs.append(
        (
            "Policies, wiki, tickets, and notices",
            "\n".join(f"- {s['title']}: {s['text']}" for s in public),
        )
    )
    return docs


def covered(text: str) -> str:
    t = text.lower()
    missed = [f for f in FACTS if f not in t]
    found = len(FACTS) - len(missed)
    return f"{found}/{len(FACTS)}  missed: {', '.join(missed) or '-'}"


async def main() -> None:
    small, large = require_llm("small"), require_llm("answer")
    docs = corpus()
    print(f"{'approach':<24}{'calls':>6}{'tokens':>8}{'sec':>6}  facts")

    r = await PgRetriever.connect(DSN, embedder())
    t0 = time.perf_counter()
    ctx = build_context(
        await r.hybrid(TASK, ["everyone"], 8), budget_tokens=1500
    )
    resp = await large.complete(
        build_messages(TASK, ctx), system=SYSTEM_PROMPT, max_tokens=1500
    )
    tok = resp.usage.input_tokens + resp.usage.output_tokens
    sec = time.perf_counter() - t0
    print(
        f"{'RAG (top 8 chunks)':<24}{1:>6}{tok:>8}{sec:>6.1f}  "
        f"{covered(resp.text)}"
    )
    await r.aclose()

    t0 = time.perf_counter()
    mr = await map_reduce(small, large, TASK, docs)
    sec = time.perf_counter() - t0
    flag = "  (TRUNCATED)" if mr.truncated else ""
    print(
        f"{'map-reduce':<24}{mr.calls:>6}{mr.tokens:>8}{sec:>6.1f}  "
        f"{covered(mr.answer)}{flag}"
    )

    t0 = time.perf_counter()
    everything = "\n\n".join(f"# {t}\n{x}" for t, x in docs)
    resp = await large.complete(
        [Message.user(f"{everything}\n\nTask: {TASK}")],
        system="Answer using only the documents provided.",
        max_tokens=1500,
    )
    tok = resp.usage.input_tokens + resp.usage.output_tokens
    sec = time.perf_counter() - t0
    print(
        f"{'whole corpus in context':<24}{1:>6}{tok:>8}{sec:>6.1f}  "
        f"{covered(resp.text)}"
    )


if __name__ == "__main__":
    asyncio.run(main())
