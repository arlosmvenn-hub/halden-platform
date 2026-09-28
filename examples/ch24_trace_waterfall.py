"""Chapter 24: the trace of one /ask request, as a waterfall.

Runs the real pipeline (hybrid retrieval, reranker, live model)
with an in-memory exporter, then prints each span's start, length,
and the attributes that matter for debugging.
"""

import asyncio
import os
import re
import textwrap

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from halden.config.settings import Settings
from halden.observability.telemetry import setup_telemetry
from halden.security.identity import Identity
from halden.services.container import build_services

SHOW = [
    "gen_ai.usage.input_tokens",
    "gen_ai.usage.output_tokens",
    "halden.cost_usd",
    "halden.candidates",
    "halden.context.sources",
    "halden.context.tokens",
    "halden.outcome",
    "halden.citation_check_ok",
    "gen_ai.prompt.version",
]


def depth(span: ReadableSpan, by_id: dict[int, ReadableSpan]) -> int:
    d = 0
    while span.parent is not None and span.parent.span_id in by_id:
        span = by_id[span.parent.span_id]
        d += 1
    return d


# [start:waterfall]
def waterfall(spans: list[ReadableSpan], width: int = 20) -> None:
    by_id = {s.context.span_id: s for s in spans}
    t0 = min(s.start_time or 0 for s in spans)
    total = max(s.end_time or 0 for s in spans) - t0
    for s in sorted(spans, key=lambda s: s.start_time or 0):
        start = (s.start_time or 0) - t0
        length = (s.end_time or 0) - (s.start_time or 0)
        a = int(start / total * width)
        b = max(1, int(length / total * width))
        bar = " " * a + "#" * b
        short = re.sub(r"\S+/", "", s.name)  # drop hub org names
        name = "  " * depth(s, by_id) + short
        print(f"{name:<36}{bar:<{width}}{length / 1e6:>8.1f} ms")
        attrs = s.attributes or {}
        shown = [
            f"{k.split('.')[-1]}={attrs[k]}" for k in SHOW if k in attrs
        ]
        if shown:
            print(f"{'':<6}{', '.join(shown)}")


# [end:waterfall]


async def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY and HALDEN_MODEL to run this.")
        return
    exporter = InMemorySpanExporter()
    setup_telemetry("halden-api", exporter, batch=False)
    settings = Settings(
        telemetry="console",  # enables the traced components
        reranker_model="cross-encoder/ms-marco-MiniLM-L6-v2",
    )
    services = await build_services(settings, install_exporter=False)
    who = Identity(user_id="lena", groups=["everyone", "support"])
    await services.embedder.embed_query("warm up")  # load models
    if services.reranker is not None:
        await services.reranker.score("warm up", ["models"])
    exporter.clear()
    answer = await services.ask.ask(
        "What does E-17 mean on the PX-200, and what do I do?", who
    )
    await services.pool.close()
    print(textwrap.fill(answer.answer, 68), "\n")
    waterfall(list(exporter.get_finished_spans()))


if __name__ == "__main__":
    asyncio.run(main())
