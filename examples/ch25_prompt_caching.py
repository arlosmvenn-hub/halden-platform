"""Chapter 25: provider prompt caching on a long, fixed prefix.

Three questions about the same two manuals, placed whole in the
system prompt. The first call writes the prefix to the provider's
cache; the next two read it. Live model.
"""

import asyncio
import os
from pathlib import Path

from halden.adapters.anthropic_llm import AnthropicLLM
from halden.observability.cost import cost_usd
from halden.ports.llm import Message, Usage

MANUALS = Path(__file__).parent.parent / "data/manuals"
QUESTIONS = [
    "What does E-17 mean on each product?",
    "Which product needs a longer straight pipe run?",
    "How often is the PX-200 zero verified in steam service?",
]


async def main() -> None:
    key, model = (
        os.environ.get("ANTHROPIC_API_KEY"),
        os.environ.get("HALDEN_MODEL"),
    )
    if not key or not model:
        print("Set ANTHROPIC_API_KEY and HALDEN_MODEL to run this.")
        return
    # [start:cached-call]
    manuals = "\n\n".join(
        p.read_text() for p in sorted(MANUALS.glob("*.md"))
    )
    system = (
        "Answer in one sentence from these manuals only.\n\n" + manuals
    )
    llm = AnthropicLLM(key, model, prompt_cache=True)
    # [end:cached-call]
    print(
        f"{'call':<6}{'uncached':>9}{'written':>9}{'read':>7}"
        f"{'output':>8}{'cost':>10}{'no cache':>10}"
    )
    for i, q in enumerate(QUESTIONS, start=1):
        r = await llm.complete([Message.user(q)], system=system)
        u = r.usage
        flat = Usage(  # the same call billed without caching
            input_tokens=u.total_input_tokens,
            output_tokens=u.output_tokens,
        )
        print(
            f"{i:<6}{u.input_tokens:>9}"
            f"{u.cache_creation_input_tokens:>9}"
            f"{u.cache_read_input_tokens:>7}{u.output_tokens:>8}"
            f"{cost_usd(model, u) or 0:>10.5f}"
            f"{cost_usd(model, flat) or 0:>10.5f}"
        )
        print(f"      {r.text[:66]}")
    await llm.aclose()


if __name__ == "__main__":
    asyncio.run(main())
