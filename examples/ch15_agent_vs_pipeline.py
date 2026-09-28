"""Chapter 15: fixed RAG pipeline vs bounded agentic retrieval."""

import asyncio
import time

from examples.live_common import DSN, embedder, require_llm
from halden.agent.loop import Budget, run_agent
from halden.agent.search_tool import SearchSession
from halden.generation.context import build_context
from halden.generation.grounded import (
    SYSTEM_PROMPT,
    build_messages,
    check_citations,
)
from halden.retrieval.pg_retriever import PgRetriever

# [start:questions]
# Multi-part questions and facts a complete answer must contain.
QUESTIONS = [
    (
        "What does E-17 mean on the PX-200 and on the FM-310, and what "
        "should I do in each case?",
        ["diaphragm", "repair", "empty pipe", "full"],
    ),
    (
        "What replaced the HX-441, how accurate is it, and which "
        "firmware version fixed a drift problem?",
        ["HX-4410", "0.1", "3.2"],
    ),
    (
        "For a five-hour flight to a customer in Germany, which class "
        "can I book, and when is my expense report due?",
        ["economy", "30 days"],
    ),
    (
        "Can the FM-310 measure diesel, and what is its accuracy?",
        ["hydrocarbon", "0.3%"],
    ),
    (
        "How do I zero a PX-200, and how often should it be checked "
        "in steam service?",
        ["ZERO", "annual"],
    ),
]
# [end:questions]

AGENT_SYSTEM = (
    SYSTEM_PROMPT
    + """

You have a search_documents tool. Break multi-part questions into
focused searches, one per part. Cite the [n] labels the tool
returns."""
)


def coverage(answer: str, facts: list[str]) -> int:
    a = answer.lower()
    return sum(f.lower() in a for f in facts)


async def main() -> None:
    llm = require_llm("answer")
    r = await PgRetriever.connect(DSN, embedder())
    groups = ["everyone"]
    rows = []
    checks = []
    for q, facts in QUESTIONS:
        t0 = time.perf_counter()
        hits = await r.hybrid(q, groups, k=8)
        ctx = build_context(hits, budget_tokens=1200)
        resp = await llm.complete(
            build_messages(q, ctx), system=SYSTEM_PROMPT, max_tokens=500
        )
        p_ms = (time.perf_counter() - t0) * 1000
        p_chk = check_citations(resp.text, ctx)
        p_ok = p_chk.ok
        p_tok = resp.usage.input_tokens + resp.usage.output_tokens

        session = SearchSession(r, groups)
        t0 = time.perf_counter()
        answer, trace = await run_agent(
            llm,
            system=AGENT_SYSTEM,
            question=q,
            tools=[session.tool()],
            budget=Budget(max_steps=4, max_tool_calls=6),
        )
        a_ms = (time.perf_counter() - t0) * 1000
        a_chk = check_citations(answer, session.context())
        a_ok = a_chk.ok
        a_tok = trace.input_tokens + trace.output_tokens
        checks.append({"pipeline": p_chk, "agent": a_chk})
        rows.append(
            (
                q,
                facts,
                resp.text,
                answer,
                trace,
                (p_ok, p_tok, p_ms),
                (a_ok, a_tok, a_ms),
            )
        )

    print(f"{'':4}{'pipeline':>28}{'agent':>30}")
    print(
        f"{'Q':<4}{'facts':>7}{'cites':>7}{'tokens':>8}{'sec':>6}"
        f"{'facts':>8}{'cites':>7}{'tokens':>8}{'sec':>6}{'calls':>6}"
    )
    for i, (_q, facts, pa, aa, tr, p, a) in enumerate(rows, 1):
        n = len(facts)
        print(
            f"Q{i:<3}{coverage(pa, facts):>5}/{n}{p[0]!s:>7}{p[1]:>8}"
            f"{p[2] / 1000:>6.1f}{coverage(aa, facts):>6}/{n}"
            f"{a[0]!s:>7}{a[1]:>8}{a[2] / 1000:>6.1f}"
            f"{len(tr.tool_calls):>6}"
        )
    print("\nwhy citation checks failed:")
    for i, pair in enumerate(checks, 1):
        for who, c in pair.items():
            if c.ok:
                continue
            print(
                f"  Q{i} {who}: abstained={c.abstained} "
                f"cited={c.cited} unknown={c.unknown} "
                f"uncited={len(c.uncited_sentences)}"
            )
            for s in c.uncited_sentences[:2]:
                print(f"      uncited: {s[:64]!r}")
    _q, _f, _pa, aa, tr, _, _ = rows[0]
    print("\nQ1 agent searches:")
    for name, args in tr.tool_calls:
        print(f"  {name}({args})")
    print(f"\nQ1 agent answer:\n{aa}")
    await r.aclose()


if __name__ == "__main__":
    asyncio.run(main())
