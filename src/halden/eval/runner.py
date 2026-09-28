"""Run a dataset through the real AskService and grade every case."""

import asyncio
import time
from collections.abc import Sequence

from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel

from halden.eval.dataset import EvalCase
from halden.eval.judge import judge_correctness, judge_faithfulness
from halden.eval.metrics import (
    coverage,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from halden.ports.llm import LLMClient
from halden.security.identity import Identity
from halden.services.ask import AskService
from halden.store.scope import writer_scope


# [start:result]
class CaseResult(BaseModel):
    id: str
    tags: list[str]
    answerable: bool
    retrieved: list[str]  # chunk IDs, rank order
    # Retrieval (answerable cases only)
    hit: float | None = None
    precision: float | None = None
    recall: float | None = None
    rr: float | None = None
    ndcg: float | None = None
    context_recall: float | None = None  # evidence in the prompt
    retrieve_ms: float = 0.0
    # Generation (only when an answering model is configured)
    answer: str | None = None
    abstained: bool | None = None
    abstention_ok: bool | None = None
    citations_valid: bool | None = None
    citation_support: bool | None = None
    faithfulness: float | None = None
    unsupported: list[str] = []
    correctness: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0


# [end:result]


# [start:run-case]
async def run_case(
    svc: AskService,
    case: EvalCase,
    *,
    k: int = 5,
    generate: bool = False,
    judge: LLMClient | None = None,
) -> CaseResult:
    who = Identity(user_id=f"eval:{case.id}", groups=case.groups)
    t0 = time.perf_counter()
    hits = await svc.retrieve(case.question, who)
    ms = (time.perf_counter() - t0) * 1000
    ctx = svc.context(hits)
    r = CaseResult(
        id=case.id,
        tags=case.tags,
        answerable=case.answerable,
        retrieved=[h.chunk_id for h in hits[:k]],
        retrieve_ms=round(ms, 1),
    )
    if case.answerable:
        n = len(case.evidence)
        cov = coverage([h.text for h in hits], case.evidence)
        r.hit = hit_at_k(cov, k)
        r.precision = precision_at_k(cov, k)
        r.recall = recall_at_k(cov, n, k)
        r.rr = reciprocal_rank(cov)
        r.ndcg = ndcg_at_k(cov, n, k)
        in_ctx = coverage([s.text for s in ctx.sources], case.evidence)
        r.context_recall = recall_at_k(in_ctx, n, len(in_ctx) or 1)
    if not generate:
        return r

    ans = await svc.ask(case.question, who)
    r.answer, r.abstained = ans.answer, ans.abstained
    r.input_tokens, r.output_tokens = (
        ans.input_tokens,
        ans.output_tokens,
    )
    r.abstention_ok = ans.abstained != case.answerable
    r.citations_valid = ans.citation_check_ok
    if case.answerable and not ans.abstained:
        cited = {c.chunk_id for c in ans.citations}
        texts = [s.text for s in ctx.sources if s.chunk_id in cited]
        r.citation_support = any(coverage(texts, case.evidence))
    if judge is not None and not ans.abstained:
        faith = await judge_faithfulness(
            judge, case.question, ctx.render(), ans.answer
        )
        r.faithfulness = faith.score
        r.unsupported = [
            c.claim for c in faith.claims if not c.supported
        ]
    if judge is not None and case.answerable:
        if ans.abstained:
            r.correctness = "incorrect"
        else:
            verdict = await judge_correctness(
                judge, case.question, case.reference, ans.answer
            )
            r.correctness = verdict.verdict
    return r


# [end:run-case]


# [start:run-dataset]
async def run_dataset(
    svc: AskService,
    cases: Sequence[EvalCase],
    *,
    k: int = 5,
    generate: bool = False,
    judge: LLMClient | None = None,
    concurrency: int = 4,
) -> list[CaseResult]:
    """Grade every case, a few at a time. Model calls dominate the
    run time, and a bounded semaphore keeps us under rate limits."""
    gate = asyncio.Semaphore(concurrency)

    async def one(case: EvalCase) -> CaseResult:
        async with gate:
            return await run_case(
                svc, case, k=k, generate=generate, judge=judge
            )

    return list(await asyncio.gather(*(one(c) for c in cases)))


# [end:run-dataset]


async def index_texts(pool: AsyncConnectionPool) -> list[str]:
    """Every chunk's text, for checking evidence labels."""
    async with pool.connection() as conn, writer_scope(conn):
        cur = await conn.execute("SELECT text FROM chunks")
        return [row[0] for row in await cur.fetchall()]


async def index_sources(pool: AsyncConnectionPool) -> list[str]:
    """Every chunk as the model sees it: title, headings, text."""
    async with pool.connection() as conn, writer_scope(conn):
        cur = await conn.execute(
            "SELECT d.title, c.heading_path, c.text FROM chunks c "
            "JOIN documents d USING (doc_id)"
        )
        return [
            " > ".join([title, *path]) + "\n" + text
            for title, path, text in await cur.fetchall()
        ]
