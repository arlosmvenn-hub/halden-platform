"""Answer cache and daily budget, against a real database."""

from datetime import timedelta
from pathlib import Path

import pytest
from psycopg_pool import AsyncConnectionPool

from halden.adapters.fake_llm import ScriptedLLM
from halden.adapters.local_embedder import HashingEmbedder
from halden.config.settings import Settings
from halden.ingestion.service import IngestionService
from halden.ports.llm import LLMResponse, TextBlock, Usage
from halden.security.identity import Identity
from halden.services.ask import AskService
from halden.services.guarded import BudgetedAsk, CachedAsk
from halden.store.cache import AnswerCache, cache_key
from halden.store.documents import DocumentStore
from halden.store.usage import BudgetExceeded, UsageLedger

DATA = Path(__file__).parents[2] / "data"
EMB = HashingEmbedder(dimensions=384)
ALICE = Identity(user_id="alice", groups=["everyone"])
LENA = Identity(user_id="lena", groups=["legal", "everyone"])
Q = "What loop resistance does HART need?"


def reply(text: str = "At least 250 ohms [1].") -> LLMResponse:
    return LLMResponse(
        content=[TextBlock(text=text)],
        stop_reason="end_turn",
        usage=Usage(input_tokens=900, output_tokens=100),
        model="claude-sonnet-5",
    )


async def index(pool: AsyncConnectionPool, version: str = "1") -> None:
    md = (DATA / "manuals/px200_manual.md").read_bytes()
    if version != "1":  # a real content change
        md += b"\n\n## 9 Addendum\n\nNew text."
    await IngestionService(
        DocumentStore(pool), EMB, lambda t: len(t.split()), 60
    ).ingest(
        doc_id="m:px",
        system="m",
        filename="px.md",
        data=md,
        acl=["everyone"],
        source_version=version,
    )


def cached(
    pool: AsyncConnectionPool, llm: ScriptedLLM
) -> tuple[CachedAsk, AnswerCache]:
    cache = AnswerCache(pool, timedelta(hours=1))
    settings = Settings(model="claude-sonnet-5")
    svc = AskService(pool, EMB, llm, settings)
    return CachedAsk(svc, cache, settings), cache


def test_cache_key_ignores_group_order_and_spacing() -> None:
    a = cache_key("t", "What  is E-17?", ["b", "a"], 3, "fp")
    assert a == cache_key(
        "t", "what is e-17?", ["a", "b", "a"], 3, "fp"
    )
    assert a != cache_key("t", "What is E-17?", ["a"], 3, "fp")
    assert a != cache_key("t", "What is E-17?", ["a", "b"], 4, "fp")
    assert a != cache_key("u", "What  is E-17?", ["b", "a"], 3, "fp")


# [start:cache-tests]
async def test_reindexing_moves_the_generation(
    pool: AsyncConnectionPool,
) -> None:
    cache = AnswerCache(pool, timedelta(hours=1))
    before = await cache.generation()
    await index(pool)
    assert await cache.generation() > before


async def test_hit_then_miss_after_reindex(
    pool: AsyncConnectionPool,
) -> None:
    await index(pool)
    llm = ScriptedLLM([reply(), reply()])
    ask, _ = cached(pool, llm)
    first = await ask.ask(Q, ALICE)
    again = await ask.ask(Q.upper(), ALICE)  # normalized: same key
    assert not first.cached and again.cached
    assert len(llm.calls) == 1
    await index(pool, version="2")  # content changed
    assert not (await ask.ask(Q, ALICE)).cached
    assert len(llm.calls) == 2


async def test_other_groups_never_share_an_entry(
    pool: AsyncConnectionPool,
) -> None:
    await index(pool)
    llm = ScriptedLLM([reply(), reply()])
    ask, _ = cached(pool, llm)
    await ask.ask(Q, LENA)
    assert not (await ask.ask(Q, ALICE)).cached
    assert len(llm.calls) == 2


async def test_withheld_answers_are_not_cached(
    pool: AsyncConnectionPool,
) -> None:
    await index(pool)
    bad = "Made up [42]."  # cites a source that is not there, twice
    llm = ScriptedLLM([reply(bad), reply(bad), reply()])
    ask, _ = cached(pool, llm)
    assert (await ask.ask(Q, ALICE)).abstained
    assert not (await ask.ask(Q, ALICE)).cached


# [end:cache-tests]


async def test_budget_blocks_before_the_model_call(
    pool: AsyncConnectionPool,
) -> None:
    await index(pool)
    llm = ScriptedLLM([reply()])
    svc = AskService(pool, EMB, llm, Settings())
    ledger = UsageLedger(pool, daily_tokens=500)
    ask = BudgetedAsk(svc, ledger, "claude-sonnet-5")
    await ask.ask(Q, ALICE)  # under budget before: allowed
    assert await ledger.used_today("alice") == 1000
    with pytest.raises(BudgetExceeded):
        await ask.ask(Q, ALICE)
    assert len(llm.calls) == 1  # the second call never happened
    await ledger.check("lena")  # other users unaffected


def test_over_budget_is_a_429_with_retry_after(schema_dsn: str) -> None:
    from fastapi.testclient import TestClient

    from halden.api.app import create_app
    from halden.services.container import Services
    from halden.store.conversations import ConversationStore
    from halden.store.db import create_pool

    settings = Settings(dsn=schema_dsn, auth_mode="dev")

    async def factory(s: Settings) -> Services:
        pool = create_pool(s)
        await pool.open()
        ledger = UsageLedger(pool, daily_tokens=1)
        await ledger.record("dev", 5, 0, 0.0)  # already over
        svc = AskService(pool, EMB, ScriptedLLM([]), s)
        ask = BudgetedAsk(svc, ledger, "claude-sonnet-5")
        convs = ConversationStore(pool)
        return Services(s, pool, EMB, ScriptedLLM([]), ask, convs)

    with TestClient(create_app(settings, factory)) as client:
        r = client.post("/v1/ask", json={"question": Q})
    assert r.status_code == 429
    assert r.headers["content-type"] == "application/problem+json"
    assert 0 < int(r.headers["retry-after"]) <= 86_401
