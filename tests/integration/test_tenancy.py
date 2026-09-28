"""Tenant isolation: shared tables, row-level security."""

import random
from datetime import timedelta

import psycopg
import pytest
from psycopg_pool import AsyncConnectionPool

from halden.adapters.fake_llm import ScriptedLLM
from halden.adapters.local_embedder import HashingEmbedder
from halden.config.settings import Settings
from halden.ingestion.service import IngestionService
from halden.security.identity import Identity
from halden.services.ask import AskService
from halden.services.guarded import CachedAsk
from halden.store.cache import AnswerCache
from halden.store.documents import DocumentStore

EMB = HashingEmbedder(dimensions=384)
WORDS = [
    "pump",
    "valve",
    "sensor",
    "drift",
    "flange",
    "torque",
    "seal",
    "gasket",
]


async def ingest(
    pool: AsyncConnectionPool,
    tenant: str,
    doc_id: str,
    text: str,
    acl: list[str],
) -> None:
    await IngestionService(
        DocumentStore(pool, tenant), EMB, lambda t: len(t.split()), 60
    ).ingest(
        doc_id=doc_id,
        system="t",
        filename=f"{doc_id}.md",
        data=f"# {doc_id}\n\n## 1 Body\n\n{text}".encode(),
        acl=acl,
        source_version="1",
    )


def svc(pool: AsyncConnectionPool) -> AskService:
    return AskService(pool, EMB, ScriptedLLM([]), Settings())


# [start:isolation]
async def test_same_groups_different_tenants_see_nothing_shared(
    pool: AsyncConnectionPool,
) -> None:
    await ingest(pool, "halden", "h1", "pump seal torque", ["everyone"])
    await ingest(pool, "acme", "a1", "pump seal torque", ["everyone"])
    for tenant, own in (("halden", "h1"), ("acme", "a1")):
        who = Identity(user_id="u", groups=["everyone"], tenant=tenant)
        hits = await svc(pool).retrieve("pump seal torque", who)
        assert hits and {h.doc_id for h in hits} == {own}


async def test_a_tenant_cannot_overwrite_another_tenants_document(
    pool: AsyncConnectionPool,
) -> None:
    await ingest(pool, "halden", "shared-id", "original", ["everyone"])
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        await ingest(pool, "acme", "shared-id", "hijack", ["everyone"])


async def test_writes_without_a_tenant_fail_closed(
    pool: AsyncConnectionPool,
) -> None:
    async with pool.connection() as conn:
        await conn.execute(
            "SELECT set_config('halden.writer','on',false)"
        )
        with pytest.raises(psycopg.errors.UndefinedObject):
            await conn.execute(  # tenant_id default needs a tenant
                "INSERT INTO documents (doc_id, title, source_system,"
                " source_uri, content_hash, acl_groups)"
                " VALUES ('x', 'x', 's', 'u', 'h', '{everyone}')"
            )


# [end:isolation]


async def test_randomized_tenants_and_groups(
    pool: AsyncConnectionPool,
) -> None:
    rng = random.Random(26)
    tenants, groups = ["halden", "acme", "zenith"], ["a", "b", "c"]
    docs = {}
    for i in range(18):
        tenant = rng.choice(tenants)
        acl = rng.sample(groups, rng.randint(1, 2))
        doc_id = f"{tenant}-d{i}"
        text = " ".join(rng.choices(WORDS, k=12))
        await ingest(pool, tenant, doc_id, text, acl)
        docs[doc_id] = (tenant, set(acl))
    for _ in range(30):
        who = Identity(
            user_id="u",
            groups=rng.sample(groups, rng.randint(1, 3)),
            tenant=rng.choice(tenants),
        )
        query = " ".join(rng.choices(WORDS, k=3))
        for hit in await svc(pool).retrieve(query, who):
            tenant, acl = docs[hit.doc_id]
            assert tenant == who.tenant
            assert acl & set(who.groups)


async def test_answer_cache_never_crosses_tenants(
    pool: AsyncConnectionPool,
) -> None:
    await ingest(pool, "halden", "h1", "pump seal torque", ["everyone"])
    await ingest(pool, "acme", "a1", "pump seal torque", ["everyone"])
    llm = ScriptedLLM(["Halden answer [1].", "Acme answer [1]."])
    settings = Settings(model="m")
    ask = CachedAsk(
        AskService(pool, EMB, llm, settings),
        AnswerCache(pool, timedelta(hours=1)),
        settings,
    )
    q = "What torque for the pump seal?"
    h = Identity(user_id="h", groups=["everyone"], tenant="halden")
    a = Identity(user_id="a", groups=["everyone"], tenant="acme")
    assert (await ask.ask(q, h)).answer.startswith("Halden")
    second = await ask.ask(q, a)
    assert second.answer.startswith("Acme") and not second.cached
