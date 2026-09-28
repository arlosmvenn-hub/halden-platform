"""Erasure leaves no trace in any table."""

from datetime import timedelta

import psycopg
from psycopg_pool import AsyncConnectionPool

from halden.adapters.fake_llm import ScriptedLLM
from halden.adapters.local_embedder import HashingEmbedder
from halden.config.settings import Settings
from halden.ingestion.service import IngestionService
from halden.security.api_keys import create_key, create_user
from halden.security.forget import forget_document, forget_user
from halden.security.identity import Identity
from halden.services.ask import AskService
from halden.services.guarded import CachedAsk
from halden.store.cache import AnswerCache
from halden.store.conversations import ConversationStore
from halden.store.documents import DocumentStore

EMB = HashingEmbedder(dimensions=384)
MARK = "zircon-7731"  # appears only in the data being erased


# [start:scan]
async def tables_containing(
    pool: AsyncConnectionPool, needle: str
) -> list[str]:
    """Every table with a text-like column holding ``needle``."""
    found = []
    async with pool.connection() as conn:
        await conn.execute(  # see all rows, all tenants
            "SELECT set_config('halden.writer', 'on', false),"
            " set_config('halden.tenant', 'halden', false)"
        )
        cur = await conn.execute(
            "SELECT table_name, column_name FROM information_schema"
            ".columns WHERE table_schema = current_schema()"
            " AND data_type IN ('text', 'jsonb', 'ARRAY')"
        )
        for table, column in await cur.fetchall():
            hit = await conn.execute(
                psycopg.sql.SQL(
                    "SELECT 1 FROM {} WHERE {}::text ILIKE %s LIMIT 1"
                ).format(
                    psycopg.sql.Identifier(table),
                    psycopg.sql.Identifier(column),
                ),
                (f"%{needle}%",),
            )
            if await hit.fetchone():
                found.append(f"{table}.{column}")
    return sorted(set(found))


# [end:scan]


async def test_forgetting_a_document_reaches_the_cache(
    pool: AsyncConnectionPool,
) -> None:
    await IngestionService(
        DocumentStore(pool), EMB, lambda t: len(t.split()), 60
    ).ingest(
        doc_id="hr:case",
        system="hr",
        filename="case.md",
        data=f"# Case\n\n## 1 Notes\n\nCode {MARK} applies.".encode(),
        acl=["everyone"],
        source_version="1",
    )
    settings = Settings(model="m")
    ask = CachedAsk(
        AskService(
            pool, EMB, ScriptedLLM([f"It is {MARK} [1]."]), settings
        ),
        AnswerCache(pool, timedelta(hours=1)),
        settings,
    )
    await ask.ask(
        "Which code applies?",
        Identity(user_id="u", groups=["everyone"]),
    )
    assert await tables_containing(pool, MARK) == [
        "answer_cache.answer",
        "chunks.embed_text",
        "chunks.text",
    ]
    erased = await forget_document(pool, "hr:case")
    assert erased.documents == 1 and erased.cache_entries == 1
    assert await tables_containing(pool, MARK) == []


async def test_forgetting_a_person_removes_their_traces(
    pool: AsyncConnectionPool,
) -> None:
    await create_user(pool, "bob", "Bob", ["everyone"])
    await create_key(pool, "bob")
    convs = ConversationStore(pool)
    cid = await convs.create("bob")
    await convs.append(cid, f"My badge is {MARK}", "Noted.")
    assert "messages.text" in await tables_containing(pool, MARK)
    erased = await forget_user(pool, "bob")
    assert erased.conversations == 1 and erased.api_keys == 1
    assert await tables_containing(pool, MARK) == []
    assert await tables_containing(pool, "bob") == []
