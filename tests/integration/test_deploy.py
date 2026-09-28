"""Deployment safety checks."""

import pytest
from psycopg_pool import AsyncConnectionPool

from halden.adapters.local_embedder import HashingEmbedder
from halden.config.settings import Settings
from halden.ingestion.service import IngestionService
from halden.services.container import check_index_model
from halden.store.db import create_pool
from halden.store.documents import DocumentStore
from tests.integration.conftest import BASE_DSN


async def test_mismatched_index_model_refuses_to_start(
    pool: AsyncConnectionPool,
) -> None:
    emb = HashingEmbedder(dimensions=384)
    await IngestionService(
        DocumentStore(pool), emb, lambda t: len(t.split()), 60
    ).ingest(
        doc_id="d",
        system="s",
        filename="d.md",
        data=b"# D\n\n## 1 S\n\nText.",
        acl=["everyone"],
        source_version="1",
    )
    await check_index_model(
        pool, Settings(embedding_model=emb.model_id)
    )
    with pytest.raises(RuntimeError, match="holds vectors from"):
        await check_index_model(pool, Settings())


async def test_index_schema_puts_that_schema_first() -> None:
    pool = create_pool(Settings(dsn=BASE_DSN, index_schema="green"))
    await pool.open()
    async with pool.connection() as conn:
        cur = await conn.execute("SHOW search_path")
        row = await cur.fetchone()
    await pool.close()
    assert row is not None and row[0] == "green,public"
