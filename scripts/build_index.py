"""Build a complete second copy of the index, for a blue/green
change of embedding model (Chapter 27).

    uv run python scripts/build_index.py green BAAI/bge-base-en-v1.5

The new copy lives in its own schema with only the index tables
(documents, chunks, jobs, index_state, answer_cache). Users, keys,
and conversations stay in "public" and are shared. Evaluate it with
HALDEN_INDEX_SCHEMA=green HALDEN_EMBEDDING_MODEL=... scripts/eval.py,
then switch by deploying that pair of settings. Re-running only
catches up on documents that changed since the last run.
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

import psycopg
from psycopg import sql

from halden.adapters.local_embedder import SentenceTransformerEmbedder
from halden.config.settings import Settings
from halden.ingestion.chunking import hf_token_counter
from halden.ingestion.folder_connector import sync_folder
from halden.ingestion.service import IngestionService
from halden.ingestion.worker import make_handler, run_worker
from halden.store.db import create_pool
from halden.store.documents import DocumentStore
from halden.store.jobs import JobQueue

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ["manuals", "policies", "contracts", "tickets"]

# [start:schema]
TABLES = ["documents", "chunks", "jobs", "index_state", "answer_cache"]
SECURE = """
ALTER TABLE {s}.chunks ADD FOREIGN KEY (doc_id)
    REFERENCES {s}.documents ON DELETE CASCADE;
INSERT INTO {s}.index_state DEFAULT VALUES;
CREATE TRIGGER chunks_changed
    AFTER INSERT OR UPDATE OR DELETE OR TRUNCATE ON {s}.chunks
    FOR EACH STATEMENT EXECUTE FUNCTION public.bump_index_generation();
ALTER TABLE {s}.chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE {s}.chunks FORCE ROW LEVEL SECURITY;
ALTER TABLE {s}.documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE {s}.documents FORCE ROW LEVEL SECURITY;
"""
POLICIES = """
CREATE POLICY {reader} ON {s}.{t} FOR SELECT USING (
    tenant_id = halden_tenant() AND acl_groups && halden_groups());
CREATE POLICY {writer} ON {s}.{t}
    USING (halden_writer() AND tenant_id = halden_tenant())
    WITH CHECK (halden_writer() AND tenant_id = halden_tenant());
"""


async def create_schema(
    conn: psycopg.AsyncConnection, schema: str, dims: int
) -> None:
    """Index tables only, copied from public (columns, defaults,
    indexes), then the parts LIKE doesn't copy: foreign key,
    trigger, row-level security. A new model may change the
    vector size, so the column and its HNSW index are rebuilt."""
    s = sql.Identifier(schema)
    await conn.execute(sql.SQL("CREATE SCHEMA {}").format(s))
    for t in TABLES:
        await conn.execute(
            sql.SQL(
                "CREATE TABLE {}.{} (LIKE public.{} INCLUDING ALL)"
            ).format(s, sql.Identifier(t), sql.Identifier(t))
        )
    await conn.execute(sql.SQL(SECURE).format(s=s))
    for t in ("chunks", "documents"):
        await conn.execute(
            sql.SQL(POLICIES).format(
                s=s,
                t=sql.Identifier(t),
                reader=sql.Identifier(f"{t}_reader"),
                writer=sql.Identifier(f"{t}_writer"),
            )
        )
    cur = await conn.execute(
        "SELECT indexname FROM pg_indexes WHERE schemaname = %s"
        " AND tablename = 'chunks' AND indexdef LIKE '%%hnsw%%'",
        (schema,),
    )
    for (name,) in await cur.fetchall():
        await conn.execute(
            sql.SQL("DROP INDEX {}.{}").format(s, sql.Identifier(name))
        )
    await conn.execute(
        sql.SQL(
            "ALTER TABLE {s}.chunks ALTER COLUMN embedding"
            " TYPE vector({d});"
            "CREATE INDEX ON {s}.chunks USING hnsw"
            " (embedding vector_cosine_ops)"
            " WITH (m = 16, ef_construction = 64)"
        ).format(s=s, d=sql.Literal(dims))
    )


# [end:schema]


async def main(schema: str, model: str, fresh: bool) -> None:
    base = Settings()
    embedder = SentenceTransformerEmbedder(model)
    async with await psycopg.AsyncConnection.connect(
        base.dsn, autocommit=True
    ) as conn:
        if fresh:  # start over, e.g. to time a full rebuild
            await conn.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                    sql.Identifier(schema)
                )
            )
        cur = await conn.execute(
            "SELECT to_regclass(%s)", (f"{schema}.chunks",)
        )
        row = await cur.fetchone()
        if row is None or row[0] is None:
            async with conn.transaction():
                await create_schema(conn, schema, embedder.dimensions)
            print(f"created schema {schema}")
    settings = Settings(index_schema=schema, embedding_model=model)
    pool = create_pool(settings)
    await pool.open()
    store, queue = DocumentStore(pool), JobQueue(pool)
    service = IngestionService(
        store,
        embedder,
        hf_token_counter(model),
        settings.chunk_max_tokens,
    )
    for system in SOURCES:
        folder = ROOT / "data/sources" / system
        await sync_folder(folder, system, store, queue)
    t0 = time.perf_counter()
    n = await run_worker(
        queue,
        make_handler(service, store),
        stop=asyncio.Event(),
        drain=True,
    )
    await pool.close()
    took = time.perf_counter() - t0
    print(
        f"{schema}: {n} document(s) indexed with {model}"
        f" in {took:.0f} s"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    args = [a for a in sys.argv[1:] if a != "--fresh"]
    asyncio.run(main(args[0], args[1], "--fresh" in sys.argv))
