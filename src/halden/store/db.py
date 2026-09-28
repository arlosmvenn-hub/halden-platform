"""Connection pool and schema migrations."""

import asyncio
import logging
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector_async
from psycopg_pool import AsyncConnectionPool

from halden.config.settings import Settings

log = logging.getLogger(__name__)


# [start:pool]
async def _configure(conn: psycopg.AsyncConnection) -> None:
    await register_vector_async(conn)


def create_pool(settings: Settings) -> AsyncConnectionPool:
    """One pool per process. Connections are borrowed per request
    (``async with pool.connection() as conn``) and returned, so a
    burst of requests queues for a connection instead of opening
    hundreds of them."""
    kwargs = {}
    if settings.index_schema != "public":  # Chapter 27
        path = f"{settings.index_schema},public"
        kwargs["options"] = f"-csearch_path={path}"
    return AsyncConnectionPool(
        settings.dsn,
        min_size=settings.db_pool_min,
        max_size=settings.db_pool_max,
        configure=_configure,
        kwargs=kwargs,
        open=False,  # opened explicitly at startup
    )


# [end:pool]


# [start:migrate]
def _read_migrations(directory: Path) -> list[tuple[str, str]]:
    files = sorted(directory.glob("[0-9][0-9][0-9]_*.sql"))
    return [(f.name, f.read_text()) for f in files]


async def migrate(
    conn: psycopg.AsyncConnection, directory: Path
) -> list[str]:
    """Apply migrations/NNN_*.sql in order, each exactly once.

    Each file runs in its own transaction together with the row
    that records it, so a failed migration leaves no half-applied
    state and is retried on the next start. An advisory lock stops
    two instances starting at once from racing.
    """
    applied_now: list[str] = []
    await conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        " name text PRIMARY KEY,"
        " applied_at timestamptz NOT NULL DEFAULT now())"
    )
    await conn.execute("SELECT pg_advisory_lock(427001)")
    try:
        cur = await conn.execute("SELECT name FROM schema_migrations")
        done = {row[0] for row in await cur.fetchall()}
        # File reads are blocking I/O: keep them off the event loop.
        files = await asyncio.to_thread(_read_migrations, directory)
        for name, sql in files:
            if name in done:
                continue
            async with conn.transaction():
                await conn.execute(sql)
                await conn.execute(
                    "INSERT INTO schema_migrations (name) VALUES (%s)",
                    (name,),
                )
            log.info("applied migration %s", name)
            applied_now.append(name)
    finally:
        await conn.execute("SELECT pg_advisory_unlock(427001)")
    return applied_now


# [end:migrate]
