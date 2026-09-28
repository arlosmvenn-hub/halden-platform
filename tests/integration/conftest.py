"""Shared fixtures: a migrated, throwaway schema per test."""

import os
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from urllib.parse import quote

import psycopg
import pytest
from psycopg_pool import AsyncConnectionPool

from halden.config.settings import Settings
from halden.store.db import create_pool, migrate

BASE_DSN = os.environ.get(
    "HALDEN_DSN", "postgresql://halden:halden@localhost/halden"
)
MIGRATIONS = Path(__file__).parents[2] / "migrations"


@pytest.fixture
async def schema_dsn() -> AsyncIterator[str]:
    try:
        admin = await psycopg.AsyncConnection.connect(
            BASE_DSN, autocommit=True
        )
    except psycopg.OperationalError as exc:
        pytest.skip(f"no database: {exc}")
    schema = f"t_{uuid.uuid4().hex[:10]}"
    await admin.execute(f"CREATE SCHEMA {schema}")
    sep = "&" if "?" in BASE_DSN else "?"
    opts = quote(f"-csearch_path={schema},public", safe="")
    dsn = f"{BASE_DSN}{sep}options={opts}"
    async with await psycopg.AsyncConnection.connect(
        dsn, autocommit=True
    ) as conn:
        await migrate(conn, MIGRATIONS)
    yield dsn
    await admin.execute(f"DROP SCHEMA {schema} CASCADE")
    await admin.close()


@pytest.fixture
async def pool(schema_dsn: str) -> AsyncIterator[AsyncConnectionPool]:
    p = create_pool(Settings(dsn=schema_dsn, db_pool_max=8))
    await p.open()
    yield p
    await p.close()
