"""Transactions that declare who is reading or writing, for RLS."""

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager

import psycopg


# [start:scope]
@asynccontextmanager
async def reader_scope(
    conn: psycopg.AsyncConnection,
    groups: Sequence[str],
    tenant: str = "halden",
) -> AsyncIterator[None]:
    """Run the block in a transaction whose row-level security
    groups are the caller's. set_config(..., true) is SET LOCAL:
    it ends with the transaction, so a pooled connection never
    carries one user's groups into the next user's request."""
    async with conn.transaction():
        await conn.execute(
            "SELECT set_config('halden.user_groups', %s::text, true),"
            " set_config('halden.tenant', %s, true)",
            (list(groups), tenant),
        )
        yield


@asynccontextmanager
async def writer_scope(
    conn: psycopg.AsyncConnection, tenant: str = "halden"
) -> AsyncIterator[None]:
    """For ingestion only: may read and write every chunk of one
    tenant (Chapter 26)."""
    async with conn.transaction():
        await conn.execute(
            "SELECT set_config('halden.writer', 'on', true),"
            " set_config('halden.tenant', %s, true)",
            (tenant,),
        )
        yield


# [end:scope]
