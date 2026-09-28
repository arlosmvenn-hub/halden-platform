"""Erasure: when content or a person must be forgotten, every copy
goes, not just the obvious row."""

from dataclasses import dataclass

from psycopg_pool import AsyncConnectionPool

from halden.store.scope import writer_scope


@dataclass
class Erased:
    documents: int = 0
    chunks: int = 0  # text *and* embeddings: same row
    cache_entries: int = 0
    conversations: int = 0
    usage_rows: int = 0
    api_keys: int = 0


# [start:forget-doc]
async def forget_document(
    pool: AsyncConnectionPool, doc_id: str, tenant: str = "halden"
) -> Erased:
    """Delete a document, its chunks and embeddings, and every
    cached answer. Cached answers may quote it, and the cache is
    keyed by question, not by source, so all entries go."""
    out = Erased()
    async with pool.connection() as conn, writer_scope(conn, tenant):
        cur = await conn.execute(
            "DELETE FROM chunks WHERE doc_id = %s", (doc_id,)
        )
        out.chunks = cur.rowcount
        cur = await conn.execute(
            "DELETE FROM documents WHERE doc_id = %s", (doc_id,)
        )
        out.documents = cur.rowcount
        cur = await conn.execute(
            "DELETE FROM answer_cache WHERE tenant_id = %s", (tenant,)
        )
        out.cache_entries = cur.rowcount
    return out


# [end:forget-doc]


# [start:forget-user]
async def forget_user(
    pool: AsyncConnectionPool, user_id: str
) -> Erased:
    """Delete a person's conversations, usage records, and keys.

    Cached answers hold question text but not who asked: shared
    entries serve everyone with the same groups. Which entries came
    from this person can't be known, so the tenant's whole answer
    cache goes: it refills on its own, and a missed copy cannot
    be recalled later.
    The user row goes last; ON DELETE CASCADE covers messages and
    group links."""
    out = Erased()
    async with pool.connection() as conn, conn.transaction():
        cur = await conn.execute(
            "DELETE FROM answer_cache WHERE tenant_id ="
            " (SELECT tenant_id FROM users WHERE user_id = %s)",
            (user_id,),
        )
        out.cache_entries = cur.rowcount
        cur = await conn.execute(
            "DELETE FROM conversations WHERE user_id = %s", (user_id,)
        )
        out.conversations = cur.rowcount
        cur = await conn.execute(
            "DELETE FROM usage_daily WHERE user_id = %s", (user_id,)
        )
        out.usage_rows = cur.rowcount
        cur = await conn.execute(
            "DELETE FROM api_keys WHERE user_id = %s", (user_id,)
        )
        out.api_keys = cur.rowcount
        await conn.execute(
            "DELETE FROM users WHERE user_id = %s", (user_id,)
        )
    return out


# [end:forget-user]
