"""Conversation memory, always scoped to its owner."""

import uuid

from psycopg_pool import AsyncConnectionPool

from halden.retrieval.query import Turn


# [start:store]
class ConversationStore:
    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._pool = pool

    async def create(self, user_id: str) -> uuid.UUID:
        async with self._pool.connection() as conn:
            row = await (
                await conn.execute(
                    "INSERT INTO conversations (user_id) VALUES (%s)"
                    " RETURNING conversation_id",
                    (user_id,),
                )
            ).fetchone()
        assert row is not None
        return uuid.UUID(str(row[0]))

    async def history(
        self, conversation_id: uuid.UUID, user_id: str, limit: int = 6
    ) -> list[Turn] | None:
        """Last ``limit`` turns, or None if the conversation does not
        exist *or belongs to someone else*: callers cannot tell the
        two apart, so IDs cannot be probed."""
        async with self._pool.connection() as conn:
            owner = await (
                await conn.execute(
                    "SELECT 1 FROM conversations"
                    " WHERE conversation_id = %s AND user_id = %s",
                    (conversation_id, user_id),
                )
            ).fetchone()
            if owner is None:
                return None
            rows = await (
                await conn.execute(
                    "SELECT role, text FROM (SELECT role, text, id"
                    " FROM messages WHERE conversation_id = %s"
                    " ORDER BY id DESC LIMIT %s) t ORDER BY id",
                    (conversation_id, limit),
                )
            ).fetchall()
        return [Turn(role=r[0], text=r[1]) for r in rows]

    async def append(
        self, conversation_id: uuid.UUID, question: str, answer: str
    ) -> None:
        async with self._pool.connection() as conn, conn.transaction():
            await conn.execute(
                "INSERT INTO messages (conversation_id, role, text)"
                " VALUES (%s, 'user', %s), (%s, 'assistant', %s)",
                (conversation_id, question, conversation_id, answer),
            )
            await conn.execute(
                "UPDATE conversations SET updated_at = now()"
                " WHERE conversation_id = %s",
                (conversation_id,),
            )

    async def purge_older_than(self, days: int) -> int:
        """Retention: run daily. Memory is data the user gave us;
        keep it only as long as it is useful."""
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "DELETE FROM conversations WHERE updated_at <"
                " now() - make_interval(days => %s)",
                (days,),
            )
            return cur.rowcount


# [end:store]
