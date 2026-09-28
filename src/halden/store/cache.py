"""An exact-match answer cache that respects permissions and
invalidates itself when the index changes."""

import hashlib
import json
import unicodedata
from collections.abc import Sequence
from datetime import timedelta

from psycopg_pool import AsyncConnectionPool

from halden.services.ask import Answer


def normalize(question: str) -> str:
    return " ".join(
        unicodedata.normalize("NFKC", question).casefold().split()
    )


# [start:key]
def cache_key(
    tenant: str,
    question: str,
    groups: Sequence[str],
    generation: int,
    fingerprint: str,
) -> str:
    """Everything that can change the answer is in the key:

    - the tenant (Chapter 26): two customers can have the same
      group names and ask the same question;
    - the question, normalized (Unicode form, case, spacing);
    - the asker's groups, sorted: an answer built from what the
      legal team may see is never served to anyone else;
    - the index generation: any re-index moves it, so answers
      built from old chunks become unreachable;
    - a fingerprint of the pipeline: model, prompt version,
      retrieval settings.
    """
    material = json.dumps(
        [
            tenant,
            normalize(question),
            sorted(set(groups)),
            generation,
            fingerprint,
        ]
    )
    return hashlib.sha256(material.encode()).hexdigest()


# [end:key]


class AnswerCache:
    def __init__(
        self, pool: AsyncConnectionPool, ttl: timedelta
    ) -> None:
        self._pool = pool
        self._ttl = ttl

    async def generation(self) -> int:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT generation FROM index_state"
            )
            row = await cur.fetchone()
        return int(row[0]) if row else 0

    # [start:get]
    async def get(
        self, key: str, groups: Sequence[str], tenant: str
    ) -> Answer | None:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT answer FROM answer_cache WHERE key = %s"
                " AND groups = %s AND tenant_id = %s"
                " AND created_at > now() - %s",
                (key, sorted(set(groups)), tenant, self._ttl),
            )
            row = await cur.fetchone()
        if row is None:
            return None
        return Answer.model_validate(row[0]).model_copy(
            update={"cached": True, "timings_ms": {}}
        )

    # [end:get]

    async def put(
        self,
        key: str,
        groups: Sequence[str],
        tenant: str,
        answer: Answer,
    ) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(
                "INSERT INTO answer_cache"
                " (key, groups, tenant_id, answer)"
                " VALUES (%s, %s, %s, %s) ON CONFLICT (key) DO UPDATE"
                " SET answer = EXCLUDED.answer, created_at = now()",
                (
                    key,
                    sorted(set(groups)),
                    tenant,
                    answer.model_dump_json(),
                ),
            )

    async def purge(self, *, everything: bool = False) -> int:
        """Delete expired entries, or all of them. Entries keyed to
        an old generation are unreachable but still hold text, so
        purge on a schedule, and fully when content must be
        forgotten (Chapter 26)."""
        max_age = timedelta(0) if everything else self._ttl
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "DELETE FROM answer_cache"
                " WHERE created_at <= now() - %s",
                (max_age,),
            )
            return cur.rowcount
