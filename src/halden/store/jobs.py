"""A durable job queue on PostgreSQL."""

import json
from dataclasses import dataclass
from typing import Any

from psycopg_pool import AsyncConnectionPool


@dataclass(frozen=True)
class Job:
    id: int
    kind: str
    payload: dict[str, Any]
    attempts: int
    max_attempts: int


# [start:queue]
CLAIM_SQL = """
UPDATE jobs SET status = 'running', locked_at = now(),
       attempts = attempts + 1, updated_at = now()
WHERE id = (
    SELECT id FROM jobs
    WHERE status = 'queued' AND run_after <= now()
    ORDER BY run_after, id
    FOR UPDATE SKIP LOCKED
    LIMIT 1
)
RETURNING id, kind, payload, attempts, max_attempts
"""


class JobQueue:
    """Workers claim jobs with FOR UPDATE SKIP LOCKED: each worker
    locks one row and skips rows other workers hold, so many
    workers can poll concurrently without claiming the same job."""

    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._pool = pool

    async def enqueue(
        self, kind: str, payload: dict[str, Any], dedupe_key: str
    ) -> bool:
        """Returns False if the same work is already pending."""
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "INSERT INTO jobs (kind, payload, dedupe_key)"
                " VALUES (%s, %s, %s)"
                " ON CONFLICT (kind, dedupe_key)"
                " WHERE status IN ('queued', 'running') DO NOTHING",
                (kind, json.dumps(payload), dedupe_key),
            )
            return cur.rowcount == 1

    async def claim(self) -> Job | None:
        async with self._pool.connection() as conn:
            row = await (await conn.execute(CLAIM_SQL)).fetchone()
        if row is None:
            return None
        return Job(row[0], row[1], row[2], row[3], row[4])

    async def complete(self, job: Job) -> None:
        await self._set(job.id, "status = 'done'", ())

    async def fail(self, job: Job, error: str, retryable: bool) -> None:
        """Retry with exponential backoff (2, 4, 8 ... seconds,
        capped at 10 minutes); after max_attempts, or for a
        non-retryable error, park the job as 'dead' for a human."""
        if retryable and job.attempts < job.max_attempts:
            delay = min(2**job.attempts, 600)
            await self._set(
                job.id,
                "status = 'queued', last_error = %s,"
                " run_after = now() + make_interval(secs => %s)",
                (error[:2000], delay),
            )
        else:
            await self._set(
                job.id,
                "status = 'dead', last_error = %s",
                (error[:2000],),
            )

    async def requeue_stale(self, older_than_s: int = 900) -> int:
        """Recover jobs whose worker died mid-run."""
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "UPDATE jobs SET status = 'queued', updated_at = now()"
                " WHERE status = 'running' AND locked_at <"
                " now() - make_interval(secs => %s)",
                (older_than_s,),
            )
            return cur.rowcount

    async def _set(
        self, job_id: int, assignments: str, params: tuple[Any, ...]
    ) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(
                f"UPDATE jobs SET {assignments}, updated_at = now()"
                " WHERE id = %s",
                (*params, job_id),
            )


# [end:queue]
