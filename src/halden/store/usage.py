"""Per-user daily token and cost accounting, for budgets."""

from psycopg_pool import AsyncConnectionPool


class BudgetExceeded(Exception):
    pass


# [start:budget]
class UsageLedger:
    """Counts each user's tokens per UTC day and enforces a daily
    token budget. The check happens *before* the model call, so an
    over-budget user costs nothing; the recording happens after,
    with the provider's actual counts."""

    def __init__(
        self, pool: AsyncConnectionPool, daily_tokens: int
    ) -> None:
        self._pool = pool
        self._limit = daily_tokens

    async def used_today(self, user_id: str) -> int:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT input_tokens + output_tokens FROM usage_daily"
                " WHERE user_id = %s"
                " AND day = (now() AT TIME ZONE 'utc')::date",
                (user_id,),
            )
            row = await cur.fetchone()
        return int(row[0]) if row else 0

    async def check(self, user_id: str) -> None:
        if await self.used_today(user_id) >= self._limit:
            raise BudgetExceeded(user_id)

    async def record(
        self,
        user_id: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
    ) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(
                "INSERT INTO usage_daily AS u (user_id, day,"
                " input_tokens, output_tokens, cost_usd)"
                " VALUES (%s, (now() AT TIME ZONE 'utc')::date,"
                " %s, %s, %s)"
                " ON CONFLICT (user_id, day) DO UPDATE SET"
                " input_tokens = u.input_tokens"
                " + EXCLUDED.input_tokens,"
                " output_tokens = u.output_tokens"
                " + EXCLUDED.output_tokens,"
                " cost_usd = u.cost_usd + EXCLUDED.cost_usd",
                (user_id, input_tokens, output_tokens, cost_usd),
            )


# [end:budget]
