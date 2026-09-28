"""Apply database migrations: uv run python scripts/migrate.py"""

import asyncio

import psycopg

from halden.config.settings import Settings
from halden.store.db import migrate


async def main() -> None:
    settings = Settings()
    async with await psycopg.AsyncConnection.connect(
        settings.dsn, autocommit=True
    ) as conn:
        applied = await migrate(conn, settings.migrations_dir)
    print("applied:", ", ".join(applied) or "nothing (up to date)")


if __name__ == "__main__":
    asyncio.run(main())
