"""Create or update a user and issue an API key (shown once).

uv run python scripts/create_user.py alice "Alice Ng" everyone
uv run python scripts/create_user.py lena "Lena Roth" everyone,legal
"""

import asyncio
import sys

from halden.config.settings import Settings
from halden.security.api_keys import create_key, create_user
from halden.store.db import create_pool


async def main(user_id: str, name: str, groups: list[str]) -> None:
    pool = create_pool(Settings())
    await pool.open()
    await create_user(pool, user_id, name, groups)
    key = await create_key(pool, user_id)
    await pool.close()
    print(f"user {user_id} groups={groups}")
    print(f"API key (store it now; it is not shown again):\n{key}")


if __name__ == "__main__":
    uid, name, groups = sys.argv[1], sys.argv[2], sys.argv[3]
    asyncio.run(main(uid, name, groups.split(",")))
