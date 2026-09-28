"""Chapter 27: an API configured with the wrong embedding model
for its index refuses to start."""

import asyncio

from halden.config.settings import Settings
from halden.services.container import check_index_model
from halden.store.db import create_pool


async def main() -> None:
    settings = Settings(index_schema="green")  # default: bge-small
    pool = create_pool(settings)
    await pool.open()
    try:
        await check_index_model(pool, settings)
        print("index and model match")
    except RuntimeError as exc:
        print(f"startup refused: {exc}")
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
