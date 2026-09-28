"""Sync a folder into the index and process the resulting jobs.

    uv run python scripts/ingest.py [folder] [system]

Run it twice: the second run finds nothing to do.
"""

import asyncio
import logging
import sys
from pathlib import Path

from halden.adapters.local_embedder import SentenceTransformerEmbedder
from halden.config.settings import Settings
from halden.ingestion.chunking import hf_token_counter
from halden.ingestion.folder_connector import sync_folder
from halden.ingestion.service import IngestionService
from halden.ingestion.worker import make_handler, run_worker
from halden.store.db import create_pool, migrate
from halden.store.documents import DocumentStore
from halden.store.jobs import JobQueue

ROOT = Path(__file__).resolve().parent.parent


async def main(folder: Path, system: str) -> None:
    settings = Settings()
    pool = create_pool(settings)
    await pool.open()
    async with pool.connection() as conn:
        await conn.set_autocommit(True)
        await migrate(conn, settings.migrations_dir)
    store, queue = DocumentStore(pool), JobQueue(pool)
    plan = await sync_folder(folder, system, store, queue)
    print(
        f"plan: upsert={plan.upsert} delete={plan.delete} "
        f"unchanged={len(plan.unchanged)}"
    )
    if plan.upsert or plan.delete:
        embedder = SentenceTransformerEmbedder(settings.embedding_model)
        service = IngestionService(
            store,
            embedder,
            hf_token_counter(settings.embedding_model),
            settings.chunk_max_tokens,
        )
        n = await run_worker(
            queue,
            make_handler(service, store),
            stop=asyncio.Event(),
            drain=True,
        )
        print(f"processed {n} job(s)")
    await pool.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("httpx", "sentence_transformers", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    args = sys.argv[1:]
    folder = Path(args[0]) if args else ROOT / "data/sources/manuals"
    asyncio.run(main(folder, args[1] if len(args) > 1 else "manuals"))
