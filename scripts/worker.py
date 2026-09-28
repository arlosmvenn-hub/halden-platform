"""The long-running ingestion worker for a deployment.

    HALDEN_SOURCES=manuals=/data/manuals,policies=/data/policies \
        python scripts/worker.py

Every HALDEN_SYNC_INTERVAL_S seconds it lists each source folder
and enqueues changes; in between, it processes jobs. SIGTERM (what
Kubernetes and Docker send on shutdown) finishes the current job,
then exits, so a deploy never leaves a document half-written.
"""

import asyncio
import contextlib
import logging
import os
import signal
from pathlib import Path

from halden.adapters.local_embedder import SentenceTransformerEmbedder
from halden.config.settings import Settings
from halden.ingestion.chunking import hf_token_counter
from halden.ingestion.folder_connector import sync_folder
from halden.ingestion.service import IngestionService
from halden.ingestion.worker import make_handler, run_worker
from halden.store.db import create_pool
from halden.store.documents import DocumentStore
from halden.store.jobs import JobQueue

log = logging.getLogger("halden.worker")


def sources() -> dict[str, Path]:
    raw = os.environ.get("HALDEN_SOURCES", "")
    items = (i.strip() for i in raw.split(","))
    pairs = (i.split("=", 1) for i in items if i)
    return {name: Path(path) for name, path in pairs}


# [start:loop]
async def main() -> None:
    settings = Settings()
    interval = float(os.environ.get("HALDEN_SYNC_INTERVAL_S", "300"))
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)

    pool = create_pool(settings)
    await pool.open(wait=True, timeout=30)
    store, queue = DocumentStore(pool), JobQueue(pool)
    service = IngestionService(
        store,
        SentenceTransformerEmbedder(settings.embedding_model),
        hf_token_counter(settings.embedding_model),
        settings.chunk_max_tokens,
    )
    handler = make_handler(service, store)
    while not stop.is_set():
        await queue.requeue_stale()  # jobs of a worker that died
        for system, folder in sources().items():
            try:
                plan = await sync_folder(folder, system, store, queue)
                log.info("%s: %d to upsert", system, len(plan.upsert))
            except Exception:  # one bad source must not stop others
                log.exception("sync failed for %s", system)
        # Process jobs until the next sync is due, or shutdown.
        due = asyncio.Event()
        work = asyncio.create_task(run_worker(queue, handler, stop=due))
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), interval)
        due.set()
        await work  # the current job finishes first
    await pool.close()
    log.info("worker stopped cleanly")


# [end:loop]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(main())
