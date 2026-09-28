"""Chapter 17: the ingestion pipeline's behavior, step by step.

Runs in a throwaway schema on a temporary copy of the PDF manuals:
first sync, an idempotent re-run, an edited file, a deleted file.
"""

import asyncio
import logging
import shutil
import sys
import tempfile
import uuid
from pathlib import Path
from urllib.parse import quote

import psycopg

from halden.adapters.local_embedder import SentenceTransformerEmbedder
from halden.config.settings import Settings
from halden.ingestion.chunking import hf_token_counter
from halden.ingestion.folder_connector import sync_folder
from halden.ingestion.service import IngestionService
from halden.ingestion.worker import make_handler, run_worker
from halden.store.db import create_pool, migrate
from halden.store.documents import DocumentStore
from halden.store.jobs import JobQueue
from halden.store.scope import writer_scope

ROOT = Path(__file__).resolve().parent.parent


async def step(
    label: str,
    folder: Path,
    store: DocumentStore,
    queue: JobQueue,
    handle: object,
) -> None:
    plan = await sync_folder(folder, "manuals", store, queue)
    n = await run_worker(
        queue,
        handle,
        stop=asyncio.Event(),  # type: ignore[arg-type]
        drain=True,
    )
    print(
        f"{label}\n  upsert={[d.split(':')[1] for d in plan.upsert]}"
        f" delete={[d.split(':')[1] for d in plan.delete]}"
        f" unchanged={len(plan.unchanged)} jobs={n}"
    )


async def main() -> None:
    base = Settings()
    schema = f"demo17_{uuid.uuid4().hex[:6]}"
    async with await psycopg.AsyncConnection.connect(
        base.dsn, autocommit=True
    ) as admin:
        await admin.execute(f"CREATE SCHEMA {schema}")
    opts = quote(f"-csearch_path={schema},public", safe="")
    settings = Settings(dsn=f"{base.dsn}?options={opts}")
    pool = create_pool(settings)
    await pool.open()
    async with pool.connection() as conn:
        await conn.set_autocommit(True)
        await migrate(conn, settings.migrations_dir)
    store, queue = DocumentStore(pool), JobQueue(pool)
    service = IngestionService(
        store,
        SentenceTransformerEmbedder(settings.embedding_model),
        hf_token_counter(settings.embedding_model),
        120,
    )
    handle = make_handler(service, store)
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        for f in (ROOT / "data/sources/manuals").iterdir():
            shutil.copy(f, folder / f.name)
        await step("1. first sync", folder, store, queue, handle)
        await step(
            "2. run again, nothing changed",
            folder,
            store,
            queue,
            handle,
        )
        (folder / "fm310_manual.pdf").write_bytes(
            (folder / "fm310_manual.pdf").read_bytes() + b"\n%touched\n"
        )
        await step(
            "3. FM-310 file re-saved (bytes differ, text same)",
            folder,
            store,
            queue,
            handle,
        )
        (folder / "hx4410_manual.pdf").unlink()
        await step(
            "4. HX-4410 file deleted", folder, store, queue, handle
        )
    async with pool.connection() as conn, writer_scope(conn):
        rows = await (
            await conn.execute(
                "SELECT doc_id, count(*) FROM chunks"
                " GROUP BY 1 ORDER BY 1"
            )
        ).fetchall()
        jobs = await (
            await conn.execute(
                "SELECT status, count(*) FROM jobs GROUP BY 1"
            )
        ).fetchall()
    print("chunks per document:", {d.split(":")[1]: n for d, n in rows})
    print("jobs:", dict(jobs))
    await pool.close()
    async with await psycopg.AsyncConnection.connect(
        base.dsn, autocommit=True
    ) as admin:
        await admin.execute(f"DROP SCHEMA {schema} CASCADE")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.ERROR,
        format="    %(message)s",
        stream=sys.stdout,
    )
    logging.getLogger("halden.ingestion.worker").setLevel(logging.INFO)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    asyncio.run(main())
