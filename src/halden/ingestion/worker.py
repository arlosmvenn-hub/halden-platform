"""The ingestion worker: claim a job, run it, record the outcome."""

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path

from halden.ingestion.service import IngestionService, PermanentError
from halden.store.documents import DocumentStore
from halden.store.jobs import Job, JobQueue

log = logging.getLogger(__name__)
Handler = Callable[[Job], Awaitable[None]]


def make_handler(
    service: IngestionService, store: DocumentStore
) -> Handler:
    async def handle(job: Job) -> None:
        p = job.payload
        if job.kind == "ingest_file":
            path = Path(p["path"])
            try:
                data = await asyncio.to_thread(path.read_bytes)
            except FileNotFoundError as exc:
                raise PermanentError(f"missing file {path}") from exc
            result = await service.ingest(
                doc_id=p["doc_id"],
                system=p["system"],
                filename=path.name,
                data=data,
                acl=p["acl"],
                source_version=p["version"],
            )
            log.info("ingested %s: %s", p["doc_id"], result)
        elif job.kind == "delete_document":
            await store.delete(p["doc_id"])
            log.info("deleted %s", p["doc_id"])
        else:
            raise PermanentError(f"unknown job kind {job.kind}")

    return handle


# [start:worker]
async def run_worker(
    queue: JobQueue,
    handle: Handler,
    *,
    stop: asyncio.Event,
    drain: bool = False,
    idle_sleep_s: float = 1.0,
) -> int:
    """Process jobs until ``stop`` is set (or, with ``drain``, until
    the queue is empty). Returns the number of jobs processed."""
    processed = 0
    while not stop.is_set():
        job = await queue.claim()
        if job is None:
            if drain:
                break
            # Sleep, but wake immediately when shutdown is requested.
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), idle_sleep_s)
            continue
        try:
            await handle(job)
        except PermanentError as exc:
            log.warning("job %s dead: %s", job.id, exc)
            await queue.fail(job, str(exc), retryable=False)
        except Exception as exc:  # transient until proven otherwise
            log.exception("job %s failed", job.id)
            await queue.fail(job, repr(exc), retryable=True)
        else:
            await queue.complete(job)
        processed += 1
    return processed


# [end:worker]
