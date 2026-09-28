import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest
from psycopg_pool import AsyncConnectionPool

from halden.adapters.local_embedder import HashingEmbedder
from halden.ingestion import folder_connector
from halden.ingestion.folder_connector import (
    DeletionCapExceeded,
    sync_folder,
)
from halden.ingestion.markdown_parser import load_manual
from halden.ingestion.pdf_parser import parse_pdf
from halden.ingestion.service import (
    IngestionService,
    IngestResult,
    PermanentError,
)
from halden.ingestion.worker import make_handler, run_worker
from halden.models.document import SourceRef
from halden.store.documents import DocumentStore
from halden.store.jobs import Job, JobQueue
from halden.store.scope import writer_scope

DATA = Path(__file__).parents[2] / "data"


def words(text: str) -> int:
    return len(text.split())


def service(pool: AsyncConnectionPool) -> IngestionService:
    return IngestionService(
        DocumentStore(pool), HashingEmbedder(dimensions=384), words, 60
    )


async def count(pool: AsyncConnectionPool, sql: str) -> int:
    # Chunks are under row-level security: read as a writer.
    async with pool.connection() as conn, writer_scope(conn):
        row = await (await conn.execute(sql)).fetchone()
    assert row is not None
    return int(row[0])


# ---------- queue ----------


async def test_concurrent_claims_never_share_a_job(
    pool: AsyncConnectionPool,
) -> None:
    q = JobQueue(pool)
    for i in range(30):
        await q.enqueue("k", {"i": i}, dedupe_key=str(i))

    async def claimer() -> list[int]:
        got = []
        while (job := await q.claim()) is not None:
            got.append(job.id)
        return got

    results = await asyncio.gather(*(claimer() for _ in range(6)))
    claimed = [j for r in results for j in r]
    assert len(claimed) == 30 == len(set(claimed))


async def test_pending_jobs_are_deduplicated(
    pool: AsyncConnectionPool,
) -> None:
    q = JobQueue(pool)
    assert await q.enqueue("k", {}, dedupe_key="doc@v1")
    assert not await q.enqueue("k", {}, dedupe_key="doc@v1")
    job = await q.claim()
    assert job is not None
    await q.complete(job)
    assert await q.enqueue(
        "k", {}, dedupe_key="doc@v1"
    )  # done: allowed


async def test_retry_with_backoff_then_dead(
    pool: AsyncConnectionPool,
) -> None:
    q = JobQueue(pool)
    await q.enqueue("k", {}, dedupe_key="x")
    async with pool.connection() as conn:
        await conn.execute("UPDATE jobs SET max_attempts = 2")
    job = await q.claim()
    assert job is not None
    await q.fail(job, "timeout", retryable=True)
    assert await q.claim() is None  # backing off, not yet due
    async with pool.connection() as conn:
        await conn.execute("UPDATE jobs SET run_after = now()")
    job = await q.claim()
    assert job is not None and job.attempts == 2
    await q.fail(job, "timeout again", retryable=True)
    assert (
        await count(
            pool, "SELECT count(*) FROM jobs WHERE status = 'dead'"
        )
        == 1
    )


async def test_permanent_errors_are_not_retried(
    pool: AsyncConnectionPool,
) -> None:
    q = JobQueue(pool)
    await q.enqueue("bad", {}, dedupe_key="b")

    async def handle(job: Job) -> None:
        raise PermanentError("unsupported file type")

    n = await run_worker(q, handle, stop=asyncio.Event(), drain=True)
    assert n == 1
    assert await count(pool, "SELECT attempts FROM jobs") == 1
    assert (
        await count(
            pool, "SELECT count(*) FROM jobs WHERE status = 'dead'"
        )
        == 1
    )


# ---------- ingestion ----------


async def test_ingest_is_idempotent_and_replaces_atomically(
    pool: AsyncConnectionPool,
) -> None:
    svc = service(pool)
    md = (DATA / "manuals" / "fm310_manual.md").read_bytes()
    args = dict(
        doc_id="m:fm", system="m", filename="fm.md", acl=["everyone"]
    )
    assert await svc.ingest(data=md, source_version="v1", **args) == (
        IngestResult.INDEXED
    )
    first = await count(pool, "SELECT count(*) FROM chunks")
    assert await svc.ingest(data=md, source_version="v2", **args) == (
        IngestResult.UNCHANGED
    )
    shorter = md.split(b"## 5 Maintenance")[0]
    assert (
        await svc.ingest(data=shorter, source_version="v3", **args)
        == IngestResult.INDEXED
    )
    after = await count(pool, "SELECT count(*) FROM chunks")
    assert 0 < after < first
    assert (
        await count(
            pool,
            "SELECT count(*) FROM chunks WHERE text LIKE '%gasket%'",
        )
        == 0
    )  # old chunks gone


async def test_folder_sync_end_to_end(
    pool: AsyncConnectionPool, tmp_path: Path
) -> None:
    store, q = DocumentStore(pool), JobQueue(pool)
    handler = make_handler(service(pool), store)
    for name in ("a", "b"):
        (tmp_path / f"{name}.md").write_text(
            f"# {name}\n## 1 S\nText {name}."
        )
    await sync_folder(tmp_path, "t", store, q)
    await run_worker(q, handler, stop=asyncio.Event(), drain=True)
    assert sorted(await store.versions("t")) == ["t:a", "t:b"]

    (tmp_path / "a.md").write_text("# a\n## 1 S\nChanged.")
    (tmp_path / "b.md").unlink()
    plan = await sync_folder(tmp_path, "t", store, q)
    assert plan.upsert == ["t:a"] and plan.delete == ["t:b"]
    await run_worker(q, handler, stop=asyncio.Event(), drain=True)
    assert sorted(await store.versions("t")) == ["t:a"]
    assert (
        await count(
            pool,
            "SELECT count(*) FROM chunks WHERE text LIKE '%Changed%'",
        )
        == 1
    )


async def test_pipeline_version_bump_requeues_unchanged_files(
    pool: AsyncConnectionPool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression: a parser fix changes no file, so without a
    # pipeline version in the marker nothing was re-processed.
    store, q = DocumentStore(pool), JobQueue(pool)
    (tmp_path / "a.md").write_text("# a\n## 1 S\nText.")
    await sync_folder(tmp_path, "t", store, q)
    await run_worker(
        q,
        make_handler(service(pool), store),
        stop=asyncio.Event(),
        drain=True,
    )
    plan = await sync_folder(tmp_path, "t", store, q)
    assert plan.upsert == []
    monkeypatch.setattr(folder_connector, "PIPELINE_VERSION", "next")
    plan = await sync_folder(tmp_path, "t", store, q)
    assert plan.upsert == ["t:a"]


async def test_mass_deletion_is_refused(
    pool: AsyncConnectionPool, tmp_path: Path
) -> None:
    store, q = DocumentStore(pool), JobQueue(pool)
    for i in range(5):
        (tmp_path / f"d{i}.md").write_text(f"# d{i}\n## 1 S\nx {i}.")
    await sync_folder(tmp_path, "t", store, q)
    await run_worker(
        q,
        make_handler(service(pool), store),
        stop=asyncio.Event(),
        drain=True,
    )
    for i in range(1, 5):
        (tmp_path / f"d{i}.md").unlink()
    with pytest.raises(DeletionCapExceeded):
        await sync_folder(tmp_path, "t", store, q)


# ---------- PDF parsing ----------


def test_pdf_parser_matches_markdown_structure() -> None:
    ref = SourceRef(system="t", uri="t", fetched_at=datetime.now(UTC))
    pdf = parse_pdf(
        (DATA / "sources/manuals/px200_manual.pdf").read_bytes(),
        doc_id="px",
        source=ref,
        acl=[],
    )
    md = load_manual(DATA / "manuals/px200_manual.md", [])
    assert pdf.title == md.title
    assert [s.heading_path for s in pdf.sections] == [
        s.heading_path for s in md.sections
    ]
    assert [s.kind for s in pdf.sections] == [
        s.kind for s in md.sections
    ]
    assert not any("Page " in s.text for s in pdf.sections)


def test_one_page_pdf_loses_its_running_header() -> None:
    # Regression: repeated-line detection needs two or more pages,
    # so the one-page HX-4410 guide once kept its header as title.
    ref = SourceRef(system="t", uri="t", fetched_at=datetime.now(UTC))
    doc = parse_pdf(
        (DATA / "sources/manuals/hx4410_manual.pdf").read_bytes(),
        doc_id="hx",
        source=ref,
        acl=[],
    )
    assert doc.title.startswith("HX-4410 Differential")
    assert all(
        "HI-MAN" not in s.text or not s.heading_path
        for s in doc.sections
    )
    assert not any("Page 1" in s.text for s in doc.sections)
    table = next(s.text for s in doc.sections if s.kind == "table")
    assert "| Range | −50 to 250 °C |" in table  # fragment order kept
