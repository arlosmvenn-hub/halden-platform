"""Persist documents and their chunks, one transaction per document."""

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager

from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool

from halden.ingestion.chunking import Chunk
from halden.models.document import Document
from halden.ports.embedder import Matrix
from halden.store.scope import writer_scope


class DocumentStore:
    def __init__(
        self, pool: AsyncConnectionPool, tenant: str = "halden"
    ) -> None:
        self._pool = pool
        self._tenant = tenant

    @asynccontextmanager
    async def _scope(self) -> AsyncIterator[AsyncConnection]:
        """A connection in a writer transaction for this tenant."""
        async with (
            self._pool.connection() as conn,
            writer_scope(conn, self._tenant),
        ):
            yield conn

    async def versions(self, system: str) -> dict[str, str]:
        """doc_id -> source_version for one source system."""
        async with self._scope() as conn:
            cur = await conn.execute(
                "SELECT doc_id, source_version FROM documents"
                " WHERE source_system = %s",
                (system,),
            )
            return {r[0]: r[1] or "" for r in await cur.fetchall()}

    async def content_hash(self, doc_id: str) -> str | None:
        async with self._scope() as conn:
            row = await (
                await conn.execute(
                    "SELECT content_hash FROM documents"
                    " WHERE doc_id = %s",
                    (doc_id,),
                )
            ).fetchone()
        return row[0] if row else None

    # [start:replace]
    async def replace(
        self,
        doc: Document,
        chunks: Sequence[Chunk],
        vectors: Matrix,
        model_id: str,
        source_version: str,
    ) -> None:
        """Swap a document's chunks atomically.

        Readers see either the old version or the new one, never a
        mix: the delete and inserts commit together or not at all.
        """
        async with self._scope() as conn:
            await conn.execute(
                "INSERT INTO documents (doc_id, title, source_system,"
                " source_uri, content_hash, acl_groups, language,"
                " source_version) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)"
                " ON CONFLICT (doc_id) DO UPDATE SET"
                " title = EXCLUDED.title,"
                " content_hash = EXCLUDED.content_hash,"
                " acl_groups = EXCLUDED.acl_groups,"
                " source_version = EXCLUDED.source_version,"
                " indexed_at = now()",
                (
                    doc.doc_id,
                    doc.title,
                    doc.source.system,
                    doc.source.uri,
                    doc.content_hash(),
                    doc.acl_groups,
                    doc.language,
                    source_version,
                ),
            )
            await conn.execute(
                "DELETE FROM chunks WHERE doc_id = %s", (doc.doc_id,)
            )
            async with conn.cursor() as cur:
                await cur.executemany(
                    "INSERT INTO chunks (chunk_id, doc_id, ordinal,"
                    " heading_path, kind, text, embed_text,"
                    " acl_groups, model_id, embedding)"
                    " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    [
                        (
                            c.chunk_id,
                            c.doc_id,
                            c.ordinal,
                            c.heading_path,
                            c.kind,
                            c.text,
                            c.embed_text,
                            doc.acl_groups,
                            model_id,
                            v,
                        )
                        for c, v in zip(chunks, vectors, strict=True)
                    ],
                )

    # [end:replace]

    async def touch_version(self, doc_id: str, version: str) -> None:
        async with self._scope() as conn:
            await conn.execute(
                "UPDATE documents SET source_version = %s"
                " WHERE doc_id = %s",
                (version, doc_id),
            )

    async def delete(self, doc_id: str) -> None:
        async with self._scope() as conn:
            await conn.execute(
                "DELETE FROM documents WHERE doc_id = %s", (doc_id,)
            )
