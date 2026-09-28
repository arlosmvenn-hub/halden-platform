"""Parse, chunk, embed, and store one document."""

import asyncio
from datetime import UTC, datetime
from enum import StrEnum

from halden.ingestion.chunking import TokenCounter, chunk_document
from halden.ingestion.markdown_parser import parse_markdown
from halden.ingestion.pdf_parser import parse_pdf
from halden.models.document import Document, SourceRef
from halden.ports.embedder import Embedder
from halden.store.documents import DocumentStore


class PermanentError(Exception):
    """A job that will fail the same way every time (bad input)."""


class IngestResult(StrEnum):
    INDEXED = "indexed"
    UNCHANGED = "unchanged"


# [start:service]
class IngestionService:
    def __init__(
        self,
        store: DocumentStore,
        embedder: Embedder,
        count: TokenCounter,
        max_tokens: int,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._count = count
        self._max_tokens = max_tokens

    @staticmethod
    def parse(
        filename: str,
        data: bytes,
        doc_id: str,
        source: SourceRef,
        acl: list[str],
    ) -> Document:
        if filename.endswith(".md"):
            return parse_markdown(
                data.decode("utf-8"),
                doc_id=doc_id,
                source=source,
                acl=acl,
            )
        if filename.endswith(".pdf"):
            try:
                return parse_pdf(
                    data, doc_id=doc_id, source=source, acl=acl
                )
            except ValueError as exc:  # e.g. no text layer
                raise PermanentError(str(exc)) from exc
        raise PermanentError(f"unsupported file type: {filename}")

    async def ingest(
        self,
        *,
        doc_id: str,
        system: str,
        filename: str,
        data: bytes,
        acl: list[str],
        source_version: str,
    ) -> IngestResult:
        source = SourceRef(
            system=system,
            uri=filename,
            version=source_version,
            fetched_at=datetime.now(UTC),
        )
        # Parsing and tokenizing are CPU-bound: run them off the
        # event loop so one large PDF cannot stall other work.
        doc = await asyncio.to_thread(
            self.parse, filename, data, doc_id, source, acl
        )
        if await self._store.content_hash(doc_id) == doc.content_hash():
            # The file changed (new source_version) but nothing we
            # index did, e.g. a re-saved PDF: skip re-embedding.
            await self._store.touch_version(doc_id, source_version)
            return IngestResult.UNCHANGED
        chunks = await asyncio.to_thread(
            chunk_document,
            doc,
            max_tokens=self._max_tokens,
            count=self._count,
        )
        vectors = await self._embedder.embed_documents(
            [c.embed_text for c in chunks]
        )
        await self._store.replace(
            doc,
            chunks,
            vectors,
            self._embedder.model_id,
            source_version,
        )
        return IngestResult.INDEXED


# [end:service]
