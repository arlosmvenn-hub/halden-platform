"""Load the Halden sample corpus into the "demo" schema for the
Part II-III examples.

Manuals are parsed and chunked (structure-aware, with context
headers); the 40 short snippets become one-chunk documents. Part IV
replaces this script with the real ingestion pipeline.

    uv run python scripts/load_demo_index.py
"""

import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector_async

from halden.adapters.local_embedder import SentenceTransformerEmbedder
from halden.ingestion.chunking import chunk_document, hf_token_counter
from halden.ingestion.markdown_parser import load_manual
from halden.models.document import Document, Section, SourceRef

ROOT = Path(__file__).resolve().parent.parent
BASE_DSN = os.environ.get(
    "HALDEN_DSN", "postgresql://halden:halden@localhost/halden"
)
MODEL = "BAAI/bge-small-en-v1.5"
PREFIX = "Represent this sentence for searching relevant passages: "
RESTRICTED = {"contract": ["legal"], "hr": ["hr", "everyone"]}


def snippet_docs() -> list[Document]:
    docs = []
    path = ROOT / "data" / "halden_snippets.jsonl"
    for line in path.open():
        row = json.loads(line)
        prefix = row["doc_id"].split("-")[0]
        docs.append(
            Document(
                doc_id=row["id"],
                title=row["title"],
                source=SourceRef(
                    system="snippets",
                    uri=row["doc_id"],
                    fetched_at=datetime.now(UTC),
                ),
                sections=[Section(heading_path=[], text=row["text"])],
                acl_groups=RESTRICTED.get(prefix, ["everyone"]),
            )
        )
    return docs


async def main() -> None:
    count = hf_token_counter(MODEL)
    emb = SentenceTransformerEmbedder(MODEL, query_prefix=PREFIX)
    docs = [
        load_manual(p, ["everyone"])
        for p in sorted((ROOT / "data" / "manuals").glob("*.md"))
    ] + snippet_docs()
    async with await psycopg.AsyncConnection.connect(BASE_DSN) as conn:
        await conn.execute("CREATE SCHEMA IF NOT EXISTS demo")
        await conn.execute("SET search_path = demo, public")
        await register_vector_async(conn)
        sql = (ROOT / "migrations" / "001_chunks.sql").read_text()
        await conn.execute(sql)
        await conn.execute("TRUNCATE documents CASCADE")
        n = 0
        for doc in docs:
            chunks = chunk_document(doc, max_tokens=120, count=count)
            vecs = await emb.embed_documents(
                [c.embed_text for c in chunks]
            )
            await conn.execute(
                "INSERT INTO documents (doc_id, title, source_system,"
                " source_uri, content_hash, acl_groups)"
                " VALUES (%s, %s, %s, %s, %s, %s)",
                (
                    doc.doc_id,
                    doc.title,
                    doc.source.system,
                    doc.source.uri,
                    doc.content_hash(),
                    doc.acl_groups,
                ),
            )
            async with conn.cursor() as cur:
                await cur.executemany(
                    "INSERT INTO chunks (chunk_id, doc_id, ordinal,"
                    " heading_path, kind, text, embed_text, acl_groups,"
                    " model_id, embedding) VALUES"
                    " (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
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
                            emb.model_id,
                            v,
                        )
                        for c, v in zip(chunks, vecs, strict=True)
                    ],
                )
            n += len(chunks)
        await conn.commit()
    print(f"loaded {len(docs)} documents, {n} chunks")


if __name__ == "__main__":
    asyncio.run(main())
