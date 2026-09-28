"""Retrieval against a real Postgres + pgvector, in a throwaway
schema. Skipped when no database is reachable."""

import os
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import psycopg
import pytest
from pgvector.psycopg import register_vector_async

from halden.adapters.local_embedder import HashingEmbedder
from halden.retrieval.pg_retriever import PgRetriever

DSN = os.environ.get(
    "HALDEN_DSN", "postgresql://halden:halden@localhost/halden"
)
MIGRATION = Path(__file__).parents[2] / "migrations" / "001_chunks.sql"
ROWS = [  # doc_id, title, text, acl
    (
        "pub",
        "PX-200 Manual",
        "E-17 means sensor diaphragm fault",
        ["everyone"],
    ),
    (
        "fm",
        "FM-310 Manual",
        "E-17 means empty pipe detected",
        ["everyone"],
    ),
    (
        "sec",
        "Acme Supply Agreement",
        "Acme lead time is 45 days",
        ["legal"],
    ),
]


@pytest.fixture
async def conn() -> AsyncIterator[psycopg.AsyncConnection]:
    try:
        conn = await psycopg.AsyncConnection.connect(
            DSN, autocommit=True
        )
    except psycopg.OperationalError as exc:
        pytest.skip(f"no database: {exc}")
    schema = f"test_{uuid.uuid4().hex[:8]}"
    await conn.execute(f"CREATE SCHEMA {schema}")
    await conn.execute(f"SET search_path = {schema}, public")
    await conn.execute(MIGRATION.read_text())
    await register_vector_async(conn)
    emb = HashingEmbedder(dimensions=384)
    for doc_id, title, text, acl in ROWS:
        await conn.execute(
            "INSERT INTO documents VALUES"
            " (%s, %s, 't', 't', 'h', %s, 'en', now())",
            (doc_id, title, acl),
        )
        vec = (await emb.embed_documents([text]))[0]
        await conn.execute(
            "INSERT INTO chunks (chunk_id, doc_id, ordinal,"
            " heading_path, kind, text, embed_text, acl_groups,"
            " model_id, embedding) VALUES"
            " (%s, %s, 0, '{}', 'text', %s, %s, %s, %s, %s)",
            (
                f"{doc_id}:0",
                doc_id,
                text,
                f"{title}\n{text}",
                acl,
                emb.model_id,
                vec,
            ),
        )
    yield conn
    await conn.execute(f"DROP SCHEMA {schema} CASCADE")
    await conn.close()


class OtherModel(HashingEmbedder):
    @property
    def model_id(self) -> str:
        return "some-other-model"


@pytest.fixture
def retriever(conn: psycopg.AsyncConnection) -> PgRetriever:
    return PgRetriever(conn, HashingEmbedder(dimensions=384))


async def test_acl_filter_applies_to_every_method(
    retriever: PgRetriever,
) -> None:
    for method in (
        retriever.dense,
        retriever.lexical,
        retriever.hybrid,
    ):
        public = await method("Acme lead time", ["everyone"], 10)
        legal = await method("Acme lead time", ["legal"], 10)
        assert "sec:0" not in {c.chunk_id for c in public}
        assert "sec:0" in {c.chunk_id for c in legal}


async def test_strict_lexical_requires_every_term(
    retriever: PgRetriever,
) -> None:
    strict = await retriever.lexical(
        "E-17 empty pipe", ["everyone"], any_term=False
    )
    loose = await retriever.lexical("E-17 empty pipe", ["everyone"])
    assert [c.chunk_id for c in strict] == ["fm:0"]
    assert {c.chunk_id for c in loose} == {"fm:0", "pub:0"}


async def test_dense_ignores_vectors_from_other_models(
    conn: psycopg.AsyncConnection,
) -> None:
    other = PgRetriever(conn, OtherModel(dimensions=384))
    assert await other.dense("E-17", ["everyone"]) == []
