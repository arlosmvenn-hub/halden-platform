"""Randomized permission test: many random documents, groups, and
queries; no result may ever violate the ACL, through the retriever's
filter or through row-level security alone."""

import random

from psycopg_pool import AsyncConnectionPool

from halden.adapters.local_embedder import HashingEmbedder
from halden.ingestion.service import IngestionService
from halden.retrieval.pg_retriever import PgRetriever
from halden.store.documents import DocumentStore
from halden.store.scope import reader_scope

GROUPS = ["everyone", "legal", "support", "hr", "finance"]
VOCAB = [
    "pressure",
    "flow",
    "sensor",
    "contract",
    "ticket",
    "leave",
    "policy",
    "lead",
    "time",
    "price",
    "diaphragm",
    "calibration",
    "error",
    "return",
]


# [start:randomized]
async def test_no_query_ever_crosses_an_acl(
    pool: AsyncConnectionPool,
) -> None:
    rng = random.Random(20260927)  # fixed seed: reproducible
    emb = HashingEmbedder(dimensions=384)
    svc = IngestionService(
        DocumentStore(pool), emb, lambda t: len(t.split()), 30
    )
    acl_of: dict[str, set[str]] = {}
    for i in range(40):
        acl = set(rng.sample(GROUPS, rng.randint(1, 2)))
        body = " ".join(rng.choices(VOCAB, k=40))
        md = f"# Doc {i}\n## 1 Body\n{body}\n".encode()
        await svc.ingest(
            doc_id=f"d{i}",
            system="r",
            filename="d.md",
            data=md,
            acl=sorted(acl),
            source_version="1",
        )
        acl_of[f"d{i}"] = acl

    for _ in range(150):
        user = set(rng.sample(GROUPS, rng.randint(1, 3)))
        query = " ".join(rng.choices(VOCAB, k=3))
        async with pool.connection() as conn:
            async with reader_scope(conn, sorted(user)):
                r = PgRetriever(conn, emb)
                for hits in (
                    await r.hybrid(query, sorted(user), 20),
                    await r.dense(query, sorted(user), 20),
                ):
                    assert all(acl_of[h.doc_id] & user for h in hits)
                # RLS alone, with a query that has no ACL filter:
                rows = await (
                    await conn.execute(
                        "SELECT DISTINCT doc_id FROM chunks"
                    )
                ).fetchall()
            assert all(acl_of[d] & user for (d,) in rows)
            assert {d for (d,) in rows} == {
                d for d, a in acl_of.items() if a & user
            }


# [end:randomized]
