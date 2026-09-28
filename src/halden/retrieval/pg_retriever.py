"""Dense, lexical, and hybrid retrieval over the chunks table."""

from collections.abc import Sequence

import psycopg
from pgvector.psycopg import register_vector_async
from pydantic import BaseModel

from halden.ports.embedder import Embedder, Vector
from halden.retrieval.fusion import reciprocal_rank_fusion


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    heading_path: list[str]
    text: str
    score: float


# [start:sql]
# `acl_groups && %(groups)s` means "shares at least one group with
# the user". It sits inside the query, so unauthorized chunks are
# never candidates, never ranked, and never returned.
DENSE_SQL = """
SELECT c.chunk_id, c.doc_id, d.title, c.heading_path, c.text,
       1 - (c.embedding <=> %(qvec)s) AS score
FROM chunks c JOIN documents d USING (doc_id)
WHERE c.acl_groups && %(groups)s AND c.model_id = %(model)s
ORDER BY c.embedding <=> %(qvec)s
LIMIT %(k)s
"""

# Postgres full-text search ANDs every query term by default. For
# retrieval we usually want "match any term, rank by how many":
# rewrite ' & ' to ' | ' (phrases such as hx <-> -441 are kept).
LEXICAL_SQL = """
SELECT c.chunk_id, c.doc_id, d.title, c.heading_path, c.text,
       ts_rank_cd(c.tsv, q) AS score
FROM chunks c JOIN documents d USING (doc_id),
     CAST(CASE WHEN %(any_term)s
          THEN replace(websearch_to_tsquery('english', %(query)s)
                       ::text, ' & ', ' | ')
          ELSE websearch_to_tsquery('english', %(query)s)::text
          END AS tsquery) AS q
WHERE c.tsv @@ q AND c.acl_groups && %(groups)s
ORDER BY score DESC, c.chunk_id
LIMIT %(k)s
"""
# [end:sql]


class PgRetriever:
    def __init__(
        self, conn: psycopg.AsyncConnection, embedder: Embedder
    ) -> None:
        self._conn = conn
        self._embedder = embedder

    @classmethod
    async def connect(
        cls, dsn: str, embedder: Embedder
    ) -> "PgRetriever":
        conn = await psycopg.AsyncConnection.connect(
            dsn, autocommit=True
        )
        await register_vector_async(conn)
        return cls(conn, embedder)

    async def _run(
        self, sql: str, params: dict[str, object]
    ) -> list[RetrievedChunk]:
        cur = await self._conn.execute(sql, params)
        return [
            RetrievedChunk(
                chunk_id=r[0],
                doc_id=r[1],
                title=r[2],
                heading_path=r[3],
                text=r[4],
                score=float(r[5]),
            )
            for r in await cur.fetchall()
        ]

    async def dense(
        self,
        query: str,
        groups: Sequence[str],
        k: int = 20,
        *,
        qvec: Vector | None = None,
    ) -> list[RetrievedChunk]:
        """Nearest chunks to ``query``, or to a precomputed ``qvec``
        (used by HyDE, which embeds a passage instead)."""
        if qvec is None:
            qvec = await self._embedder.embed_query(query)
        return await self._run(
            DENSE_SQL,
            {
                "qvec": qvec,
                "groups": list(groups),
                "model": self._embedder.model_id,
                "k": k,
            },
        )

    async def lexical(
        self,
        query: str,
        groups: Sequence[str],
        k: int = 20,
        *,
        any_term: bool = True,
    ) -> list[RetrievedChunk]:
        return await self._run(
            LEXICAL_SQL,
            {
                "query": query,
                "groups": list(groups),
                "k": k,
                "any_term": any_term,
            },
        )

    # [start:hybrid]
    async def hybrid(
        self,
        query: str,
        groups: Sequence[str],
        k: int = 20,
        candidates: int = 40,
        *,
        qvec: Vector | None = None,
    ) -> list[RetrievedChunk]:
        """Fuse three complementary rankings with RRF:
        dense (meaning), all-terms lexical (exact identifiers and
        phrases, high precision), any-term lexical (recall)."""
        lists = [
            await self.dense(query, groups, candidates, qvec=qvec),
            await self.lexical(
                query, groups, candidates, any_term=False
            ),
            await self.lexical(query, groups, candidates),
        ]
        by_id = {c.chunk_id: c for ranked in lists for c in ranked}
        fused = reciprocal_rank_fusion(
            [[c.chunk_id for c in ranked] for ranked in lists]
        )
        return [
            by_id[cid].model_copy(update={"score": score})
            for cid, score in fused[:k]
        ]

    # [end:hybrid]

    async def aclose(self) -> None:
        await self._conn.close()
