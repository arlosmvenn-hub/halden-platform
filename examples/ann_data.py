"""Synthetic vectors for the Chapter 9 index experiments.

Real embeddings cluster by topic, so we generate clustered data:
points scattered around a few hundred random "topic" centers. The
data is deterministic (fixed seed) so results are reproducible.
"""

import os

import numpy as np
import psycopg
from numpy.typing import NDArray
from pgvector.psycopg import register_vector

DSN = os.environ.get(
    "HALDEN_DSN", "postgresql://halden:halden@localhost/halden"
)
N, DIMS, CENTERS, QUERIES = 50_000, 384, 300, 200
LEGAL_SHARE = 0.01  # rows only the legal team may see


def unit(x: NDArray[np.float32]) -> NDArray[np.float32]:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def make_data() -> tuple[
    NDArray[np.float32], NDArray[np.bool_], NDArray[np.float32]
]:
    rng = np.random.default_rng(42)
    centers = rng.standard_normal((CENTERS, DIMS), dtype=np.float32)
    assign = rng.integers(0, CENTERS, N)
    noise = rng.standard_normal((N, DIMS), dtype=np.float32)
    docs = unit(centers[assign] + 0.9 * noise)
    legal = rng.random(N) < LEGAL_SHARE
    qa = rng.integers(0, CENTERS, QUERIES)
    qn = rng.standard_normal((QUERIES, DIMS), dtype=np.float32)
    queries = unit(centers[qa] + 0.9 * qn)
    return docs, legal, queries


def load_table(conn: psycopg.Connection[tuple[object, ...]]) -> None:
    """Create and fill ann_demo once; later runs reuse it."""
    register_vector(conn)
    exists = conn.execute(
        "SELECT to_regclass('ann_demo') IS NOT NULL"
    ).fetchone()
    if exists and exists[0]:
        return
    docs, legal, _ = make_data()
    conn.execute(
        f"CREATE TABLE ann_demo (id int PRIMARY KEY, "
        f"grp text NOT NULL, embedding vector({DIMS}) NOT NULL)"
    )
    with conn.cursor().copy(
        "COPY ann_demo (id, grp, embedding) FROM STDIN "
        "WITH (FORMAT BINARY)"
    ) as copy:
        copy.set_types(["int4", "text", "vector"])
        for i in range(N):
            grp = "legal" if legal[i] else "everyone"
            copy.write_row((i, grp, docs[i]))
    conn.commit()
