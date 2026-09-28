"""Chapter 9: HNSW recall and latency versus exact search."""

import time

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from examples.ann_data import DSN, load_table, make_data


def exact_top_k(
    docs: np.ndarray, queries: np.ndarray, k: int
) -> list[set[int]]:
    scores = queries @ docs.T
    return [set(np.argsort(-row)[:k].tolist()) for row in scores]


def main() -> None:
    k = 10
    docs, _, queries = make_data()
    truth = exact_top_k(docs, queries, k)
    with psycopg.connect(DSN, autocommit=True) as conn:
        load_table(conn)
        register_vector(conn)
        conn.execute("DROP INDEX IF EXISTS ann_demo_hnsw")
        t0 = time.perf_counter()
        # [start:build]
        conn.execute(
            "CREATE INDEX ann_demo_hnsw ON ann_demo "
            "USING hnsw (embedding vector_cosine_ops) "
            "WITH (m = 16, ef_construction = 64)"
        )
        # [end:build]
        conn.commit()
        build_s = time.perf_counter() - t0
        size = conn.execute(
            "SELECT pg_size_pretty(pg_relation_size('ann_demo_hnsw'))"
        ).fetchone()
        print(
            f"{len(docs):,} vectors x {docs.shape[1]} dims; "
            f"HNSW build {build_s:.0f} s, index size {size[0]}\n"
        )
        print(
            f"{'ef_search':>9}  {'recall@10':>9}  {'p50 ms':>7}"
            f"  {'p95 ms':>7}"
        )
        # [start:query]
        sql = (
            "SELECT id FROM ann_demo ORDER BY embedding <=> %s LIMIT %s"
        )
        # [end:query]
        settings = [str(e) for e in (10, 20, 40, 80, 160, 320)]
        for ef in [*settings, "exact"]:
            with conn.transaction():
                if ef == "exact":
                    conn.execute("SET LOCAL enable_indexscan = off")
                else:
                    conn.execute(f"SET LOCAL hnsw.ef_search = {ef}")
                hits, lat = 0, []
                for q, true_ids in zip(queries, truth, strict=True):
                    t0 = time.perf_counter()
                    rows = conn.execute(
                        sql, (q, k), prepare=False
                    ).fetchall()
                    lat.append((time.perf_counter() - t0) * 1000)
                    hits += len(true_ids & {r[0] for r in rows})
            recall = hits / (k * len(queries))
            p50, p95 = np.percentile(lat, [50, 95])
            print(f"{ef:>9}  {recall:>9.3f}  {p50:>7.1f}  {p95:>7.1f}")


if __name__ == "__main__":
    main()
