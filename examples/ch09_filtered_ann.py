"""Chapter 9: what a selective filter does to HNSW results."""

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from examples.ann_data import DSN, load_table, make_data

# [start:filtered-query]
SQL = (
    "SELECT id FROM ann_demo WHERE grp = 'legal' "
    "ORDER BY embedding <=> %s LIMIT 10"
)
# [end:filtered-query]


def main() -> None:
    docs, legal, queries = make_data()
    legal_ids = np.flatnonzero(legal)
    # Exact answer: nearest neighbors *among permitted rows*.
    scores = queries @ docs[legal_ids].T
    truth = [set(legal_ids[np.argsort(-s)[:10]]) for s in scores]
    print(
        f"{legal.sum()} of {len(docs):,} rows are 'legal' "
        f"({legal.mean():.1%})\n"
    )
    with psycopg.connect(DSN, autocommit=True) as conn:
        load_table(conn)
        register_vector(conn)
        plan = conn.execute("EXPLAIN " + SQL, (queries[0],)).fetchall()
        uses_hnsw = any("ann_demo_hnsw" in r[0] for r in plan)
        print(f"planner uses the HNSW index: {uses_hnsw}\n")
        print(f"{'mode':<32}{'rows returned':>14}{'recall':>8}")
        # Planner settings below must affect every query, so we
        # disable psycopg's automatic server-side prepared
        # statements (prepare=False): a cached plan would ignore
        # enable_indexscan changed after it was prepared.
        # [start:modes]
        modes = {
            "HNSW, ef_search=40": [],
            "HNSW, iterative scan": [
                "SET LOCAL hnsw.iterative_scan = relaxed_order"
            ],
            "exact (index disabled)": [
                "SET LOCAL enable_indexscan = off"
            ],
        }
        # [end:modes]
        for name, stmts in modes.items():
            returned, hits = [], 0
            with conn.transaction():
                for s in stmts:
                    conn.execute(s)
                for q, t in zip(queries, truth, strict=True):
                    ids = {
                        r[0]
                        for r in conn.execute(SQL, (q,), prepare=False)
                    }
                    returned.append(len(ids))
                    hits += len(ids & t)
            recall = hits / (10 * len(queries))
            print(f"{name:<32}{np.mean(returned):>14.1f}{recall:>8.3f}")


if __name__ == "__main__":
    main()
