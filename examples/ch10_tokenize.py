"""Chapter 10: how Postgres full-text search sees identifiers."""

import os

import psycopg

DSN = os.environ.get(
    "HALDEN_DSN", "postgresql://halden:halden@localhost/halden"
)
with psycopg.connect(DSN) as conn:
    text = "Error E-17 on the HX-4410; PX-200 accuracy ±0.075%"
    doc = conn.execute(
        "SELECT to_tsvector('english', %s)", (text,)
    ).fetchone()
    print("document:", doc[0] if doc else None)
    for q in ("HX-441 accuracy", "HX-4410 accuracy"):
        row = conn.execute(
            "SELECT websearch_to_tsquery('english', %s),"
            " to_tsvector('english', %s)"
            " @@ websearch_to_tsquery('english', %s)",
            (q, text, q),
        ).fetchone()
        assert row is not None
        print(f"query {q!r}: {row[0]}  matches: {row[1]}")
