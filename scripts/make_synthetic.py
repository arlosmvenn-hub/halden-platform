"""Generate eval/datasets/synthetic.jsonl from the index.

    ANTHROPIC_API_KEY=... HALDEN_MODEL_SMALL=... \
        uv run python scripts/make_synthetic.py

The output is committed, so later evaluations are repeatable and
cost nothing. Regenerate only when the corpus changes a lot.
"""

import asyncio
import os
from pathlib import Path

from halden.adapters.anthropic_llm import AnthropicLLM
from halden.config.settings import Settings
from halden.eval.dataset import save_cases
from halden.eval.synthetic import synthesize
from halden.store.db import create_pool
from halden.store.scope import writer_scope

OUT = Path(__file__).resolve().parent.parent / "eval/datasets"


async def main() -> None:
    key, name = (
        os.environ.get("ANTHROPIC_API_KEY"),
        os.environ.get("HALDEN_MODEL_SMALL"),
    )
    if not key or not name:
        raise SystemExit("Set ANTHROPIC_API_KEY and HALDEN_MODEL_SMALL")
    pool = create_pool(Settings())
    await pool.open()
    async with pool.connection() as conn, writer_scope(conn):
        cur = await conn.execute(
            "SELECT chunk_id, text, acl_groups FROM chunks "
            "WHERE length(text) >= 60 AND text NOT LIKE 'Document %' "
            "ORDER BY chunk_id"
        )
        rows = await cur.fetchall()
    await pool.close()
    llm = AnthropicLLM(key, name)
    gate = asyncio.Semaphore(4)

    async def one(row: tuple[str, str, list[str]]):  # type: ignore[no-untyped-def]
        async with gate:
            return await synthesize(llm, *row)

    made = await asyncio.gather(*(one(r) for r in rows))
    cases = [c for c in made if c is not None]
    save_cases(OUT / "synthetic.jsonl", cases)
    print(
        f"{len(rows)} chunks -> {len(cases)} cases "
        f"({len(rows) - len(cases)} rejected)"
    )


if __name__ == "__main__":
    asyncio.run(main())
