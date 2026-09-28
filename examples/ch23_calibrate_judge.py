"""Chapter 23: how much can each judge model be trusted?

Builds faithful and unfaithful answers with known labels from the
golden set, has each judge grade them, and compares. Live models.
"""

import asyncio
import os
from pathlib import Path

from halden.adapters.anthropic_llm import AnthropicLLM
from halden.config.settings import Settings
from halden.eval.calibration import (
    build_items,
    cohen_kappa,
    judge_items,
)
from halden.eval.dataset import load_cases
from halden.eval.runner import index_sources
from halden.store.db import create_pool

GOLDEN = Path(__file__).parent.parent / "eval/datasets/golden.jsonl"
# [start:judges]
JUDGES = [
    "claude-haiku-4-5-20251001",
    "claude-sonnet-5",
    "claude-opus-5-5",
]
# [end:judges]


async def main() -> None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        print("Set ANTHROPIC_API_KEY to run this example.")
        return
    pool = create_pool(Settings())
    await pool.open()
    items = build_items(load_cases(GOLDEN), await index_sources(pool))
    await pool.close()
    truth = [x.faithful for x in items]
    kinds = {
        v: sum(x.variant == v for x in items)
        for v in ("original", "number", "invented")
    }
    print(f"{len(items)} labeled items: {kinds}")
    print(
        f"{'judge':<28}{'agree':>7}{'kappa':>7}"
        f"{'missed':>8}{'false alarm':>13}"
    )
    for name in JUDGES:
        said = await judge_items(AnthropicLLM(key, name), items)
        agree = sum(s == t for s, t in zip(said, truth, strict=True))
        missed = [  # unfaithful answers the judge let through
            f"{x.case_id}/{x.variant}"
            for x, s in zip(items, said, strict=True)
            if s and not x.faithful
        ]
        alarms = [  # faithful answers the judge rejected
            x.case_id
            for x, s in zip(items, said, strict=True)
            if not s and x.faithful
        ]
        print(
            f"{name:<28}{agree / len(items):>7.2f}"
            f"{cohen_kappa(said, truth):>7.2f}"
            f"{len(missed):>8}{len(alarms):>13}"
        )
        if missed:
            print(f"{'':<4}missed: {', '.join(missed)}")
        if alarms:
            print(f"{'':<4}false alarms: {', '.join(alarms)}")


if __name__ == "__main__":
    asyncio.run(main())
