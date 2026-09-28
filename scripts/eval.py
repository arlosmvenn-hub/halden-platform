"""Evaluate the question-answering pipeline on a dataset.

    uv run python scripts/eval.py                      # retrieval
    uv run python scripts/eval.py --mode full          # + answers
    uv run python scripts/eval.py --baseline eval/baseline.json

Retrieval mode needs only the database and local models, so CI
runs it on every change. Full mode also generates answers and has
a judge grade them: it needs ANTHROPIC_API_KEY, HALDEN_MODEL, and
optionally HALDEN_JUDGE_MODEL (defaults to HALDEN_MODEL).

Exit status 1 if evidence labels are broken or the gate fails.
"""

import argparse
import asyncio
import json
import os
import sys
import textwrap
from pathlib import Path

from halden.adapters.anthropic_llm import AnthropicLLM
from halden.adapters.cross_encoder import CrossEncoderReranker
from halden.adapters.fake_llm import ScriptedLLM
from halden.adapters.local_embedder import SentenceTransformerEmbedder
from halden.config.settings import Settings
from halden.eval.dataset import load_cases, missing_evidence
from halden.eval.report import (
    gate,
    save_results,
    summarize,
    table,
)
from halden.eval.runner import index_texts, run_dataset
from halden.ports.llm import LLMClient
from halden.services.ask import AskService
from halden.store.db import create_pool

ROOT = Path(__file__).resolve().parent.parent
RERANKER = "cross-encoder/ms-marco-MiniLM-L6-v2"
RETRIEVAL = ["hit@5", "recall@5", "mrr", "ndcg@5", "context_recall"]
ANSWERS = [
    "abstention_acc",
    "citation_support",
    "faithfulness",
    "correct",
]


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "eval/datasets/golden.jsonl",
    )
    p.add_argument("--mode", choices=["retrieval", "full"])
    p.add_argument("--retrieval", choices=["dense", "hybrid"])
    p.add_argument("--rerank", action="store_true")
    p.add_argument("--out", type=Path)
    p.add_argument("--baseline", type=Path)
    p.add_argument("--write-baseline", action="store_true")
    p.add_argument("--max-drop", type=float, default=0.02)
    p.add_argument("--floors", type=Path, help="JSON: metric -> min")
    p.set_defaults(mode="retrieval", retrieval="hybrid")
    return p.parse_args()


def model(var: str, fallback: str | None = None) -> LLMClient:
    key = os.environ.get("ANTHROPIC_API_KEY")
    name = os.environ.get(var) or (
        os.environ.get(fallback) if fallback else None
    )
    if not key or not name:
        raise SystemExit(f"--mode full needs ANTHROPIC_API_KEY, {var}")
    return AnthropicLLM(key, name)


async def main(args: argparse.Namespace) -> int:
    settings = Settings(
        retrieval_mode=args.retrieval,
        reranker_model=RERANKER if args.rerank else None,
    )
    cases = load_cases(args.dataset)
    pool = create_pool(settings)
    await pool.open()
    # [start:lint-first]
    broken = missing_evidence(cases, await index_texts(pool))
    if broken:
        for case_id, gone in broken.items():
            print(f"BROKEN LABEL {case_id}: {gone} not in index")
        return 1
    # [end:lint-first]
    full = args.mode == "full"
    llm = model("HALDEN_MODEL") if full else ScriptedLLM([])
    judge = (
        model("HALDEN_JUDGE_MODEL", "HALDEN_MODEL") if full else None
    )
    embedder = SentenceTransformerEmbedder(
        settings.embedding_model,
        query_prefix=settings.embedding_query_prefix,
    )
    rr = CrossEncoderReranker(RERANKER) if args.rerank else None
    svc = AskService(pool, embedder, llm, settings, rr)
    results = await run_dataset(svc, cases, generate=full, judge=judge)
    await pool.close()

    summary = summarize(results)
    name = f"{args.retrieval}{' + rerank' if args.rerank else ''}"
    print(f"{args.dataset.name}: {len(cases)} cases, {name}")
    print(table({name: summary}, RETRIEVAL))
    if full:
        print(table({name: summary}, ANSWERS))
    if args.out:
        save_results(args.out, results, summary)
    if args.write_baseline and args.baseline:
        args.baseline.write_text(json.dumps(summary, indent=1) + "\n")
        print(f"baseline written: {args.baseline.name}")
    elif args.baseline or args.floors:
        failures = gate(
            summary,
            json.loads(args.baseline.read_text())
            if args.baseline
            else {},
            max_drop=args.max_drop,
            floors=(
                json.loads(args.floors.read_text())
                if args.floors
                else None
            ),
        )
        for f in failures:
            print("GATE FAILED", f)
        if failures:
            return 1
        print("gate passed")
    for r in results:
        problems = [
            m
            for m, bad in [
                ("retrieval", r.recall is not None and r.recall < 1),
                ("context", (r.context_recall or 1) < 1),
                ("abstention", r.abstention_ok is False),
                ("unfaithful", (r.faithfulness or 1) < 1),
                ("answer", r.correctness in ("partial", "incorrect")),
            ]
            if bad
        ]
        if problems:
            print(f"  {r.id}: {', '.join(problems)}")
        for claim in r.unsupported:  # the judge's reasons are in --out
            print(f"      unsupported: {textwrap.shorten(claim, 56)}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(parse())))
