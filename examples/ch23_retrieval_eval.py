"""Chapter 23: golden versus synthetic questions, three pipelines.

Retrieval only, so it needs no API key: the database (Part IV
sources ingested) and the local models.
"""

import asyncio
from pathlib import Path

from halden.adapters.cross_encoder import CrossEncoderReranker
from halden.adapters.fake_llm import ScriptedLLM
from halden.adapters.local_embedder import SentenceTransformerEmbedder
from halden.config.settings import Settings
from halden.eval.dataset import EvalCase, load_cases, normalize
from halden.eval.metrics import mean
from halden.eval.report import summarize, table
from halden.eval.runner import index_texts, run_dataset
from halden.eval.synthetic import lexical_overlap
from halden.services.ask import AskService
from halden.store.db import create_pool

DATA = Path(__file__).parent.parent / "eval/datasets"
RERANKER = "cross-encoder/ms-marco-MiniLM-L6-v2"
CONFIGS = {
    "dense": dict(retrieval_mode="dense"),
    "hybrid": dict(retrieval_mode="hybrid"),
    "hybrid+rerank": dict(
        retrieval_mode="hybrid", reranker_model=RERANKER
    ),
}
COLUMNS = ["hit@5", "recall@5", "mrr", "ndcg@5"]


def overlap(cases: list[EvalCase], corpus: list[str]) -> float:
    """Mean lexical overlap between each question and the first
    chunk that holds its evidence."""
    scores = []
    for c in cases:
        ev = normalize(c.evidence[0])
        src = next(t for t in corpus if ev in normalize(t))
        scores.append(lexical_overlap(c.question, src))
    return mean(scores)


async def main() -> None:
    golden = load_cases(DATA / "golden.jsonl")
    synthetic = load_cases(DATA / "synthetic.jsonl")
    base = Settings()
    pool = create_pool(base)
    await pool.open()
    corpus = await index_texts(pool)
    answerable = [c for c in golden if c.answerable]
    print(
        f"golden: {len(answerable)} answerable of {len(golden)}, "
        f"word overlap with source {overlap(answerable, corpus):.2f}"
    )
    print(
        f"synthetic: {len(synthetic)}, "
        f"word overlap with source {overlap(synthetic, corpus):.2f}"
    )
    embedder = SentenceTransformerEmbedder(
        base.embedding_model, query_prefix=base.embedding_query_prefix
    )
    rows = {}
    for name, overrides in CONFIGS.items():
        settings = Settings(**overrides)  # type: ignore[arg-type]
        rr = (
            CrossEncoderReranker(RERANKER)
            if settings.reranker_model
            else None
        )
        svc = AskService(pool, embedder, ScriptedLLM([]), settings, rr)
        for label, cases in (("golden", golden), ("synth", synthetic)):
            results = await run_dataset(svc, cases)
            rows[f"{name} / {label}"] = summarize(results)
    await pool.close()
    print(table(rows, COLUMNS))


if __name__ == "__main__":
    asyncio.run(main())
