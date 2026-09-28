"""Shared setup for examples that call a live model."""

import sys

from examples.demo_db import DEMO_DSN as DSN
from halden.adapters.local_embedder import SentenceTransformerEmbedder
from halden.config.llm_factory import llm_from_env
from halden.ports.llm import LLMClient

__all__ = ["DSN", "PREFIX", "embedder", "rank_of", "require_llm"]

PREFIX = "Represent this sentence for searching relevant passages: "


def require_llm(role: str = "small") -> LLMClient:
    llm = llm_from_env("small" if role == "small" else "answer")
    if llm is None:
        print("This example calls a live model. Set ANTHROPIC_API_KEY,")
        print("HALDEN_MODEL and HALDEN_MODEL_SMALL (see .env.example).")
        sys.exit(0)
    return llm


def embedder() -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(
        "BAAI/bge-small-en-v1.5", query_prefix=PREFIX
    )


def rank_of(chunks: list, answer: str) -> int | None:  # type: ignore[type-arg]
    return next(
        (i for i, c in enumerate(chunks, 1) if answer in c.text), None
    )
