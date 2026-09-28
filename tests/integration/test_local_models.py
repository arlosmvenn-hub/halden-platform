"""V4: behavior of a real open-weight embedding model.

Skipped automatically when sentence-transformers or the model
weights are unavailable (e.g. offline CI without a model cache).
"""

import json
from pathlib import Path

import numpy as np
import pytest

st = pytest.importorskip("sentence_transformers")

from halden.adapters.local_embedder import (  # noqa: E402
    SentenceTransformerEmbedder,
)
from halden.retrieval.brute_force import top_k_cosine  # noqa: E402

DATA = Path(__file__).parents[2] / "data" / "halden_snippets.jsonl"
PREFIX = "Represent this sentence for searching relevant passages: "


@pytest.fixture(scope="module")
def embedder() -> SentenceTransformerEmbedder:
    try:
        return SentenceTransformerEmbedder(
            "BAAI/bge-small-en-v1.5", query_prefix=PREFIX
        )
    except OSError as exc:  # weights not downloadable
        pytest.skip(f"model unavailable: {exc}")


async def test_vectors_are_normalized(
    embedder: SentenceTransformerEmbedder,
) -> None:
    mat = await embedder.embed_documents(["PX-200", "annual leave"])
    assert mat.shape == (2, embedder.dimensions) == (2, 384)
    assert np.allclose(np.linalg.norm(mat, axis=1), 1.0, atol=1e-5)


async def test_paraphrase_retrieval_beats_vocabulary_gap(
    embedder: SentenceTransformerEmbedder,
) -> None:
    rows = [json.loads(line) for line in DATA.open()]
    mat = await embedder.embed_documents([r["text"] for r in rows])
    q = await embedder.embed_query(
        "How many vacation days do staff in Germany get?"
    )
    best, _ = top_k_cosine(q, mat, k=1)[0]
    assert rows[best]["title"] == "Paid Time Off Policy §4 (Germany)"
