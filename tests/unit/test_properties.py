"""Property-based tests: invariants checked on generated inputs."""

import re

from hypothesis import given, settings
from hypothesis import strategies as st

from halden.generation.context import build_context
from halden.generation.grounded import check_citations
from halden.ingestion.chunking import split_recursive
from halden.retrieval.fusion import reciprocal_rank_fusion
from halden.retrieval.pg_retriever import RetrievedChunk

WORDS = st.text(alphabet="abcdefghij", min_size=1, max_size=8)


def words(text: str) -> int:
    return len(text.split())


# [start:chunker-property]
@given(
    paragraphs=st.lists(
        st.lists(WORDS, min_size=1, max_size=30).map(" ".join),
        min_size=1,
        max_size=8,
    ),
    limit=st.integers(min_value=1, max_value=40),
)
def test_chunker_respects_limit_and_loses_nothing(
    paragraphs: list[str], limit: int
) -> None:
    text = "\n\n".join(paragraphs)
    pieces = split_recursive(text, limit, words)
    assert all(words(p) <= limit for p in pieces)
    assert " ".join(pieces).split() == text.split()


# [end:chunker-property]

IDS = st.lists(st.sampled_from("abcdefgh"), unique=True, max_size=8)


@given(rankings=st.lists(IDS, min_size=1, max_size=4))
def test_rrf_keeps_every_item_and_orders_by_score(
    rankings: list[list[str]],
) -> None:
    fused = reciprocal_rank_fusion(rankings)
    assert {i for i, _ in fused} == {i for r in rankings for i in r}
    scores = [s for _, s in fused]
    assert scores == sorted(scores, reverse=True)


@given(rankings=st.lists(IDS.filter(bool), min_size=1, max_size=4))
def test_rrf_winner_of_every_list_wins(
    rankings: list[list[str]],
) -> None:
    rankings = [["z", *r] for r in rankings]  # z first everywhere
    assert reciprocal_rank_fusion(rankings)[0][0] == "z"


CHUNKS = st.lists(
    st.tuples(
        st.sampled_from("abc"), st.lists(WORDS, min_size=1, max_size=20)
    ),
    max_size=15,
)


@settings(max_examples=200)
@given(
    raw=CHUNKS,
    budget=st.integers(min_value=0, max_value=300),
    cap=st.integers(min_value=1, max_value=4),
)
def test_context_invariants(
    raw: list[tuple[str, list[str]]], budget: int, cap: int
) -> None:
    chunks = [
        RetrievedChunk(
            chunk_id=f"{doc}:{i}",
            doc_id=doc,
            title=doc,
            heading_path=[],
            text=" ".join(ws),
            score=1.0,
        )
        for i, (doc, ws) in enumerate(raw)
    ]
    ctx = build_context(
        chunks, budget_tokens=budget, count=words, max_per_doc=cap
    )
    assert ctx.tokens <= budget
    assert [s.label for s in ctx.sources] == list(
        range(1, len(ctx.sources) + 1)
    )
    texts = [re.sub(r"\W+", " ", s.text.lower()) for s in ctx.sources]
    assert len(texts) == len(set(texts))  # no duplicates
    order = [c.chunk_id for c in chunks]
    ids = [s.chunk_id for s in ctx.sources]
    assert ids == sorted(ids, key=order.index)  # rank order kept
    kept = set(ids) | set(ctx.dropped)
    assert kept == set(order)  # every chunk accounted for


@given(
    labels=st.lists(st.integers(min_value=1, max_value=20), max_size=6)
)
def test_unknown_citations_are_exactly_the_invalid_ones(
    labels: list[int],
) -> None:
    chunks = [
        RetrievedChunk(
            chunk_id=f"d:{i}",
            doc_id=f"d{i}",
            title="t",
            heading_path=[],
            text=f"text {i}",
            score=1.0,
        )
        for i in range(5)
    ]
    ctx = build_context(chunks, budget_tokens=1000, count=words)
    answer = " ".join(f"Claim [{n}]." for n in labels)
    check = check_citations(answer, ctx, lenient=True)
    assert set(check.unknown) == {n for n in labels if n > 5}
