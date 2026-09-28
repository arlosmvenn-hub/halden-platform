from halden.generation.context import build_context
from halden.generation.grounded import (
    ABSTAIN,
    build_messages,
    check_citations,
)
from halden.retrieval.pg_retriever import RetrievedChunk


def chunk(cid: str, doc: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid,
        doc_id=doc,
        title=f"Doc {doc}",
        heading_path=["1 Intro"],
        text=text,
        score=1.0,
    )


def words(text: str) -> int:
    return len(text.split())


def test_context_dedupes_caps_per_doc_and_respects_budget() -> None:
    chunks = [
        chunk("a:0", "a", "E-17 means diaphragm fault."),
        chunk("b:0", "b", "e-17 MEANS diaphragm fault"),  # duplicate
        chunk("a:1", "a", "Return the unit."),
        chunk("a:2", "a", "Third from doc a."),
        chunk("c:0", "c", "word " * 50),  # too big for the budget
        chunk("d:0", "d", "Small final chunk."),
    ]
    ctx = build_context(
        chunks, budget_tokens=90, count=words, max_per_doc=2
    )
    assert [s.chunk_id for s in ctx.sources] == ["a:0", "a:1", "d:0"]
    assert ctx.dropped == ["b:0", "c:0", "a:2"]
    assert [s.label for s in ctx.sources] == [1, 2, 3]
    assert ctx.tokens <= 90


def test_prompt_wraps_sources_and_labels_them() -> None:
    ctx = build_context(
        [chunk("a:0", "a", "Fact.")], budget_tokens=100, count=words
    )
    text = build_messages("Q?", ctx)[0].content[0]
    assert "<sources>\n[1] Doc a > 1 Intro\nFact.\n</sources>" in (
        getattr(text, "text", "")
    )


def test_citation_check_cases() -> None:
    ctx = build_context(
        [chunk("a:0", "a", "x"), chunk("b:0", "b", "y")],
        budget_tokens=100,
        count=words,
    )
    good = check_citations("A fact [1]. Another [2].", ctx)
    assert good.ok and good.cited == [1, 2]
    bad = check_citations("A fact [1]. Invented [7].", ctx)
    assert not bad.ok and bad.unknown == [7]
    uncited = check_citations("A fact [1]. Unsupported claim.", ctx)
    assert not uncited.ok
    assert uncited.uncited_sentences == ["Unsupported claim."]
    assert check_citations(ABSTAIN, ctx).ok
    assert not check_citations(f"{ABSTAIN} But maybe [1].", ctx).ok
    assert not check_citations("No citations at all.", ctx).ok


# Regression cases taken from real model answers recorded in
# Chapter 15, which the strict check rejected.
REAL_ANSWER = (
    "E-17 has different meanings on each device:\n\n"
    "- **PX-200**: E-17 indicates a **sensor diaphragm fault**. "
    "Return the transmitter for repair [1][2].\n\n"
    "- **FM-310**: E-17 indicates an **empty pipe detected** at the "
    "sensor [5]."
)
HONEST_GAP = (
    "The HX-4410 replaced the HX-441 [1]. The sources do not contain "
    "accuracy specifications for the HX-4410."
)


def test_lenient_check_accepts_lead_ins_bullets_and_gaps() -> None:
    ctx = build_context(
        [chunk(f"{i}:0", str(i), f"text {i}") for i in range(5)],
        budget_tokens=500,
        count=words,
    )
    assert not check_citations(REAL_ANSWER, ctx).ok
    assert check_citations(REAL_ANSWER, ctx, lenient=True).ok
    assert not check_citations(HONEST_GAP, ctx).ok
    assert check_citations(HONEST_GAP, ctx, lenient=True).ok


def test_lenient_check_still_catches_real_problems() -> None:
    ctx = build_context(
        [chunk("a:0", "a", "x")], budget_tokens=100, count=words
    )
    invented = check_citations("A claim [7].", ctx, lenient=True)
    assert not invented.ok and invented.unknown == [7]
    unsupported = check_citations(
        "A fact [1].\n- Another claim with no source.",
        ctx,
        lenient=True,
    )
    assert not unsupported.ok


def test_capped_chunks_backfill_leftover_budget() -> None:
    # One document holds every relevant chunk: the cap must not
    # throw the answer away when budget remains (Chapter 18).
    chunks = [chunk(f"hx:{i}", "hx", f"fact {i}") for i in range(5)]
    ctx = build_context(
        chunks, budget_tokens=500, count=words, max_per_doc=3
    )
    assert [s.chunk_id for s in ctx.sources] == [
        f"hx:{i}" for i in range(5)
    ]


def test_backfill_never_adds_a_duplicate() -> None:
    chunks = [
        chunk("a:0", "a", "x one"),
        chunk("a:1", "a", "same"),
        chunk("b:0", "b", "SAME"),
    ]
    ctx = build_context(
        chunks, budget_tokens=500, count=words, max_per_doc=1
    )
    texts = [s.text.lower() for s in ctx.sources]
    assert texts.count("same") == 1
