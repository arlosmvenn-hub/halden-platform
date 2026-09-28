from datetime import UTC, datetime
from pathlib import Path

import pytest

from halden.ingestion.chunking import (
    chunk_document,
    split_recursive,
    split_table,
)
from halden.ingestion.markdown_parser import load_manual, parse_markdown
from halden.ingestion.normalize import (
    normalize_text,
    strip_repeated_lines,
)
from halden.ingestion.sync import plan_sync
from halden.models.document import SourceRef

MANUALS = Path(__file__).parents[2] / "data" / "manuals"


def words(text: str) -> int:
    return len(text.split())


def ref() -> SourceRef:
    return SourceRef(system="t", uri="t", fetched_at=datetime.now(UTC))


# ---------- normalization ----------


def test_normalize_folds_ligatures_hyphens_and_spaces() -> None:
    raw = "ﬁlter  cali-\nbration step­\r\n\r\n\r\n\nend"
    assert normalize_text(raw) == "filter calibration step\n\nend"


def test_strip_repeated_lines_ignores_page_numbers() -> None:
    bodies = ["Mounting", "Wiring", "Calibration", "Safety"]
    pages = [
        f"Halden Instruments - Confidential - Page {i}\n{body}"
        for i, body in enumerate(bodies, start=1)
    ]
    assert strip_repeated_lines(pages) == bodies


def test_strip_repeated_lines_masks_numbers_too_eagerly() -> None:
    # Documented limitation: lines differing only in digits look
    # identical, so numbered body lines can be mistaken for a
    # footer. Chapter 7 discusses why this heuristic needs review.
    pages = [f"Header\nStep {i}" for i in range(1, 5)]
    assert strip_repeated_lines(pages) == ["", "", "", ""]


# ---------- parsing ----------


def test_parser_builds_heading_paths_and_keeps_tables() -> None:
    doc = load_manual(MANUALS / "px200_manual.md", ["everyone"])
    assert doc.title.startswith("PX-200 Pressure Transmitter")
    errors = [
        s
        for s in doc.sections
        if s.heading_path[-1:] == ["7.1 Error codes"]
    ]
    assert len(errors) == 1 and errors[0].kind == "table"
    assert errors[0].heading_path == [
        "7 Troubleshooting",
        "7.1 Error codes",
    ]
    assert "E-17 | Sensor diaphragm fault" in errors[0].text


def test_parser_separates_prose_from_following_table() -> None:
    md = "# T\n## A\nIntro line.\n| a | b |\n|---|---|\n| 1 | 2 |\n"
    doc = parse_markdown(md, doc_id="d", source=ref(), acl=[])
    assert [s.kind for s in doc.sections] == ["text", "table"]


def test_content_hash_changes_with_text_and_acl() -> None:
    md = "# T\n## A\nBody.\n"
    a = parse_markdown(md, doc_id="d", source=ref(), acl=["x"])
    b = parse_markdown(md, doc_id="d", source=ref(), acl=["x"])
    c = parse_markdown(md, doc_id="d", source=ref(), acl=["y"])
    d = parse_markdown(md + "More.\n", doc_id="d", source=ref(), acl=[])
    assert a.content_hash() == b.content_hash()  # fetched_at ignored
    assert a.content_hash() != c.content_hash()
    assert a.content_hash() != d.content_hash()


# ---------- sync planning ----------


def test_plan_sync_upserts_changed_and_new_deletes_missing() -> None:
    plan = plan_sync(
        indexed={"a": "1", "b": "2", "c": "3"},
        source={"a": "1", "b": "9", "d": "4"},
    )
    assert plan.upsert == ["b", "d"]
    assert plan.delete == ["c"]
    assert plan.unchanged == ["a"]


# ---------- chunking ----------


@pytest.mark.parametrize("max_tokens", [5, 12, 40])
def test_split_recursive_respects_limit_and_loses_nothing(
    max_tokens: int,
) -> None:
    text = (MANUALS / "px200_manual.md").read_text()
    pieces = split_recursive(text, max_tokens, words)
    assert all(words(p) <= max_tokens for p in pieces)
    assert " ".join(pieces).split() == text.split()


def test_split_table_repeats_header() -> None:
    table = "| a | b |\n|---|---|\n" + "\n".join(
        f"| {i} | row {i} |" for i in range(10)
    )
    pieces = split_table(table, max_tokens=20, count=words)
    assert len(pieces) > 1
    assert all(p.startswith("| a | b |\n|---|---|") for p in pieces)


def test_chunks_carry_context_header_only_in_embed_text() -> None:
    doc = load_manual(MANUALS / "fm310_manual.md", ["everyone"])
    chunks = chunk_document(doc, max_tokens=60, count=words)
    e17 = next(c for c in chunks if "Empty pipe" in c.text)
    assert "FM-310" not in e17.text.split("\n")[0]
    assert e17.embed_text.startswith("FM-310 Electromagnetic")
    assert "6 Troubleshooting > 6.1 Error codes" in e17.embed_text
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
