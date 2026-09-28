"""Parse text-based PDFs into the canonical Document model."""

import io
import re
from typing import Literal

from pypdf import PageObject, PdfReader

from halden.ingestion.normalize import (
    normalize_text,
    strip_repeated_lines,
)
from halden.models.document import Document, Section, SourceRef

# [start:parse]
NUMBERED = re.compile(r"^(\d+(?:\.\d+)*)\s+([A-Z][^.]{1,60})$")
ROW = " | "


def _is_heading(line: str, next_line: str) -> int:
    """Return the heading depth (1, 2, ...) or 0 if not a heading.

    PDFs carry no heading markup once text is extracted, so this
    is a heuristic: numbered lines like "4.1 Zero adjustment", or a
    short unpunctuated line directly above a table.
    """
    m = NUMBERED.match(line)
    if m:
        return m.group(1).count(".") + 1
    if (
        len(line) < 40
        and line[:1].isupper()
        and not line.endswith((".", ":"))
        and ROW in next_line
        and ROW not in line
    ):
        return 1
    return 0


# [start:margins]
def _body_lines(page: PageObject, margin: float) -> str:
    """Text outside the top and bottom margins, in reading order.

    Running headers and footers live in the margins. Filtering by
    position works even for a one-page document, where repeated-
    line detection has nothing to compare against. Text is kept in
    the order the extractor emits it: re-sorting fragments by x
    position scrambles lines that mix fonts or symbols.
    """
    height = float(page.mediabox.height)
    parts: list[str] = []

    def visit(
        text: str,
        cm: list[float],
        tm: list[float],
        font: object,
        size: object,
    ) -> None:
        y = tm[4] * cm[1] + tm[5] * cm[3] + cm[5]
        if text and margin < y < height - margin:
            parts.append(text)

    page.extract_text(visitor_text=visit)
    return "".join(parts)


# [end:margins]


def parse_pdf(
    data: bytes,
    *,
    doc_id: str,
    source: SourceRef,
    acl: list[str],
    margin: float = 40.0,
) -> Document:
    reader = PdfReader(io.BytesIO(data))
    pages = [_body_lines(p, margin) for p in reader.pages]
    pages = strip_repeated_lines(pages)  # boilerplate inside margins
    lines = [x.strip() for page in pages for x in page.splitlines()]
    lines = [x for x in lines if x]
    if not lines:
        raise ValueError(f"{doc_id}: no text layer (scanned PDF?)")

    title, lines = lines[0], lines[1:]
    sections: list[Section] = []
    path: list[str] = []
    prose: list[str] = []
    rows: list[str] = []

    def flush() -> None:
        if prose:
            sections.append(
                Section(
                    heading_path=list(path),
                    text=normalize_text(" ".join(prose)),
                )
            )
            prose.clear()
        if rows:
            header = rows[0]
            width = header.count("|") + 1
            table = [f"| {header} |", "|" + "---|" * width]
            table += [f"| {r} |" for r in rows[1:]]
            kind: Literal["table"] = "table"
            sections.append(
                Section(
                    heading_path=list(path),
                    text="\n".join(table),
                    kind=kind,
                )
            )
            rows.clear()

    for i, line in enumerate(lines):
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        depth = _is_heading(line, nxt)
        if depth:
            flush()
            del path[depth - 1 :]
            path.append(line)
        elif ROW in line:
            if prose:
                flush()
            rows.append(line)
        else:
            if rows:
                flush()
            prose.append(line)  # wrapped lines rejoin with spaces
    flush()
    return Document(
        doc_id=doc_id,
        title=title,
        source=source,
        sections=sections,
        acl_groups=acl,
    )


# [end:parse]
