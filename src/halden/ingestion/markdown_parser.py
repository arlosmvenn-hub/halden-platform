"""Parse Markdown into structured sections with heading paths."""

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from halden.ingestion.normalize import normalize_text
from halden.models.document import Document, Section, SourceRef

HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


# [start:parse]
def parse_markdown(
    text: str, *, doc_id: str, source: SourceRef, acl: list[str]
) -> Document:
    """Split on headings; keep tables and code blocks intact.

    The first H1 becomes the title. Each section records the full
    path of headings above it, which chunking later uses to give
    every chunk its context.
    """
    title = doc_id
    path: list[str] = []
    sections: list[Section] = []
    buf: list[str] = []
    in_code = False

    def flush() -> None:
        body = "\n".join(buf).strip()
        buf.clear()
        if not body:
            return
        # A block made only of table rows is a table section.
        rows = [x for x in body.splitlines() if x.strip()]
        is_table = all(x.lstrip().startswith("|") for x in rows)
        kind: Literal["text", "table", "code"] = "text"
        if is_table:
            kind = "table"
        elif body.startswith("```"):
            kind = "code"
        sections.append(
            Section(
                heading_path=list(path),
                text=normalize_text(body),
                kind=kind,
            )
        )

    def is_row(line: str) -> bool:
        return line.lstrip().startswith("|")

    for line in text.splitlines():
        if line.startswith("```"):
            in_code = not in_code
        m = None if in_code else HEADING.match(line)
        if m:
            flush()
            level, heading = len(m.group(1)), m.group(2).strip()
            if level == 1 and title == doc_id:
                title = heading
                continue
            del path[level - 2 :]  # H2 is path[0]
            path.append(heading)
            continue
        last = next((x for x in reversed(buf) if x.strip()), None)
        if (
            not in_code
            and line.strip()
            and last is not None
            and is_row(line) != is_row(last)
        ):
            flush()  # boundary between prose and a table
        buf.append(line)
    flush()
    return Document(
        doc_id=doc_id,
        title=title,
        source=source,
        sections=sections,
        acl_groups=acl,
    )


# [end:parse]


def load_manual(path: Path, acl: list[str]) -> Document:
    ref = SourceRef(
        system="manuals",
        uri=f"file://{path.name}",
        fetched_at=datetime.now(UTC),
    )
    return parse_markdown(
        path.read_text(), doc_id=path.stem, source=ref, acl=acl
    )
