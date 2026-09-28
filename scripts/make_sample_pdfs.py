"""Render the sample manuals as paginated PDFs for Chapter 17.

Each page carries a running header and a page-number footer, the
boilerplate that real manuals have and that ingestion must remove.
Output: data/sources/manuals/*.pdf (committed to the repository).
"""

import re
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen.canvas import Canvas

ROOT = Path(__file__).resolve().parent.parent
SOURCES = [ROOT / "data" / "manuals", ROOT / "data" / "pdf_only"]
OUT = ROOT / "data" / "sources" / "manuals"
LEFT, TOP, BOTTOM, LINE = 56, 800, 60, 14
WIDTH_CHARS = 92


def wrap(text: str) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > WIDTH_CHARS:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    return lines + ([cur] if cur else [])


def render(md: Path, pdf: Path) -> None:
    doc_code = re.search(r"Document (\S+),", md.read_text())
    header = f"Halden Instruments  |  {doc_code[1] if doc_code else ''}"
    c = Canvas(str(pdf), pagesize=A4)
    page, y = 1, TOP

    def new_page() -> None:
        nonlocal page, y
        c.setFont("Helvetica", 8)
        c.drawString(LEFT, 820, header)
        c.drawString(LEFT, 30, f"Page {page}")
        y = TOP

    def emit(
        line: str, font: str = "Helvetica", size: int = 10
    ) -> None:
        nonlocal page, y
        if y < BOTTOM:
            c.showPage()
            page += 1
            new_page()
        c.setFont(font, size)
        c.drawString(LEFT, y, line)
        y -= LINE

    new_page()
    for raw in md.read_text().splitlines():
        line = raw.rstrip()
        if not line:
            y -= LINE // 2
        elif line.startswith("# "):
            emit(line[2:], "Helvetica-Bold", 14)
        elif line.startswith("#"):
            emit(line.lstrip("# "), "Helvetica-Bold", 11)
        elif line.startswith("|"):
            if set(line) <= set("|-: "):
                continue  # Markdown separator row
            cells = [x.strip() for x in line.strip("|").split("|")]
            emit(" | ".join(cells), "Courier", 9)
        else:
            for part in wrap(line):
                emit(part)
    c.save()


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for md in sorted(p for s in SOURCES for p in s.glob("*.md")):
        render(md, OUT / f"{md.stem}.pdf")
        print("wrote", (OUT / f"{md.stem}.pdf").relative_to(ROOT))
