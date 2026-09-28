"""Chunking strategies, from naive to structure-aware."""

from collections.abc import Callable, Sequence

from pydantic import BaseModel

from halden.models.document import Document

TokenCounter = Callable[[str], int]


# [start:chunk-model]
class Chunk(BaseModel):
    chunk_id: str  # "{doc_id}:{ordinal}", stable for a doc version
    doc_id: str
    ordinal: int
    heading_path: list[str]
    kind: str
    text: str  # what the model reads and the user sees cited
    embed_text: str  # what is embedded: text plus its context
    token_count: int


# [end:chunk-model]


def split_fixed(text: str, size: int, overlap: int = 0) -> list[str]:
    """The naive baseline: fixed character windows."""
    step = max(1, size - overlap)
    return [text[i : i + size] for i in range(0, len(text), step)]


# [start:recursive]
DEFAULT_SEPARATORS = ("\n\n", "\n", ". ", " ")


def split_recursive(
    text: str,
    max_tokens: int,
    count: TokenCounter,
    separators: Sequence[str] = DEFAULT_SEPARATORS,
) -> list[str]:
    """Split on the coarsest separator that works, then merge
    adjacent pieces greedily up to ``max_tokens``.

    Paragraph breaks are preferred over line breaks, line breaks
    over sentence ends, and sentence ends over spaces, so pieces
    end at the most natural boundary available.
    """
    if count(text) <= max_tokens:
        return [text]
    for i, sep in enumerate(separators):
        raw = text.split(sep)
        if len(raw) > 1:
            finer = separators[i + 1 :]
            break
    else:
        return [text]  # one unbreakable token run; keep it whole
    # Keep each separator attached to the part before it, so no
    # text (such as a sentence's final period) is ever dropped.
    parts = [p + sep for p in raw[:-1]] + [raw[-1]]

    pieces: list[str] = []
    current = ""
    for part in parts:
        if count(current + part) <= max_tokens:
            current += part
            continue
        if current:
            pieces.append(current)
        if count(part) > max_tokens:
            pieces.extend(
                split_recursive(part, max_tokens, count, finer)
            )
            current = ""
        else:
            current = part
    if current:
        pieces.append(current)
    return [p.strip() for p in pieces if p.strip()]


# [end:recursive]


# [start:table]
def split_table(
    table: str, max_tokens: int, count: TokenCounter
) -> list[str]:
    """Split a Markdown table by rows, repeating the header in
    every piece so each chunk remains a readable table."""
    lines = table.splitlines()
    header, rows = lines[:2], lines[2:]
    pieces: list[list[str]] = []
    current: list[str] = []
    for row in rows:
        if current and count("\n".join(header + current + [row])) > (
            max_tokens
        ):
            pieces.append(current)
            current = []
        current.append(row)
    if current:
        pieces.append(current)
    return ["\n".join(header + p) for p in pieces]


# [end:table]


# [start:chunk-document]
def chunk_document(
    doc: Document,
    *,
    max_tokens: int,
    count: TokenCounter,
    context_header: bool = True,
) -> list[Chunk]:
    """Structure-aware chunking.

    Sections are never merged across headings, so a chunk always
    belongs to exactly one section. With ``context_header``, the
    embedded text is prefixed with the document title and heading
    path, so "E-17 | Empty pipe detected" is embedded as part of
    the FM-310 manual's error table, not as a free-floating row.
    """
    chunks: list[Chunk] = []
    for section in doc.sections:
        if section.kind == "table":
            pieces = split_table(section.text, max_tokens, count)
        else:
            pieces = split_recursive(section.text, max_tokens, count)
        crumbs = " > ".join(section.heading_path)
        header = (
            f"{doc.title}\n{crumbs}\n\n"
            if crumbs
            else (f"{doc.title}\n\n")
        )
        for piece in pieces:
            n = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}:{n}",
                    doc_id=doc.doc_id,
                    ordinal=n,
                    heading_path=section.heading_path,
                    kind=section.kind,
                    text=piece,
                    embed_text=(header + piece)
                    if context_header
                    else piece,
                    token_count=count(piece),
                )
            )
    return chunks


# [end:chunk-document]


def hf_token_counter(model_name: str) -> TokenCounter:
    """Count tokens with the embedding model's own tokenizer."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)

    def count(text: str) -> int:
        return len(tok.encode(text, add_special_tokens=False))

    return count
