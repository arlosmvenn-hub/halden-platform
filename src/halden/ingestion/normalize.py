"""Text normalization applied to every parser's output."""

import re
import unicodedata
from collections import Counter

# [start:normalize]
SOFT_HYPHEN = "­"
HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
SPACES = re.compile(r"[ \t   ]+")
BLANK_LINES = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """Make equivalent text byte-identical.

    NFKC folds compatibility characters (ligatures, full-width
    digits) into canonical forms, so 'ﬁlter' and 'filter' match.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace(SOFT_HYPHEN, "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = HYPHEN_BREAK.sub(r"\1\2", text)  # "cali-\nbration"
    text = SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return BLANK_LINES.sub("\n\n", text).strip()


def strip_repeated_lines(
    pages: list[str], min_share: float = 0.5
) -> list[str]:
    """Remove running headers/footers from paginated text.

    A line that appears on more than ``min_share`` of pages is
    boilerplate ("Halden Instruments — Confidential — Page 3"),
    once digits are ignored so page numbers still match.
    """
    if len(pages) < 2:
        return pages

    def key(line: str) -> str:
        return re.sub(r"\d+", "#", line.strip().lower())

    counts = Counter(
        k for page in pages for k in {key(x) for x in page.splitlines()}
    )
    limit = min_share * len(pages)
    repeated = {k for k, n in counts.items() if n > limit and k}
    return [
        "\n".join(
            x for x in page.splitlines() if key(x) not in repeated
        )
        for page in pages
    ]


# [end:normalize]
