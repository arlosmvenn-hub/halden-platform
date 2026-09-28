"""Evaluation datasets: questions, who asks them, and evidence."""

import re
import unicodedata
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

DASHES = re.compile(r"[‐-―−]")


def normalize(text: str) -> str:
    """Compare text the way a reader would: case, Unicode forms,
    dash variants, and whitespace don't matter."""
    text = unicodedata.normalize("NFKC", text)
    text = DASHES.sub("-", text).casefold()
    return " ".join(text.split())


# [start:case]
class EvalCase(BaseModel):
    """One question with everything needed to grade it.

    Relevance is labeled with *evidence*: short verbatim strings
    that a relevant chunk must contain. Unlike chunk IDs, evidence
    labels survive re-chunking and re-indexing, so the dataset
    doesn't rot every time the pipeline changes.
    """

    id: str
    question: str = Field(min_length=3)
    groups: list[str] = Field(default_factory=lambda: ["everyone"])
    evidence: list[str] = Field(default_factory=list)
    reference: str = ""  # a correct answer, for the judge
    answerable: bool = True  # False: the right answer is to abstain
    tags: list[str] = Field(default_factory=list)
    origin: Literal["golden", "synthetic"] = "golden"

    @model_validator(mode="after")
    def _consistent(self) -> "EvalCase":
        if self.answerable and not self.evidence:
            raise ValueError(f"{self.id}: answerable needs evidence")
        if not self.answerable and self.evidence:
            raise ValueError(f"{self.id}: unanswerable has evidence")
        return self


# [end:case]


def load_cases(path: Path) -> list[EvalCase]:
    cases = [
        EvalCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [c.id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate case ids")
    return cases


def save_cases(path: Path, cases: list[EvalCase]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(c.model_dump_json() + "\n" for c in cases),
        encoding="utf-8",
    )


# [start:lint]
def missing_evidence(
    cases: list[EvalCase], corpus: list[str]
) -> dict[str, list[str]]:
    """Evidence strings that occur in no chunk of the index.

    Run this before every evaluation. A label that matches nothing
    makes a case unwinnable, and the metric blames retrieval for
    what is really a broken label, or a broken index.
    """
    texts = [normalize(t) for t in corpus]
    out: dict[str, list[str]] = {}
    for case in cases:
        gone = [
            e
            for e in case.evidence
            if not any(normalize(e) in t for t in texts)
        ]
        if gone:
            out[case.id] = gone
    return out


# [end:lint]
