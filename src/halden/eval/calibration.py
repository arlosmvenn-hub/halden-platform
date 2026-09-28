"""Measure how far a judge's verdicts can be trusted.

Real calibration compares the judge with human labels. This module
builds labeled items *by construction*, so the ground truth is
known without a labeling session: a reference answer shown with the
sources it came from is faithful; the same answer with one number
changed, or with one invented sentence appended, is not.
"""

import asyncio
import re
from collections.abc import Sequence
from dataclasses import dataclass

from halden.eval.dataset import EvalCase, normalize
from halden.eval.judge import judge_faithfulness
from halden.ports.llm import LLMClient

# Plausible, specific, and absent from every Halden source.
INVENTED = [
    "A printed copy of this guidance ships with every unit.",
    "Exceptions are approved by the regional director.",
    "This requirement was introduced in the 2023 revision.",
    "Contact the Halden service desk at extension 4410 first.",
    "The same rule applies to contractors and interns.",
    "Allow up to two business days for confirmation.",
]
NUMBER = re.compile(r"\d+(?:\.\d+)?")


@dataclass(frozen=True)
class LabeledItem:
    case_id: str
    question: str
    variant: str  # original | number | invented
    sources: str
    answer: str
    faithful: bool  # the ground truth


# [start:construct]
def mutate_number(text: str) -> str | None:
    """Change the first number: 15 -> 25, 0.075 -> 0.175. None if
    the text has no number."""
    m = NUMBER.search(text)
    if m is None:
        return None
    old = m.group()
    new = (
        str(int(old) + 10) if old.isdigit() else f"{float(old) + 0.1:g}"
    )
    return text[: m.start()] + new + text[m.end() :]


def build_items(
    cases: Sequence[EvalCase], corpus: Sequence[str]
) -> list[LabeledItem]:
    items = []
    for i, case in enumerate(c for c in cases if c.answerable):
        chunks = [
            t
            for t in corpus
            if any(normalize(e) in normalize(t) for e in case.evidence)
        ]
        sources = "\n\n".join(
            f"[{n}] {t}" for n, t in enumerate(chunks, start=1)
        )
        ref = case.reference
        items.append(
            LabeledItem(
                case.id, case.question, "original", sources, ref, True
            )
        )
        wrong = mutate_number(ref)
        if wrong is not None:
            items.append(
                LabeledItem(
                    case.id,
                    case.question,
                    "number",
                    sources,
                    wrong,
                    False,
                )
            )
        extra = f"{ref} {INVENTED[i % len(INVENTED)]}"
        items.append(
            LabeledItem(
                case.id,
                case.question,
                "invented",
                sources,
                extra,
                False,
            )
        )
    return items


# [end:construct]


# [start:kappa]
def cohen_kappa(a: Sequence[bool], b: Sequence[bool]) -> float:
    """Agreement between two raters, corrected for chance.

    1.0 is perfect agreement, 0.0 is what two raters guessing at
    their own base rates would reach. A judge that calls every
    answer faithful can score high raw agreement on a mostly
    faithful set and still have a kappa of zero.
    """
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa, pb = sum(a) / n, sum(b) / n
    chance = pa * pb + (1 - pa) * (1 - pb)
    if chance == 1:
        return 1.0
    return (observed - chance) / (1 - chance)


# [end:kappa]


async def judge_items(
    judge: LLMClient,
    items: Sequence[LabeledItem],
    concurrency: int = 4,
) -> list[bool]:
    """The judge's call per item: faithful means every claim
    was supported."""
    gate = asyncio.Semaphore(concurrency)

    async def one(item: LabeledItem) -> bool:
        async with gate:
            v = await judge_faithfulness(
                judge, item.question, item.sources, item.answer
            )
            return v.score == 1.0

    return list(await asyncio.gather(*(one(x) for x in items)))
