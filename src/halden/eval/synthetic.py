"""Generate evaluation questions from the corpus with a model.

Synthetic questions are cheap and cover every document, but they
are written by a model that is *looking at the answer*: they tend
to reuse the passage's own words, which makes retrieval look
better than it is. Use them for breadth, keep a human-written
golden set for truth, and report the two separately.
"""

import re

from pydantic import BaseModel, Field

from halden.eval.dataset import EvalCase, normalize
from halden.generation.structured import (
    StructuredOutputError,
    complete_structured,
)
from halden.ports.llm import LLMClient, Message

# [start:prompt]
SYNTH_SYSTEM = """\
You write test questions for a company documentation assistant.
Given one PASSAGE, write a question that an employee who has NOT
read the passage might ask, which the passage answers. Use the
words such an employee would use, not the passage's wording.

Also return EVIDENCE: the shortest verbatim quote from the passage
(at most 12 words) that contains the answer, and a one-sentence
REFERENCE answer."""


class SyntheticItem(BaseModel):
    question: str = Field(min_length=8)
    evidence: str = Field(min_length=3)
    reference: str


# [end:prompt]


# [start:generate]
async def synthesize(
    llm: LLMClient,
    chunk_id: str,
    passage: str,
    groups: list[str],
) -> EvalCase | None:
    """One case from one chunk, or None if the model's output is
    unusable. The evidence quote is *checked* against the passage:
    a paraphrased 'quote' would silently become an unmatchable
    label."""
    try:
        item = await complete_structured(
            llm,
            SyntheticItem,
            [Message.user(f"<passage>\n{passage}\n</passage>")],
            system=SYNTH_SYSTEM,
        )
    except StructuredOutputError:
        return None
    if normalize(item.evidence) not in normalize(passage):
        return None
    return EvalCase(
        id=f"syn:{chunk_id}",
        question=item.question,
        groups=groups,
        evidence=[item.evidence],
        reference=item.reference,
        tags=["synthetic"],
        origin="synthetic",
    )


# [end:generate]

WORD = re.compile(r"[a-z0-9][a-z0-9\-]+")
STOPWORDS = (
    "the a an and or of to in on for is are was be do does i my "
    "what which how when where why can with at by it this that "
    "should must much many from as"
)
STOP = frozenset(STOPWORDS.split())


# [start:overlap]
def lexical_overlap(question: str, passage: str) -> float:
    """Share of the question's content words that also appear in
    the passage. High overlap makes keyword and embedding search
    easy; real users rarely echo the document this closely."""
    q = set(WORD.findall(normalize(question))) - STOP
    p = set(WORD.findall(normalize(passage)))
    return len(q & p) / len(q) if q else 0.0


# [end:overlap]
