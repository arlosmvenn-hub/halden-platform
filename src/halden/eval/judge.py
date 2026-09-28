"""LLM-as-judge: faithfulness to sources, correctness vs reference.

A judge is a model call like any other: it can be wrong, biased,
and inconsistent. Calibrate it against labeled examples before
trusting its numbers (see calibration.py).
"""

from typing import Literal

from pydantic import BaseModel, Field

from halden.generation.structured import complete_structured
from halden.ports.llm import LLMClient, Message

# [start:faithfulness]
FAITHFULNESS_RUBRIC = """\
You are auditing an answer produced by a documentation assistant.
You receive the QUESTION, numbered SOURCES, and the ANSWER.

1. Split the ANSWER into atomic factual claims. Ignore citation
   markers, greetings, and statements that information is
   missing from the sources.
2. Read each claim in the context of the QUESTION: an answer
   need not repeat conditions the question already states.
   For each claim decide: is it fully supported by the SOURCES?
   "Supported" means a careful reader could verify it from the
   sources alone. General knowledge does not count. A claim that
   is plausible but absent from the sources is NOT supported. A
   number, unit, or limit that differs from the sources is NOT
   supported.
3. Write a one-sentence reason before each verdict.

Judge only support, not style, length, or helpfulness."""


class ClaimVerdict(BaseModel):
    claim: str
    reason: str
    supported: bool


class FaithfulnessVerdict(BaseModel):
    claims: list[ClaimVerdict] = Field(default_factory=list)

    @property
    def score(self) -> float:
        """Share of claims supported. No claims: nothing to doubt."""
        if not self.claims:
            return 1.0
        return sum(c.supported for c in self.claims) / len(self.claims)


async def judge_faithfulness(
    judge: LLMClient, question: str, sources: str, answer: str
) -> FaithfulnessVerdict:
    prompt = (
        f"<question>{question}</question>\n"
        f"<sources>\n{sources}\n</sources>\n\n"
        f"<answer>\n{answer}\n</answer>"
    )
    return await complete_structured(
        judge,
        FaithfulnessVerdict,
        [Message.user(prompt)],
        system=FAITHFULNESS_RUBRIC,
    )


# [end:faithfulness]

# [start:correctness]
CORRECTNESS_RUBRIC = """\
You compare an assistant's ANSWER with a REFERENCE answer written
by a subject-matter expert, for the same QUESTION.

- "correct": contains every essential fact of the REFERENCE and
  contradicts none of it. Extra detail is fine if it is not wrong.
- "partial": some essential facts present, some missing, and
  nothing contradicted.
- "incorrect": contradicts the REFERENCE, answers a different
  question, or declines to answer.

Wording, length, and citation markers do not matter. Write your
reasoning first, in at most two sentences, then the verdict."""


class CorrectnessVerdict(BaseModel):
    reasoning: str
    verdict: Literal["correct", "partial", "incorrect"]

    @property
    def score(self) -> float:
        return {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}[
            self.verdict
        ]


async def judge_correctness(
    judge: LLMClient, question: str, reference: str, answer: str
) -> CorrectnessVerdict:
    prompt = (
        f"<question>{question}</question>\n"
        f"<reference>{reference}</reference>\n"
        f"<answer>\n{answer}\n</answer>"
    )
    return await complete_structured(
        judge,
        CorrectnessVerdict,
        [Message.user(prompt)],
        system=CORRECTNESS_RUBRIC,
    )


# [end:correctness]
