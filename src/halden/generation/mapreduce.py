"""Map-reduce over many documents for whole-corpus questions."""

from collections.abc import Sequence
from dataclasses import dataclass

from halden.ports.llm import LLMClient, Message

# [start:mapreduce]
MAP_SYSTEM = """\
Extract only the statements from the document that directly answer
the task. Quote them briefly and precisely, one per line. Exclude
general instructions that do not answer the task. If nothing is
relevant, reply exactly: NONE"""

REDUCE_SYSTEM = """\
Combine the notes into one complete, deduplicated answer to the
task, grouped sensibly. Use only the notes. Name the source
document of each item in parentheses. Be concise."""


@dataclass(frozen=True)
class MapReduceResult:
    answer: str
    calls: int
    tokens: int
    truncated: bool  # the final answer hit its output limit


async def map_reduce(
    mapper: LLMClient,
    reducer: LLMClient,
    task: str,
    documents: Sequence[tuple[str, str]],  # (title, text)
    *,
    reduce_max_tokens: int = 1500,
) -> MapReduceResult:
    """Map calls are independent, so production code runs them
    concurrently with a limit (Chapter 25)."""
    notes, calls, tokens = [], 0, 0
    for title, text in documents:
        r = await mapper.complete(
            [
                Message.user(
                    f"Task: {task}\n\nDocument: {title}\n{text}"
                )
            ],
            system=MAP_SYSTEM,
            max_tokens=500,
        )
        calls += 1
        tokens += r.usage.input_tokens + r.usage.output_tokens
        if r.text.strip() != "NONE":
            notes.append(f"From {title}:\n{r.text.strip()}")
    r = await reducer.complete(
        [
            Message.user(
                f"Task: {task}\n\nNotes:\n\n" + "\n\n".join(notes)
            )
        ],
        system=REDUCE_SYSTEM,
        max_tokens=reduce_max_tokens,
    )
    return MapReduceResult(
        answer=r.text,
        calls=calls + 1,
        tokens=tokens + r.usage.input_tokens + r.usage.output_tokens,
        # A truncated answer is silently incomplete: surface it.
        truncated=r.stop_reason == "max_tokens",
    )


# [end:mapreduce]
