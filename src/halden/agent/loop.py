"""A bounded tool-using agent loop."""

import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from halden.ports.llm import (
    ContentBlock,
    LLMClient,
    LLMError,
    Message,
    ToolResultBlock,
    ToolSpec,
)

ToolHandler = Callable[[dict[str, Any]], Awaitable[str]]


@dataclass(frozen=True)
class Tool:
    spec: ToolSpec
    handler: ToolHandler


# [start:budget]
@dataclass(frozen=True)
class Budget:
    """Hard limits: the loop stops when any is reached, whatever
    the model wants. They are safety features, not tuning knobs."""

    max_steps: int = 4  # model calls that may request tools
    max_tool_calls: int = 6
    max_input_tokens: int = 40_000  # summed over all calls


@dataclass
class Trace:
    steps: int = 0
    tool_calls: list[tuple[str, dict[str, Any]]] = field(
        default_factory=list
    )
    input_tokens: int = 0
    output_tokens: int = 0
    stop: str = ""


# [end:budget]

FINAL_NUDGE = (
    "Tool budget exhausted. Answer now using only the sources "
    "already retrieved, or say you don't know."
)


# [start:loop]
async def run_agent(
    llm: LLMClient,
    *,
    system: str,
    question: str,
    tools: Sequence[Tool],
    budget: Budget | None = None,
    max_tokens: int = 800,
) -> tuple[str, Trace]:
    budget = budget or Budget()
    by_name = {t.spec.name: t for t in tools}
    specs = [t.spec for t in tools]
    history = [Message.user(question)]
    trace = Trace()

    while True:
        out_of_budget = (
            trace.steps >= budget.max_steps
            or len(trace.tool_calls) >= budget.max_tool_calls
            or trace.input_tokens >= budget.max_input_tokens
        )
        if out_of_budget:
            # One last call with no tools: the model must answer.
            history.append(Message.user(FINAL_NUDGE))
        resp = await llm.complete(
            history,
            system=system,
            tools=() if out_of_budget else specs,
            max_tokens=max_tokens,
        )
        trace.steps += 1
        trace.input_tokens += resp.usage.input_tokens
        trace.output_tokens += resp.usage.output_tokens
        if resp.stop_reason != "tool_use" or out_of_budget:
            trace.stop = "budget" if out_of_budget else resp.stop_reason
            return resp.text, trace

        history.append(Message(role="assistant", content=resp.content))
        results: list[ContentBlock] = []
        for call in resp.tool_calls:
            trace.tool_calls.append((call.name, call.input))
            tool = by_name.get(call.name)
            try:
                if tool is None:
                    raise LLMError(
                        f"unknown tool {call.name}", retryable=False
                    )
                content = await tool.handler(call.input)
                is_error = False
            except Exception as exc:  # report, don't crash the loop
                content = json.dumps({"error": str(exc)[:300]})
                is_error = True
            results.append(
                ToolResultBlock(
                    tool_use_id=call.id,
                    content=content,
                    is_error=is_error,
                )
            )
        history.append(Message(role="user", content=results))


# [end:loop]
