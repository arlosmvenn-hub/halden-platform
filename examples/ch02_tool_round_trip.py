"""Chapter 2: one complete tool-calling round trip.

Run offline (scripted model):   python examples/ch02_tool_round_trip.py
Run live: set ANTHROPIC_API_KEY and HALDEN_MODEL, then run the same.
"""

import asyncio
import json
import os
from typing import Any

from halden.adapters.anthropic_llm import AnthropicLLM
from halden.adapters.fake_llm import ScriptedLLM
from halden.ports.llm import (
    LLMClient,
    LLMResponse,
    Message,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
    Usage,
)

STOCK_TOOL = ToolSpec(
    name="get_stock_level",
    description=(
        "Return units on hand for one Halden part number, "
        "e.g. 'PX-200'. Use for any question about stock."
    ),
    input_schema={
        "type": "object",
        "properties": {"part_number": {"type": "string"}},
        "required": ["part_number"],
    },
)

FAKE_INVENTORY = {"PX-200": 42, "FM-310": 0}


def get_stock_level(part_number: str) -> dict[str, Any]:
    """The application's real code. The model never runs this."""
    units = FAKE_INVENTORY.get(part_number.upper())
    if units is None:
        return {"error": f"unknown part number {part_number}"}
    return {"part_number": part_number.upper(), "units": units}


# [start:ask]
async def ask(llm: LLMClient, question: str) -> str:
    history = [Message.user(question)]

    # Round 1: the model decides whether it needs the tool.
    first = await llm.complete(history, tools=[STOCK_TOOL])
    if first.stop_reason != "tool_use":
        return first.text

    # The application executes every requested call.
    history.append(Message(role="assistant", content=first.content))
    results = []
    for call in first.tool_calls:
        output = get_stock_level(**call.input)
        results.append(
            ToolResultBlock(
                tool_use_id=call.id,
                content=json.dumps(output),
                is_error="error" in output,
            )
        )
    history.append(Message(role="user", content=results))

    # Round 2: the model turns tool output into an answer.
    final = await llm.complete(history, tools=[STOCK_TOOL])
    return final.text


# [end:ask]
def scripted_model() -> ScriptedLLM:
    call = ToolUseBlock(
        id="toolu_01",
        name="get_stock_level",
        input={"part_number": "PX-200"},
    )
    return ScriptedLLM(
        [
            LLMResponse(
                content=[call],
                stop_reason="tool_use",
                usage=Usage(),
                model="scripted",
            ),
            "We have 42 PX-200 transmitters in stock.",
        ]
    )


async def main() -> None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    llm: LLMClient
    if key:
        llm = AnthropicLLM(key, os.environ["HALDEN_MODEL"])
    else:
        llm = scripted = scripted_model()
    answer = await ask(llm, "How many PX-200s do we have?")
    print("Answer:", answer)
    if not key:
        for i, c in enumerate(scripted.calls, start=1):
            msgs = c["messages"]
            assert isinstance(msgs, list)
            print(f"--- request {i}: {len(msgs)} message(s)")
            for m in msgs:
                print(json.dumps(m.model_dump())[:70])


if __name__ == "__main__":
    asyncio.run(main())
