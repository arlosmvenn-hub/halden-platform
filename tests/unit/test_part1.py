import json
from typing import Any

import httpx
import numpy as np
import pytest

from halden.adapters.anthropic_llm import AnthropicLLM
from halden.adapters.fake_llm import ScriptedLLM
from halden.adapters.local_embedder import HashingEmbedder
from halden.generation.structured import (
    StructuredOutputError,
    complete_structured,
)
from halden.models.triage import Severity, TicketTriage
from halden.ports.llm import (
    LLMError,
    Message,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
)
from halden.retrieval.brute_force import top_k_cosine

# ---------- Anthropic adapter (recorded wire format) ----------

TOOL_USE_REPLY = {
    "id": "msg_01",
    "type": "message",
    "role": "assistant",
    "model": "test-model",
    "content": [
        {"type": "text", "text": "Checking stock."},
        {
            "type": "tool_use",
            "id": "toolu_01",
            "name": "get_stock_level",
            "input": {"part_number": "PX-200"},
        },
    ],
    "stop_reason": "tool_use",
    "usage": {"input_tokens": 120, "output_tokens": 30},
}


def adapter_with(
    handler: Any,
) -> tuple[AnthropicLLM, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        result: httpx.Response = handler(request)
        return result

    http = httpx.AsyncClient(
        base_url="https://api.anthropic.com",
        transport=httpx.MockTransport(record),
    )
    return AnthropicLLM("sk-test", "test-model", http=http), seen


async def test_adapter_sends_documented_request_shape() -> None:
    llm, seen = adapter_with(
        lambda r: httpx.Response(200, json=TOOL_USE_REPLY)
    )
    tool = ToolSpec(
        name="get_stock_level",
        description="stock",
        input_schema={"type": "object", "properties": {}},
    )
    resp = await llm.complete(
        [Message.user("How many PX-200?")],
        system="You are Ask Halden.",
        tools=[tool],
    )
    req = seen[0]
    body = json.loads(req.content)
    assert req.url.path == "/v1/messages"
    assert req.headers["x-api-key"] == "sk-test"
    assert req.headers["anthropic-version"] == "2023-06-01"
    assert body["system"] == "You are Ask Halden."
    assert body["messages"][0]["content"][0]["type"] == "text"
    assert body["tools"][0]["input_schema"]["type"] == "object"
    assert "temperature" not in body  # omitted unless requested
    assert resp.stop_reason == "tool_use"
    assert resp.tool_calls[0].input == {"part_number": "PX-200"}
    assert resp.usage.input_tokens == 120


async def test_adapter_serializes_tool_results() -> None:
    llm, seen = adapter_with(
        lambda r: httpx.Response(200, json=TOOL_USE_REPLY)
    )
    call = ToolUseBlock(id="toolu_01", name="x", input={})
    await llm.complete(
        [
            Message.user("q"),
            Message(role="assistant", content=[call]),
            Message(
                role="user",
                content=[
                    ToolResultBlock(tool_use_id="toolu_01", content="7")
                ],
            ),
        ]
    )
    msgs = json.loads(seen[0].content)["messages"]
    assert msgs[1]["content"][0]["type"] == "tool_use"
    assert msgs[2]["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "toolu_01",
        "content": "7",
        "is_error": False,
    }


@pytest.mark.parametrize(
    ("status", "retryable"), [(429, True), (529, True), (400, False)]
)
async def test_adapter_classifies_errors(
    status: int, retryable: bool
) -> None:
    llm, _ = adapter_with(
        lambda r: httpx.Response(status, json={"error": {}})
    )
    with pytest.raises(LLMError) as info:
        await llm.complete([Message.user("q")])
    assert info.value.retryable is retryable


async def test_adapter_maps_unknown_stop_reason() -> None:
    reply = dict(TOOL_USE_REPLY, stop_reason="pause_turn")
    llm, _ = adapter_with(lambda r: httpx.Response(200, json=reply))
    resp = await llm.complete([Message.user("q")])
    assert resp.stop_reason == "other"


# ---------- Chapter 2 example ----------


async def test_tool_round_trip_example() -> None:
    import examples.ch02_tool_round_trip as ex

    llm = ex.scripted_model()
    answer = await ex.ask(llm, "How many PX-200s do we have?")
    assert "42" in answer
    second_request = llm.calls[1]["messages"]
    assert isinstance(second_request, list)
    result = second_request[2].content[0]
    assert json.loads(result.content)["units"] == 42


# ---------- Chapter 3: structured output ----------

VALID = json.dumps(
    {
        "product_line": "PX-200",
        "severity": "safety",
        "summary": "Transmitter vented process gas during service.",
    }
)


async def test_structured_output_accepts_valid_json() -> None:
    llm = ScriptedLLM([f"```json\n{VALID}\n```"])
    triage = await complete_structured(
        llm, TicketTriage, [Message.user("ticket text")]
    )
    assert triage.severity is Severity.SAFETY
    assert triage.needs_human


# [start:repair-test]
async def test_structured_output_repairs_once() -> None:
    bad = VALID.replace("PX-200", "pressure thing")
    llm = ScriptedLLM([bad, VALID])
    triage = await complete_structured(
        llm, TicketTriage, [Message.user("ticket text")]
    )
    assert triage.product_line == "PX-200"
    retry_msgs = llm.calls[1]["messages"]
    assert isinstance(retry_msgs, list)
    assert "invalid" in retry_msgs[-1].content[0].text


# [end:repair-test]


async def test_structured_output_gives_up() -> None:
    llm = ScriptedLLM(["not json", "still not json"])
    with pytest.raises(StructuredOutputError):
        await complete_structured(
            llm, TicketTriage, [Message.user("t")]
        )


# ---------- Chapter 4: embeddings and brute force ----------


async def test_hashing_embedder_is_normalized_and_stable() -> None:
    emb = HashingEmbedder(dimensions=64)
    a = await emb.embed_documents(["PX-200 zero adjust", "gasket"])
    b = await emb.embed_query("PX-200 zero adjust")
    assert a.shape == (2, 64)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)
    assert np.allclose(a[0], b)


def test_top_k_cosine_orders_by_score() -> None:
    docs = np.array(
        [[1, 0, 0], [0.6, 0.8, 0], [0, 0, 1]], dtype=np.float32
    )
    q = np.array([0.8, 0.6, 0], dtype=np.float32)
    hits = top_k_cosine(q, docs, k=2)
    assert [i for i, _ in hits] == [1, 0]
    assert hits[0][1] == pytest.approx(0.96)
    assert top_k_cosine(q, docs[:0], k=3) == []


async def test_adapter_sends_temperature_only_when_set() -> None:
    llm, seen = adapter_with(
        lambda r: httpx.Response(200, json=TOOL_USE_REPLY)
    )
    await llm.complete([Message.user("q")], temperature=0.2)
    assert json.loads(seen[0].content)["temperature"] == 0.2


async def test_prompt_cache_marks_the_system_prompt() -> None:
    reply = {
        **TOOL_USE_REPLY,
        "usage": {
            "input_tokens": 12,
            "output_tokens": 30,
            "cache_read_input_tokens": 2700,
            "cache_creation_input_tokens": 0,
        },
    }
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=reply)

    http = httpx.AsyncClient(
        base_url="https://api.anthropic.com",
        transport=httpx.MockTransport(record),
    )
    llm = AnthropicLLM("sk-test", "m", http=http, prompt_cache=True)
    resp = await llm.complete([Message.user("q")], system="Long manual")
    body = json.loads(seen[0].content)
    assert body["system"] == [
        {
            "type": "text",
            "text": "Long manual",
            "cache_control": {"type": "ephemeral"},
        }
    ]
    assert resp.usage.cache_read_input_tokens == 2700
    assert resp.usage.total_input_tokens == 2712
