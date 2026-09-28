from collections.abc import Sequence
from typing import Any

from halden.adapters.fake_llm import ScriptedLLM, text_response
from halden.agent.loop import Budget, Tool, run_agent
from halden.agent.search_tool import SearchSession
from halden.generation.mapreduce import map_reduce
from halden.ports.llm import LLMResponse, ToolSpec, ToolUseBlock, Usage
from halden.retrieval.pg_retriever import RetrievedChunk

SPEC = ToolSpec(
    name="search_documents",
    description="d",
    input_schema={"type": "object"},
)


def call(name: str, i: int) -> LLMResponse:
    return LLMResponse(
        content=[
            ToolUseBlock(
                id=f"t{i}", name=name, input={"query": f"q{i}"}
            )
        ],
        stop_reason="tool_use",
        usage=Usage(input_tokens=100, output_tokens=10),
        model="scripted",
    )


async def echo(args: dict[str, Any]) -> str:
    return f"result for {args['query']}"


async def test_agent_uses_tool_then_answers() -> None:
    llm = ScriptedLLM([call("search_documents", 1), "Done [1]."])
    answer, trace = await run_agent(
        llm, system="s", question="q", tools=[Tool(SPEC, echo)]
    )
    assert answer == "Done [1]."
    assert trace.stop == "end_turn" and trace.steps == 2
    result = llm.calls[1]["messages"][2].content[0]  # type: ignore[index]
    assert result.content == "result for q1"


async def test_budget_forces_a_final_answer_without_tools() -> None:
    runaway = [call("search_documents", i) for i in range(10)]
    llm = ScriptedLLM([*runaway[:2], "Best effort answer."])
    answer, trace = await run_agent(
        llm,
        system="s",
        question="q",
        tools=[Tool(SPEC, echo)],
        budget=Budget(max_steps=2),
    )
    assert answer == "Best effort answer." and trace.stop == "budget"
    assert llm.calls[2]["tools"] == []  # last call offers no tools


async def test_unknown_tool_and_handler_errors_are_reported() -> None:
    async def broken(args: dict[str, Any]) -> str:
        raise RuntimeError("index offline")

    llm = ScriptedLLM(
        [
            call("delete_everything", 1),
            call("search_documents", 2),
            "Sorry.",
        ]
    )
    _, trace = await run_agent(
        llm, system="s", question="q", tools=[Tool(SPEC, broken)]
    )
    r1 = llm.calls[1]["messages"][2].content[0]  # type: ignore[index]
    r2 = llm.calls[2]["messages"][4].content[0]  # type: ignore[index]
    assert r1.is_error and "unknown tool" in r1.content
    assert r2.is_error and "index offline" in r2.content
    assert len(trace.tool_calls) == 2


class FakeRetriever:
    def __init__(self) -> None:
        self.seen_groups: list[list[str]] = []

    async def hybrid(
        self, query: str, groups: Sequence[str], k: int = 20
    ) -> list[RetrievedChunk]:
        self.seen_groups.append(list(groups))
        ids = ["a:0", "b:0"] if query == "first" else ["b:0", "c:0"]
        return [
            RetrievedChunk(
                chunk_id=i,
                doc_id=i[0],
                title=i,
                heading_path=[],
                text=f"text {i}",
                score=1.0,
            )
            for i in ids
        ]


async def test_search_session_numbers_sources_across_searches() -> None:
    fake = FakeRetriever()
    s = SearchSession(fake, ["everyone"])  # type: ignore[arg-type]
    await s.search({"query": "first"})
    second = await s.search({"query": "second", "groups": ["legal"]})
    assert [x.chunk_id for x in s.sources] == ["a:0", "b:0", "c:0"]
    assert second.startswith("[2] b:0")  # same chunk, same label
    assert fake.seen_groups == [["everyone"], ["everyone"]]


async def test_map_reduce_flags_truncation() -> None:
    cut = text_response("partial")
    cut = cut.model_copy(update={"stop_reason": "max_tokens"})
    llm = ScriptedLLM(["note one", "NONE", cut])
    result = await map_reduce(
        llm, llm, "task", [("a", "x"), ("b", "y")]
    )
    assert result.truncated and result.calls == 3
    reduce_prompt = llm.calls[2]["messages"][0].content[0]  # type: ignore[index]
    assert "note one" in reduce_prompt.text and "NONE" not in (
        reduce_prompt.text
    )
