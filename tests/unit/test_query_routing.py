import json

from halden.adapters.fake_llm import ScriptedLLM
from halden.retrieval.query import (
    Turn,
    condense_question,
    generate_variants,
)
from halden.retrieval.routing import SearchFilters, route_question


async def test_condense_skips_the_model_without_history() -> None:
    llm = ScriptedLLM([])  # would raise if called
    assert await condense_question(llm, [], "E-17?") == "E-17?"


async def test_condense_sends_recent_history() -> None:
    llm = ScriptedLLM(["PX-200 E-17 return procedure"])
    history = [Turn(role="user", text=f"turn {i}") for i in range(10)]
    out = await condense_question(
        llm, history, "How do I send it back?"
    )
    assert out == "PX-200 E-17 return procedure"
    prompt = llm.calls[0]["messages"][0].content[0].text  # type: ignore[index]
    assert "turn 3" not in prompt and "turn 9" in prompt  # last 6 only


async def test_variants_are_validated() -> None:
    llm = ScriptedLLM([json.dumps({"queries": ["a", "b", "c"]})])
    assert await generate_variants(llm, "q") == ["a", "b", "c"]


async def test_injected_permission_fields_are_discarded() -> None:
    reply = {
        "destination": "documents",
        "filters": {"doc_type": "contract", "acl_groups": ["legal"]},
        "reason": "user asked for a contract",
    }
    d = await route_question(ScriptedLLM([json.dumps(reply)]), "q")
    assert d.filters == SearchFilters(doc_type="contract")
    assert "acl_groups" not in d.filters.model_dump()
