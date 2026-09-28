"""Traces and metrics from a real AskService run, scripted model."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from psycopg_pool import AsyncConnectionPool

from halden.adapters.fake_llm import ScriptedLLM
from halden.adapters.local_embedder import HashingEmbedder
from halden.config.settings import Settings
from halden.ingestion.service import IngestionService
from halden.observability.instrumented import (
    TracedAskService,
    TracedEmbedder,
    TracedLLM,
)
from halden.observability.telemetry import (
    pseudonym,
    redact,
    setup_telemetry,
)
from halden.ports.llm import LLMResponse, TextBlock, Usage
from halden.security.identity import Identity
from halden.store.documents import DocumentStore

DATA = Path(__file__).parents[2] / "data"
EXPORTER = InMemorySpanExporter()
READER = InMemoryMetricReader()
# Global providers can be installed once per process.
setup_telemetry("test", EXPORTER, metric_reader=READER, batch=False)


@pytest.fixture(autouse=True)
def clean() -> Iterator[None]:
    EXPORTER.clear()
    yield


def reply(text: str) -> LLMResponse:
    return LLMResponse(
        content=[TextBlock(text=text)],
        stop_reason="end_turn",
        usage=Usage(input_tokens=1200, output_tokens=40),
        model="claude-sonnet-5",
    )


async def service(
    pool: AsyncConnectionPool, capture: bool = False
) -> TracedAskService:
    emb = HashingEmbedder(dimensions=384)
    await IngestionService(
        DocumentStore(pool), emb, lambda t: len(t.split()), 60
    ).ingest(
        doc_id="m:px",
        system="m",
        filename="px.md",
        data=(DATA / "manuals/px200_manual.md").read_bytes(),
        acl=["everyone"],
        source_version="1",
    )
    EXPORTER.clear()  # ingestion spans are not under test
    settings = Settings(telemetry_capture_content=capture)
    llm = TracedLLM(
        ScriptedLLM([reply("At least 250 ohms [1].")]),
        "claude-sonnet-5",
        capture_content=capture,
    )
    return TracedAskService(pool, TracedEmbedder(emb), llm, settings)


ALICE = Identity(user_id="alice@halden.example", groups=["everyone"])


# [start:spans]
async def test_one_ask_is_one_trace(pool: AsyncConnectionPool) -> None:
    svc = await service(pool)
    await svc.ask("What loop resistance does HART need?", ALICE)
    spans = {s.name: s for s in EXPORTER.get_finished_spans()}
    assert set(spans) == {
        "ask",
        "retrieval chunks",
        "embeddings hashing-384",
        "build_context",
        "chat claude-sonnet-5",
    }
    root = spans["ask"]
    assert {s.context.trace_id for s in spans.values()} == {
        root.context.trace_id
    }
    chat = spans["chat claude-sonnet-5"]
    assert chat.parent is not None
    assert chat.parent.span_id == root.context.span_id
    assert chat.attributes is not None
    assert chat.attributes["gen_ai.provider.name"] == "anthropic"
    assert chat.attributes["gen_ai.usage.input_tokens"] == 1200
    assert chat.attributes["halden.cost_usd"] == pytest.approx(0.0028)
    retrieval = spans["retrieval chunks"].attributes or {}
    assert "m:px:" in str(retrieval["gen_ai.retrieval.documents"])


async def test_no_content_or_identity_by_default(
    pool: AsyncConnectionPool,
) -> None:
    svc = await service(pool)
    await svc.ask("What loop resistance does HART need?", ALICE)
    recorded = " ".join(
        str(v)
        for s in EXPORTER.get_finished_spans()
        for v in (s.attributes or {}).values()
    )
    assert "loop resistance" not in recorded  # no question text
    assert "250 ohms" not in recorded  # no answer text
    assert "alice" not in recorded  # only a pseudonym


# [end:spans]


async def test_content_capture_is_redacted(
    pool: AsyncConnectionPool,
) -> None:
    svc = await service(pool, capture=True)
    await svc.ask("HART loop? Call me at +1 425 555 0100", ALICE)
    spans = {s.name: s for s in EXPORTER.get_finished_spans()}
    query = (spans["retrieval chunks"].attributes or {})[
        "gen_ai.retrieval.query.text"
    ]
    assert "[PHONE]" in str(query) and "555" not in str(query)


async def test_metrics_are_recorded(pool: AsyncConnectionPool) -> None:
    svc = await service(pool)
    await svc.ask("What loop resistance does HART need?", ALICE)
    data = READER.get_metrics_data()
    assert data is not None
    names = {
        m.name
        for rm in data.resource_metrics
        for sm in rm.scope_metrics
        for m in sm.metrics
    }
    assert {
        "gen_ai.client.operation.duration",
        "gen_ai.client.inference.usage.input_tokens",
        "halden.llm.cost",
        "halden.ask.duration",
    } <= names


def test_redact_and_pseudonym() -> None:
    text = "key sk-ant-abc123 and hk_live_9 for bob@x.io"
    assert redact(text) == "key [API_KEY] and [API_KEY] for [EMAIL]"
    assert pseudonym("alice", "s1") == pseudonym("alice", "s1")
    assert pseudonym("alice", "s1") != pseudonym("alice", "s2")


def test_server_span_continues_the_callers_trace(
    schema_dsn: str,
) -> None:
    from tests.integration.test_api import client_for

    trace_id = "4bf92f3577b34da6a3ce929d0e0e4736"
    with client_for(schema_dsn, ["At least 250 ohms [1]."]) as client:
        EXPORTER.clear()
        r = client.post(
            "/v1/ask",
            json={"question": "HART loop resistance?"},
            headers={
                "traceparent": f"00-{trace_id}-00f067aa0ba902b7-01"
            },
        )
    assert r.status_code == 200
    server = next(
        s
        for s in EXPORTER.get_finished_spans()
        if s.name == "POST /v1/ask"
    )
    assert f"{server.context.trace_id:032x}" == trace_id
    attrs = server.attributes or {}
    assert attrs["http.response.status_code"] == 200
    assert attrs["halden.request_id"] == r.headers["x-request-id"]
