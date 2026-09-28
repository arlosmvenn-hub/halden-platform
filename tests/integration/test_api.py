"""The HTTP API end to end, with a scripted model."""

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from halden.adapters.fake_llm import ScriptedLLM
from halden.adapters.local_embedder import HashingEmbedder
from halden.api.app import create_app
from halden.config.settings import Settings
from halden.ingestion.service import IngestionService
from halden.ports.llm import LLMError
from halden.services.ask import AskService
from halden.services.container import Services
from halden.store.conversations import ConversationStore
from halden.store.db import create_pool
from halden.store.documents import DocumentStore

DATA = Path(__file__).parents[2] / "data"


class FailingLLM(ScriptedLLM):
    async def complete(self, *a: object, **k: object):  # type: ignore[no-untyped-def,override]
        raise LLMError("overloaded", retryable=True)


def client_for(
    dsn: str,
    script: list[object],
    groups: list[str] | None = None,
    llm: ScriptedLLM | None = None,
) -> TestClient:
    settings = Settings(
        dsn=dsn,
        auth_mode="dev",
        dev_groups=groups or ["everyone"],
        context_budget_tokens=800,
    )
    fake = llm or ScriptedLLM(script)  # type: ignore[arg-type]

    async def factory(s: Settings) -> Services:
        pool = create_pool(s)
        await pool.open()
        emb = HashingEmbedder(dimensions=384)
        svc = IngestionService(
            DocumentStore(pool), emb, lambda t: len(t.split()), 60
        )
        md = (DATA / "manuals/px200_manual.md").read_bytes()
        await svc.ingest(
            doc_id="m:px",
            system="m",
            filename="px.md",
            data=md,
            acl=["everyone"],
            source_version="1",
        )
        convs = ConversationStore(pool)
        ask = AskService(pool, emb, fake, s, None, convs)
        return Services(s, pool, emb, fake, ask, convs)

    client = TestClient(create_app(settings, factory))
    client.fake = fake  # type: ignore[attr-defined]
    return client


@pytest.fixture
def dsn(schema_dsn: str) -> Iterator[str]:
    yield schema_dsn


def test_answer_with_validated_citations(dsn: str) -> None:
    with client_for(
        dsn, ["E-17 is a sensor diaphragm fault [1]."]
    ) as c:
        r = c.post(
            "/v1/ask",
            json={"question": "What is E-17?"},
            headers={"x-request-id": "req-123"},
        )
    body = r.json()
    assert (
        r.status_code == 200 and r.headers["x-request-id"] == "req-123"
    )
    assert body["citation_check_ok"] and not body["abstained"]
    assert body["citations"][0]["label"] == 1
    assert body["citations"][0]["chunk_id"].startswith("m:px:")


def test_no_permitted_sources_means_no_model_call(dsn: str) -> None:
    with client_for(dsn, [], groups=["nobody"]) as c:
        r = c.post("/v1/ask", json={"question": "What is E-17?"})
        assert c.fake.calls == []  # type: ignore[attr-defined]
    assert r.json()["abstained"] is True


def test_invented_citation_is_retried_then_withheld(dsn: str) -> None:
    script = ["E-17 means X [9].", "Still wrong [9]."]
    with client_for(dsn, script) as c:
        body = c.post("/v1/ask", json={"question": "E-17?"}).json()
        retry = c.fake.calls[1]["messages"][-1]  # type: ignore[attr-defined]
    assert "[9]" in retry.content[0].text
    assert body["abstained"] and not body["citation_check_ok"]
    assert "Still wrong" not in body["answer"]


def test_errors_are_problem_details(dsn: str) -> None:
    with client_for(dsn, []) as c:
        bad = c.post("/v1/ask", json={"question": ""})
    assert bad.status_code == 422
    assert bad.headers["content-type"] == "application/problem+json"
    with client_for(dsn, [], llm=FailingLLM([])) as c:
        down = c.post("/v1/ask", json={"question": "E-17?"})
    assert (
        down.status_code == 503 and down.headers["retry-after"] == "5"
    )
    assert "overloaded" not in down.text  # no internals leak


def test_streaming_events(dsn: str) -> None:
    with (
        client_for(dsn, ["Return the unit for repair [1]."]) as c,
        c.stream(
            "POST", "/v1/ask/stream", json={"question": "E-17?"}
        ) as r,
    ):
        events = [
            json.loads(line[5:])
            for line in r.iter_lines()
            if line.startswith("data:")
        ]
    kinds = [e["type"] for e in events]
    assert kinds[0] == "sources" and kinds[-1] == "done"
    assert "".join(
        e["text"] for e in events if e["type"] == "token"
    ) == ("Return the unit for repair [1].")
    assert events[-1]["citations"][0]["label"] == 1


def test_health_and_readiness(dsn: str) -> None:
    with client_for(dsn, []) as c:
        assert c.get("/healthz").json() == {"status": "ok"}
        assert c.get("/readyz").json() == {"status": "ready"}
