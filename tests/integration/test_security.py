"""Access control: row-level security, API keys, conversation
isolation, and permission-aware answers through the API."""

import uuid
from pathlib import Path

from fastapi.testclient import TestClient
from psycopg_pool import AsyncConnectionPool

from halden.adapters.fake_llm import ScriptedLLM
from halden.adapters.local_embedder import HashingEmbedder
from halden.api.app import create_app
from halden.config.settings import Settings
from halden.ingestion.service import IngestionService
from halden.security.api_keys import create_key, create_user
from halden.services.ask import AskService
from halden.services.container import Services
from halden.store.conversations import ConversationStore
from halden.store.db import create_pool
from halden.store.documents import DocumentStore
from halden.store.scope import reader_scope

DATA = Path(__file__).parents[2] / "data" / "sources"
EMB = HashingEmbedder(dimensions=384)


async def load(pool: AsyncConnectionPool) -> None:
    svc = IngestionService(
        DocumentStore(pool), EMB, lambda t: len(t.split()), 60
    )
    for system, name, acl in (
        ("contracts", "contract-acme", ["legal"]),
        ("policies", "policy-travel", ["everyone"]),
    ):
        data = (DATA / system / f"{name}.md").read_bytes()
        await svc.ingest(
            doc_id=f"{system}:{name}",
            system=system,
            filename=f"{name}.md",
            data=data,
            acl=acl,
            source_version="1",
        )


async def visible(
    pool: AsyncConnectionPool, groups: list[str] | None
) -> set[str]:
    """What an unfiltered query sees: no WHERE clause at all."""
    async with pool.connection() as conn:
        if groups is None:
            rows = await (
                await conn.execute("SELECT DISTINCT doc_id FROM chunks")
            ).fetchall()
        else:
            async with reader_scope(conn, groups):
                rows = await (
                    await conn.execute(
                        "SELECT DISTINCT doc_id FROM chunks"
                    )
                ).fetchall()
    return {r[0] for r in rows}


# [start:rls-tests]
async def test_rls_fails_closed_and_filters_unfiltered_queries(
    pool: AsyncConnectionPool,
) -> None:
    await load(pool)
    assert await visible(pool, None) == set()  # no groups: nothing
    assert await visible(pool, ["everyone"]) == {
        "policies:policy-travel"
    }
    assert await visible(pool, ["everyone", "legal"]) == {
        "policies:policy-travel",
        "contracts:contract-acme",
    }


async def test_groups_do_not_leak_across_pooled_connections(
    schema_dsn: str,
) -> None:
    single = create_pool(
        Settings(dsn=schema_dsn, db_pool_min=1, db_pool_max=1)
    )
    await single.open()
    await load(single)
    assert "contracts:contract-acme" in await visible(single, ["legal"])
    # Same physical connection, next "request": SET LOCAL is gone.
    assert await visible(single, None) == set()
    await single.close()


# [end:rls-tests]


def app_client(dsn: str, script: list[object]) -> TestClient:
    settings = Settings(
        dsn=dsn, auth_mode="api_key", context_budget_tokens=800
    )
    fake = ScriptedLLM(script)  # type: ignore[arg-type]

    async def factory(s: Settings) -> Services:
        pool = create_pool(s)
        await pool.open()
        await load(pool)
        await create_user(pool, "alice", "Alice", ["everyone"])
        await create_user(pool, "lena", "Lena", ["everyone", "legal"])
        keys = {u: await create_key(pool, u) for u in ("alice", "lena")}
        convs = ConversationStore(pool)
        svc = Services(
            s,
            pool,
            EMB,
            fake,
            AskService(pool, EMB, fake, s, None, convs, rewriter=fake),
            convs,
        )
        svc.keys = keys  # type: ignore[attr-defined]
        return svc

    client = TestClient(create_app(settings, factory))
    client.fake = fake  # type: ignore[attr-defined]
    return client


def bearer(client: TestClient, user: str) -> dict[str, str]:
    key = client.app.state.services.keys[user]  # type: ignore[attr-defined]
    return {"authorization": f"Bearer {key}"}


def test_api_keys(schema_dsn: str) -> None:
    with app_client(schema_dsn, ["ok [1]."]) as c:
        q = {"question": "Business class?"}
        assert c.post("/v1/ask", json=q).status_code == 401
        good = bearer(c, "alice")["authorization"]
        forged = good[:-4] + "AAAA"
        r = c.post("/v1/ask", json=q, headers={"authorization": forged})
        assert r.status_code == 401
        assert r.headers["www-authenticate"] == "Bearer"
        assert (
            c.post(
                "/v1/ask", json=q, headers=bearer(c, "alice")
            ).status_code
            == 200
        )


def test_answers_respect_permissions(schema_dsn: str) -> None:
    script = [
        "Lead time is 45 days [1].",
        "I don't know based on the available documents.",
    ]
    with app_client(schema_dsn, script) as c:
        q = {"question": "Acme Metals diaphragm lead time"}
        lena = c.post("/v1/ask", json=q, headers=bearer(c, "lena"))
        alice = c.post("/v1/ask", json=q, headers=bearer(c, "alice"))
        assert lena.json()["citations"][0]["chunk_id"].startswith(
            "contracts:"
        )
        # Both requests reached the model, but Alice's context was
        # built from her permitted chunks only.
        assert all(
            not x["chunk_id"].startswith("contracts:")
            for x in alice.json()["citations"]
        )
        assert len(c.fake.calls) == 2  # type: ignore[attr-defined]
        sent = str(c.fake.calls[1]["messages"])  # type: ignore[attr-defined]
        assert "45-day" not in sent  # contract text never reached it


def test_conversations_are_private(schema_dsn: str) -> None:
    script = [
        "Economy under six hours [1].",
        "travel policy flight class approval",  # condensed
        "VP approval is required [1].",
    ]
    with app_client(schema_dsn, script) as c:
        a = bearer(c, "alice")
        cid = c.post("/v1/conversations", headers=a).json()[
            "conversation_id"
        ]
        c.post(
            "/v1/ask",
            json={
                "question": "Which class can I fly?",
                "conversation_id": cid,
            },
            headers=a,
        )
        stolen = c.post(
            "/v1/ask",
            json={"question": "and?", "conversation_id": cid},
            headers=bearer(c, "lena"),
        )
        assert stolen.status_code == 404
        follow = c.post(
            "/v1/ask",
            json={"question": "Who approves?", "conversation_id": cid},
            headers=a,
        ).json()
        assert (
            follow["search_query"]
            == "travel policy flight class approval"
        )
        condense_prompt = str(c.fake.calls[1]["messages"])  # type: ignore[attr-defined]
        assert "Which class can I fly?" in condense_prompt
        missing = c.post(
            "/v1/ask",
            json={
                "question": "x",
                "conversation_id": str(uuid.uuid4()),
            },
            headers=a,
        )
        assert missing.status_code == 404


async def test_app_role_cannot_bypass_rls(
    pool: AsyncConnectionPool,
) -> None:
    # Superusers and BYPASSRLS roles skip every policy, even with
    # FORCE ROW LEVEL SECURITY. Connecting as one would turn every
    # RLS test above into a false pass in production.
    async with pool.connection() as conn:
        row = await (
            await conn.execute(
                "SELECT rolsuper, rolbypassrls FROM pg_roles"
                " WHERE rolname = current_user"
            )
        ).fetchone()
    assert row == (False, False), (
        "connect as an ordinary role: see docker/initdb/"
    )
