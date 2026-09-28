"""Chapter 19: the same questions from different people, and a
conversation with memory. Real API keys, real retrieval, live model.
Requires all four sources ingested (see docs/CHAPTERS.md)."""

import asyncio
import os

from fastapi.testclient import TestClient

from halden.api.app import create_app
from halden.config.settings import Settings
from halden.security.api_keys import create_key, create_user
from halden.store.db import create_pool

# [start:people]
PEOPLE = {
    "alice": ("Alice Ng, sales engineer", ["everyone"]),
    "lena": ("Lena Roth, legal counsel", ["everyone", "legal"]),
    "sam": ("Sam Ortiz, support engineer", ["everyone", "support"]),
}
# [end:people]


async def issue_keys(settings: Settings) -> dict[str, str]:
    pool = create_pool(settings)
    await pool.open()
    keys = {}
    for uid, (name, groups) in PEOPLE.items():
        await create_user(pool, uid, name, groups)
        keys[uid] = await create_key(pool, uid)
    await pool.close()
    return keys


def show(who: str, q: str, r: dict[str, object]) -> None:
    cites = r["citations"]
    assert isinstance(cites, list)
    sources = sorted({c["chunk_id"].split(":")[0] for c in cites})
    print(f"[{who}] {q}\n  -> {r['answer']}\n  sources: {sources}\n")


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "Set ANTHROPIC_API_KEY, HALDEN_MODEL, HALDEN_MODEL_SMALL."
        )
        return
    settings = Settings(auth_mode="api_key")
    keys = asyncio.run(issue_keys(settings))

    def auth(user: str) -> dict[str, str]:
        return {"authorization": f"Bearer {keys[user]}"}

    with TestClient(create_app(settings)) as c:
        for q, users in (
            (
                "What lead time and price cap does the Acme Metals "
                "agreement specify?",
                ("alice", "lena"),
            ),
            (
                "What was the root cause in ticket 1207?",
                ("alice", "sam"),
            ),
        ):
            for u in users:
                r = c.post(
                    "/v1/ask", json={"question": q}, headers=auth(u)
                ).json()
                show(u, q, r)

        cid = c.post("/v1/conversations", headers=auth("sam")).json()[
            "conversation_id"
        ]
        for q in (
            "What does E-17 mean on the FM-310?",
            "How do I fix it?",
            "Has a customer reported something similar?",
        ):
            r = c.post(
                "/v1/ask",
                json={"question": q, "conversation_id": cid},
                headers=auth("sam"),
            ).json()
            print(
                f"[sam, conversation] {q}\n"
                f"  searched for: {r['search_query']}"
            )
            show("sam", q, r)
        other = c.post(
            "/v1/ask",
            json={"question": "Summarize.", "conversation_id": cid},
            headers=auth("alice"),
        )
        print(
            f"[alice] reuses Sam's conversation id -> HTTP "
            f"{other.status_code}"
        )


if __name__ == "__main__":
    main()
