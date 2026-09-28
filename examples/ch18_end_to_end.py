"""Chapter 18: the API end to end with real retrieval and a live
model. Requires the manuals ingested (scripts/ingest.py) and
ANTHROPIC_API_KEY + HALDEN_MODEL. Uses dev auth (fixed identity)."""

import json
import os

from fastapi.testclient import TestClient

from halden.api.app import create_app
from halden.config.settings import Settings

QUESTIONS = [
    "What does E-17 mean on the PX-200, and what should I do?",
    "How accurate is the HX-4410, and which firmware fixes drift?",
    "What is the name of Halden's CEO?",
]


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY and HALDEN_MODEL to run this.")
        return
    settings = Settings(auth_mode="dev")
    with TestClient(create_app(settings)) as client:
        for q in QUESTIONS:
            r = client.post("/v1/ask", json={"question": q})
            a = r.json()
            print(
                f"Q: {q}\nHTTP {r.status_code}  "
                f"abstained={a['abstained']}  "
                f"citations_ok={a['citation_check_ok']}  "
                f"tokens={a['input_tokens']}+{a['output_tokens']}  "
                f"ms={a['timings_ms']}"
            )
            print(f"A: {a['answer']}")
            for c in a["citations"]:
                path = " > ".join(c["heading_path"])
                print(f"   [{c['label']}] {c['title'][:40]} > {path}")
            print()
        with client.stream(
            "POST", "/v1/ask/stream", json={"question": QUESTIONS[0]}
        ) as r:
            events = [
                json.loads(x[5:])
                for x in r.iter_lines()
                if x.startswith("data:")
            ]
        tokens = [e for e in events if e["type"] == "token"]
        done = events[-1]
        print(
            f"stream: {len(events)} events, {len(tokens)} token "
            f"events, done.citations="
            f"{[c['label'] for c in done['citations']]}, "
            f"citation_check_ok={done['citation_check_ok']}"
        )


if __name__ == "__main__":
    main()
