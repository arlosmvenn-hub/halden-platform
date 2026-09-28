"""Output sanitizing and trust labels."""

from halden.generation.context import build_context
from halden.retrieval.pg_retriever import RetrievedChunk
from halden.security.output import sanitize

HOSTS = frozenset({"halden.example"})


# [start:sanitize-tests]
def test_images_are_removed_whatever_the_host() -> None:
    text = "Done [1]. ![s](https://collector.example.invalid/p?q=x)"
    out = sanitize(text, HOSTS)
    assert out.text == "Done [1]."
    assert out.removed


def test_links_to_unknown_hosts_lose_their_target() -> None:
    text = "Sign in at [the portal](https://evil.example.invalid/x)."
    assert sanitize(text, HOSTS).text == "Sign in at the portal."


def test_allowed_links_and_subdomains_survive() -> None:
    text = "See [docs](https://support.halden.example/px200)."
    assert sanitize(text, HOSTS).text == text


def test_bare_urls_and_html() -> None:
    text = 'Go to https://x.example.invalid/a <img src="https://x/y">'
    assert sanitize(text, HOSTS).text == "Go to [link removed]"


# [end:sanitize-tests]


def chunk(doc_id: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{doc_id}:0",
        doc_id=doc_id,
        title=doc_id,
        heading_path=[],
        text=text,
        score=1.0,
    )


def test_untrusted_sources_are_labeled_in_the_prompt() -> None:
    ctx = build_context(
        [chunk("manuals:px", "Manual."), chunk("tickets:9", "Note.")],
        budget_tokens=500,
        untrusted_systems={"tickets"},
    )
    lines = ctx.render().splitlines()
    assert lines[0] == "[1] manuals:px"
    assert (
        lines[3] == "[2] (untrusted: written outside Halden) tickets:9"
    )
