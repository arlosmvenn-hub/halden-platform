"""Turn ranked chunks into a labeled, budgeted prompt context."""

import hashlib
import math
import re
from collections import Counter
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field

from halden.retrieval.pg_retriever import RetrievedChunk

TokenCounter = Callable[[str], int]


def conservative_token_estimate(text: str) -> int:
    """Over-estimate on purpose: ~3 characters per token.

    Budgets must not overflow, so when the generation model's own
    tokenizer is unavailable, err on the side of too many tokens.
    """
    return math.ceil(len(text) / 3)


# [start:model]
@dataclass(frozen=True)
class Source:
    label: int  # the [n] the model cites
    chunk_id: str
    title: str
    heading_path: list[str]
    text: str
    untrusted: bool = False  # written outside Halden (Ch 26)


@dataclass(frozen=True)
class Context:
    sources: list[Source]
    tokens: int
    dropped: list[str] = field(default_factory=list)

    def render(self) -> str:
        return "\n\n".join(
            f"[{s.label}]{UNTRUSTED if s.untrusted else ''} "
            f"{' > '.join([s.title, *s.heading_path])}\n{s.text}"
            for s in self.sources
        )


UNTRUSTED = " (untrusted: written outside Halden)"


# [end:model]


def _fingerprint(text: str) -> str:
    norm = re.sub(r"\W+", " ", text.lower()).strip()
    return hashlib.sha1(norm.encode()).hexdigest()


# [start:build]
def build_context(
    chunks: Sequence[RetrievedChunk],
    *,
    budget_tokens: int,
    count: TokenCounter = conservative_token_estimate,
    max_per_doc: int = 3,
    untrusted_systems: Collection[str] = (),
) -> Context:
    """Select chunks in rank order until the token budget is spent.

    - exact duplicates (after normalizing case and punctuation)
      are skipped: the same paragraph in two manual revisions
      adds cost, not evidence;
    - the per-document cap is soft: the first pass admits at most
      ``max_per_doc`` chunks per document so other documents get a
      chance, then leftover budget is backfilled with the capped
      chunks in rank order, so a question answered by a single
      document still gets all of it;
    - a chunk that does not fit is dropped whole, never cut, so
      no source is ever quoted mid-sentence.
    """
    seen: set[str] = set()
    per_doc: Counter[str] = Counter()
    chosen: list[RetrievedChunk] = []
    capped: list[RetrievedChunk] = []
    dropped: list[str] = []
    used = 0

    def cost(c: RetrievedChunk) -> int:
        return count(c.text) + 20  # label and heading line

    for c in chunks:
        fp = _fingerprint(c.text)
        if fp in seen:
            dropped.append(c.chunk_id)
            continue
        if per_doc[c.doc_id] >= max_per_doc:
            capped.append(c)
            continue
        if used + cost(c) > budget_tokens:
            dropped.append(c.chunk_id)
            continue
        seen.add(fp)
        per_doc[c.doc_id] += 1
        chosen.append(c)
        used += cost(c)
    for c in capped:  # backfill leftover budget
        fp = _fingerprint(c.text)
        if fp not in seen and used + cost(c) <= budget_tokens:
            seen.add(fp)
            chosen.append(c)
            used += cost(c)
        else:
            dropped.append(c.chunk_id)
    rank = {c.chunk_id: i for i, c in enumerate(chunks)}
    chosen.sort(key=lambda c: rank[c.chunk_id])  # keep rank order
    sources = [
        Source(
            i,
            c.chunk_id,
            c.title,
            c.heading_path,
            c.text,
            untrusted=c.doc_id.split(":")[0] in untrusted_systems,
        )
        for i, c in enumerate(chosen, start=1)
    ]
    return Context(sources=sources, tokens=used, dropped=dropped)


# [end:build]
