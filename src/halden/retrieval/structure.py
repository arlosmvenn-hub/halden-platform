"""Structure-aware retrieval: parent expansion and an entity graph."""

import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

from halden.ingestion.chunking import Chunk


# [start:parents]
def expand_to_parents(
    hits: Sequence[Chunk], corpus: Sequence[Chunk], depth: int
) -> list[str]:
    """Retrieve small, read big.

    For each hit, return the full text of its ancestor section at
    ``depth`` levels of heading (1 = "4 Calibration", 2 = "4.1 Zero
    adjustment"), assembled from all chunks under that heading in
    document order. Parents are deduplicated, keeping hit order.
    """
    parents: list[str] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for hit in hits:
        key = (hit.doc_id, tuple(hit.heading_path[:depth]))
        if key in seen:
            continue
        seen.add(key)
        members = sorted(
            (
                c
                for c in corpus
                if c.doc_id == hit.doc_id
                and tuple(c.heading_path[:depth]) == key[1]
            ),
            key=lambda c: c.ordinal,
        )
        parents.append("\n\n".join(c.text for c in members))
    return parents


# [end:parents]

# [start:graph]
PART = re.compile(r"\b(?:PX|FM|HX)-\d{3,4}\b")
REPLACED = re.compile(
    r"\b(?P<old>(?:PX|FM|HX)-\d{3,4})\b[^.]*?"
    r"replacement is the (?P<new>(?:PX|FM|HX)-\d{3,4})\b"
)


@dataclass
class EntityGraph:
    """Part numbers as nodes; 'replaced_by' edges; the chunks that
    mention each part. Built from text with explicit patterns."""

    edges: dict[str, set[tuple[str, str]]] = field(
        default_factory=lambda: defaultdict(set)
    )
    mentions: dict[str, set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )

    @classmethod
    def build(cls, chunks: Sequence[Chunk]) -> "EntityGraph":
        g = cls()
        for c in chunks:
            for part in PART.findall(c.text):
                g.mentions[part].add(c.chunk_id)
            for m in REPLACED.finditer(c.text):
                g.edges[m["old"]].add(("replaced_by", m["new"]))
                g.edges[m["new"]].add(("replaces", m["old"]))
        return g

    def neighbors(self, part: str, relation: str) -> list[str]:
        return sorted(
            t for r, t in self.edges.get(part, ()) if r == relation
        )


RELATION_CUES = {
    "replaced_by": ("replacement", "successor", "newer", "replaced"),
}


def expand_query(question: str, graph: EntityGraph) -> str:
    """Add entities one hop away when the question asks about a
    relationship ("the HX-441's replacement") rather than the part
    itself. Deterministic: no model call, fully testable."""
    extra: list[str] = []
    q = question.lower()
    for part in PART.findall(question):
        for relation, cues in RELATION_CUES.items():
            if any(cue in q for cue in cues):
                extra += graph.neighbors(part, relation)
    return question if not extra else f"{question} {' '.join(extra)}"


# [end:graph]
