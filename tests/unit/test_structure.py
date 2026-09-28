from halden.ingestion.chunking import Chunk
from halden.retrieval.structure import (
    EntityGraph,
    expand_query,
    expand_to_parents,
)


def c(cid: str, doc: str, n: int, path: list[str], text: str) -> Chunk:
    return Chunk(
        chunk_id=cid,
        doc_id=doc,
        ordinal=n,
        heading_path=path,
        kind="text",
        text=text,
        embed_text=text,
        token_count=1,
    )


CORPUS = [
    c("a:0", "a", 0, ["4 Cal", "4.1 Zero"], "zero one"),
    c("a:1", "a", 1, ["4 Cal", "4.1 Zero"], "zero two"),
    c("a:2", "a", 2, ["4 Cal", "4.2 Span"], "span"),
    c("a:3", "a", 3, ["5 HART"], "hart"),
]


def test_parents_at_each_depth() -> None:
    hit = [CORPUS[1]]
    assert expand_to_parents(hit, CORPUS, 2) == ["zero one\n\nzero two"]
    assert expand_to_parents(hit, CORPUS, 1) == [
        "zero one\n\nzero two\n\nspan"
    ]


def test_parents_are_deduplicated_in_hit_order() -> None:
    hits = [CORPUS[3], CORPUS[0], CORPUS[1]]
    assert expand_to_parents(hits, CORPUS, 2) == [
        "hart",
        "zero one\n\nzero two",
    ]


def test_graph_edges_and_expansion() -> None:
    g = EntityGraph.build(
        [
            c(
                "s:0",
                "s",
                0,
                [],
                "The HX-441 is discontinued; its "
                "replacement is the HX-4410.",
            ),
        ]
    )
    assert g.neighbors("HX-441", "replaced_by") == ["HX-4410"]
    assert g.neighbors("HX-4410", "replaces") == ["HX-441"]
    q = "accuracy of the HX-441's replacement"
    assert expand_query(q, g).endswith("HX-4410")
    # No relationship cue: the question is about the part itself.
    assert expand_query("HX-441 accuracy", g) == "HX-441 accuracy"
