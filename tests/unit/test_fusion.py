import pytest

from halden.retrieval.fusion import reciprocal_rank_fusion


def test_items_ranked_well_by_both_lists_win() -> None:
    dense = ["a", "b", "c", "d"]
    lexical = ["e", "b", "f"]
    fused = [cid for cid, _ in reciprocal_rank_fusion([dense, lexical])]
    # b is 2nd in both lists: 2/62 beats a's and e's single 1/61.
    assert fused[0] == "b"
    assert set(fused) == {"a", "b", "c", "d", "e", "f"}


def test_rrf_score_formula() -> None:
    fused = dict(reciprocal_rank_fusion([["x"], ["y", "x"]], k=60))
    assert fused["x"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["y"] == pytest.approx(1 / 61)


def test_weights_must_match_rankings() -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], weights=[1.0, 2.0])
