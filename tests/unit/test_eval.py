"""Evaluation metrics, datasets, calibration, and the CI gate."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from halden.eval.calibration import cohen_kappa, mutate_number
from halden.eval.dataset import EvalCase, missing_evidence, normalize
from halden.eval.metrics import (
    coverage,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from halden.eval.report import gate
from halden.eval.synthetic import lexical_overlap


# [start:worked]
def test_metrics_on_a_worked_example() -> None:
    # Two evidence items; ranks 2 and 4 hold them, rank 3 repeats
    # the first. Irrelevant chunks at ranks 1 and 5.
    texts = [
        "Wear safety glasses.",
        "Accuracy: ±0.075% of calibrated span.",
        "Again: ±0.075 % of calibrated span.",
        "HART needs at least 250 ohms.",
        "Revision history.",
    ]
    cov = coverage(texts, ["±0.075% of calibrated span", "250 ohms"])
    assert [sorted(c) for c in cov] == [[], [0], [], [1], []]
    assert hit_at_k(cov, 1) == 0.0 and hit_at_k(cov, 5) == 1.0
    assert precision_at_k(cov, 5) == pytest.approx(2 / 5)
    assert recall_at_k(cov, 2, 3) == 0.5
    assert recall_at_k(cov, 2, 5) == 1.0
    assert reciprocal_rank(cov) == 0.5
    assert 0.5 < ndcg_at_k(cov, 2, 5) < 0.7


# [end:worked]


def test_near_duplicate_evidence_counts_once() -> None:
    # "0.075 %" (with a space) doesn't match; normalization handles
    # case, dashes, and whitespace runs, not arbitrary spacing.
    assert coverage(["Range −50 to 250 °C"], ["-50 TO 250 °c"]) == [
        frozenset({0})
    ]


covers = st.lists(
    st.frozensets(st.integers(0, 3), max_size=3), max_size=8
)


@given(covers, st.integers(1, 8))
def test_metrics_stay_in_range(
    cov: list[frozenset[int]], k: int
) -> None:
    for value in (
        hit_at_k(cov, k),
        precision_at_k(cov, k),
        recall_at_k(cov, 4, k),
        reciprocal_rank(cov),
        ndcg_at_k(cov, 4, k),
    ):
        assert 0.0 <= value <= 1.0


@given(st.integers(1, 5), st.integers(1, 8))
def test_ideal_ranking_scores_one(n: int, k: int) -> None:
    ideal = [frozenset({i}) for i in range(n)]
    assert ndcg_at_k(ideal, n, k) == pytest.approx(1.0)
    assert recall_at_k(ideal, n, max(k, n)) == 1.0


def test_case_requires_consistent_labels() -> None:
    with pytest.raises(ValueError, match="needs evidence"):
        EvalCase(id="a", question="What is it?")
    with pytest.raises(ValueError, match="has evidence"):
        EvalCase(
            id="b", question="Who?", evidence=["x"], answerable=False
        )


def test_missing_evidence_reports_broken_labels() -> None:
    cases = [
        EvalCase(id="ok", question="Range?", evidence=["−50 to 250"]),
        EvalCase(id="bad", question="Range?", evidence=["-60 to 250"]),
    ]
    corpus = ["| 50 to 250 °C Range | − |", "| Range | −50 to 250 °C |"]
    assert missing_evidence(cases, corpus) == {"bad": ["-60 to 250"]}
    assert normalize("A–B  c") == "a-b c"


def test_number_mutation_and_kappa() -> None:
    assert mutate_number("15 days, 20 after") == "25 days, 20 after"
    assert mutate_number("±0.075% of span") == "±0.175% of span"
    assert mutate_number("no digits") is None
    truth = [True, True, False, False]
    assert cohen_kappa(truth, truth) == 1.0
    always_yes = [True] * 4
    assert cohen_kappa(always_yes, truth) == 0.0  # 50% agree, 0 kappa


def test_gate_catches_drops_and_floors() -> None:
    base = {"recall@5": 0.90, "false_abstain": 0.05}
    assert gate({"recall@5": 0.89, "false_abstain": 0.06}, base) == []
    failures = gate(
        {"recall@5": 0.85, "false_abstain": 0.10},
        base,
        floors={"recall@5": 0.88},
    )
    assert len(failures) == 3


def test_lexical_overlap() -> None:
    passage = "Submit expense reports within 30 days of travel."
    assert lexical_overlap(
        "When are expense reports due?", passage
    ) == (pytest.approx(2 / 3))
