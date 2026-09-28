"""Aggregate case results, render reports, and gate on baselines."""

import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path

from halden.eval.metrics import mean
from halden.eval.runner import CaseResult

Pick = Callable[[CaseResult], float | None]

# [start:summary]
METRICS: dict[str, Pick] = {
    "hit@5": lambda r: r.hit,
    "precision@5": lambda r: r.precision,
    "recall@5": lambda r: r.recall,
    "mrr": lambda r: r.rr,
    "ndcg@5": lambda r: r.ndcg,
    "context_recall": lambda r: r.context_recall,
    "abstention_acc": lambda r: _f(r.abstention_ok),
    "false_abstain": lambda r: (
        _f(r.abstained) if r.answerable else None
    ),
    "missed_abstain": lambda r: (
        None if r.answerable else _f(r.abstained, invert=True)
    ),
    "citations_valid": lambda r: _f(r.citations_valid),
    "citation_support": lambda r: _f(r.citation_support),
    "faithfulness": lambda r: r.faithfulness,
    "correct": lambda r: (
        _f(r.correctness == "correct") if r.correctness else None
    ),
    "correct_or_partial": lambda r: (
        _f(r.correctness in ("correct", "partial"))
        if r.correctness
        else None
    ),
}


def _f(b: bool | None, *, invert: bool = False) -> float | None:
    return None if b is None else float(b != invert)


def summarize(results: Sequence[CaseResult]) -> dict[str, float]:
    """Mean of each metric over the cases where it applies."""
    out = {}
    for name, pick in METRICS.items():
        vals = [v for r in results if (v := pick(r)) is not None]
        if vals:
            out[name] = round(mean(vals), 3)
    return out


# [end:summary]


def by_tag(
    results: Sequence[CaseResult], metric: str
) -> dict[str, tuple[int, float]]:
    pick = METRICS[metric]
    tags = sorted({t for r in results for t in r.tags})
    out = {}
    for tag in tags:
        vals = [
            v
            for r in results
            if tag in r.tags and (v := pick(r)) is not None
        ]
        if vals:
            out[tag] = (len(vals), round(mean(vals), 3))
    return out


def table(rows: dict[str, dict[str, float]], metrics: list[str]) -> str:
    """Rows are configurations or datasets; columns are metrics."""
    name_w = max(len(n) for n in rows) + 2
    widths = [max(9, len(m) + 2) for m in metrics]
    head = "".join(
        f"{m:>{w}}" for m, w in zip(metrics, widths, strict=True)
    )
    lines = [f"{'':<{name_w}}{head}"]
    for name, summary in rows.items():
        cells = "".join(
            f"{summary.get(m, math.nan):>{w}.3f}"
            for m, w in zip(metrics, widths, strict=True)
        )
        lines.append(f"{name:<{name_w}}{cells}")
    return "\n".join(lines)


# [start:gate]
def gate(
    current: dict[str, float],
    baseline: dict[str, float],
    *,
    max_drop: float = 0.02,
    floors: dict[str, float] | None = None,
) -> list[str]:
    """Failures, empty if the run may ship.

    Two rules: no metric may fall more than ``max_drop`` below the
    stored baseline (catches regressions), and some metrics have
    absolute floors (catches a baseline that was already bad).
    Metrics where lower is better are listed in LOWER_IS_BETTER.
    """
    failures = []
    for name, base in baseline.items():
        if name not in current:
            failures.append(f"{name}: missing from this run")
            continue
        now = current[name]
        worse = now - base if name in LOWER_IS_BETTER else base - now
        if worse > max_drop:
            failures.append(f"{name}: {now:.3f} vs baseline {base:.3f}")
    for name, floor in (floors or {}).items():
        if current.get(name, 0.0) < floor:
            failures.append(
                f"{name}: {current.get(name, 0.0):.3f} "
                f"below floor {floor:.3f}"
            )
    return failures


LOWER_IS_BETTER = {"false_abstain", "missed_abstain"}
# [end:gate]


def save_results(
    path: Path, results: Sequence[CaseResult], summary: dict[str, float]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "summary": summary,
                "cases": [r.model_dump() for r in results],
            },
            indent=1,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
