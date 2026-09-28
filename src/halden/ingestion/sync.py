"""Decide what an ingestion run must do, as a pure function."""

from dataclasses import dataclass, field


# [start:plan]
@dataclass(frozen=True)
class SyncPlan:
    upsert: list[str] = field(default_factory=list)
    delete: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)


def plan_sync(
    indexed: dict[str, str], source: dict[str, str]
) -> SyncPlan:
    """Compare doc_id -> content_hash maps.

    ``indexed`` is what the index holds now; ``source`` is what a
    *complete* listing of the source system reports. Deletions can
    only be inferred from a complete listing; a partial or failed
    listing must never be passed here, or live documents would be
    deleted.
    """
    upsert = sorted(d for d, h in source.items() if indexed.get(d) != h)
    delete = sorted(d for d in indexed if d not in source)
    unchanged = sorted(
        d for d, h in source.items() if indexed.get(d) == h
    )
    return SyncPlan(upsert=upsert, delete=delete, unchanged=unchanged)


# [end:plan]
