"""A connector for a folder of files: list, diff, enqueue."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from halden.ingestion.sync import SyncPlan, plan_sync
from halden.store.documents import DocumentStore
from halden.store.jobs import JobQueue

SUPPORTED = {".md", ".pdf"}

# [start:pipeline]
# Part of every document's version marker. Bump it whenever parsing
# or chunking changes, so unchanged files are re-processed too; the
# content-hash check then re-embeds only documents whose parsed
# text actually changed.
PIPELINE_VERSION = "2"
# [end:pipeline]


@dataclass(frozen=True)
class SourceFile:
    doc_id: str
    path: Path
    version: str  # sha256 of the bytes + pipeline version
    acl: list[str]


def scan_folder(folder: Path, system: str) -> dict[str, SourceFile]:
    """A complete listing. Permissions come from acl.json:
    {"default": [...], "files": {"name.pdf": [...]}}."""
    acl_file = folder / "acl.json"
    acl_cfg = (
        json.loads(acl_file.read_text()) if acl_file.exists() else {}
    )
    default = acl_cfg.get("default", ["everyone"])
    out = {}
    for path in sorted(folder.iterdir()):
        if path.suffix not in SUPPORTED:
            continue
        doc_id = f"{system}:{path.stem}"
        out[doc_id] = SourceFile(
            doc_id=doc_id,
            path=path.resolve(),
            version=(
                hashlib.sha256(path.read_bytes()).hexdigest()
                + f"+p{PIPELINE_VERSION}"
            ),
            acl=acl_cfg.get("files", {}).get(path.name, default),
        )
    return out


class DeletionCapExceeded(Exception):
    pass


# [start:sync]
async def sync_folder(
    folder: Path,
    system: str,
    store: DocumentStore,
    queue: JobQueue,
    *,
    max_delete_share: float = 0.5,
) -> SyncPlan:
    listing = scan_folder(folder, system)  # complete, or it raises
    indexed = await store.versions(system)
    plan = plan_sync(
        indexed, {d: f.version for d, f in listing.items()}
    )
    # A listing that suddenly lacks most documents is more likely a
    # mount or permissions problem than a mass deletion.
    if len(indexed) >= 4 and len(plan.delete) > max_delete_share * len(
        indexed
    ):
        raise DeletionCapExceeded(
            f"refusing to delete {len(plan.delete)} of {len(indexed)}"
        )
    for doc_id in plan.upsert:
        f = listing[doc_id]
        await queue.enqueue(
            "ingest_file",
            {
                "doc_id": doc_id,
                "system": system,
                "path": str(f.path),
                "acl": f.acl,
                "version": f.version,
            },
            dedupe_key=f"{doc_id}@{f.version}",
        )
    for doc_id in plan.delete:
        await queue.enqueue(
            "delete_document", {"doc_id": doc_id}, dedupe_key=doc_id
        )
    return plan


# [end:sync]
