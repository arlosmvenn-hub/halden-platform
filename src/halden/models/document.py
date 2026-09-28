"""The canonical document model every parser produces."""

import hashlib
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


# [start:model]
class SourceRef(BaseModel):
    """Where a document came from, precisely enough to fetch it
    again and to show a user the original."""

    system: str  # e.g. "manuals", "wiki", "tickets"
    uri: str  # stable location in that system
    version: str | None = None  # source's own revision marker
    fetched_at: datetime


class Section(BaseModel):
    heading_path: list[str]  # ["7 Troubleshooting", "7.1 Error codes"]
    text: str
    kind: Literal["text", "table", "code"] = "text"


class Document(BaseModel):
    doc_id: str  # stable across versions
    title: str
    source: SourceRef
    sections: list[Section]
    acl_groups: list[str] = Field(default_factory=list)
    language: str = "en"
    metadata: dict[str, str] = Field(default_factory=dict)

    def content_hash(self) -> str:
        """Changes only when retrievable content or access changes."""
        h = hashlib.sha256()
        h.update(self.title.encode())
        for s in self.sections:
            h.update("\x1f".join(s.heading_path).encode())
            h.update(s.kind.encode())
            h.update(s.text.encode())
        h.update(",".join(sorted(self.acl_groups)).encode())
        return h.hexdigest()


# [end:model]
