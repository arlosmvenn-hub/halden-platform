"""The Part II-III demo index lives in its own schema, "demo", so the
application built in Part IV (schema "public") never disturbs it."""

import os
from urllib.parse import quote

BASE_DSN = os.environ.get(
    "HALDEN_DSN", "postgresql://halden:halden@localhost/halden"
)
_SEP = "&" if "?" in BASE_DSN else "?"
DEMO_DSN = f"{BASE_DSN}{_SEP}options=" + quote(
    "-csearch_path=demo,public", safe=""
)
