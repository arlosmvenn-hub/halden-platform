"""Run every code check the book relies on. Exit non-zero on failure.

uv run python scripts/verify.py
"""

import subprocess
import sys

CHECKS = [
    ["ruff", "format", "--check", "."],
    ["ruff", "check", "."],
    ["mypy"],
    ["pytest", "-q"],
]


def main() -> int:
    for cmd in CHECKS:
        print("$", " ".join(cmd), flush=True)
        if subprocess.run(cmd).returncode != 0:
            print("FAILED:", " ".join(cmd))
            return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
