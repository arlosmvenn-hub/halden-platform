"""Every example program must at least import cleanly.

Regression guard: an automatic lint fix once deleted a re-exported
name that several examples used, and nothing else noticed.
"""

import importlib
from pathlib import Path

import pytest

EXAMPLES = sorted(
    p.stem
    for p in (Path(__file__).parents[2] / "examples").glob("*.py")
    if p.stem != "__init__"
)


@pytest.mark.parametrize("module", EXAMPLES)
def test_example_imports(module: str) -> None:
    importlib.import_module(f"examples.{module}")
