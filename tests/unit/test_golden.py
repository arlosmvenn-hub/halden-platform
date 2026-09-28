"""Prompts are code: changing one must be a deliberate, reviewed act.

If this fails, you changed SYSTEM_PROMPT. Re-run the evaluation
suite (Chapter 23), then update tests/golden/system_prompt.txt in
the same commit so reviewers see the prompt diff.
"""

from pathlib import Path

from halden.generation.grounded import SYSTEM_PROMPT

GOLDEN = Path(__file__).parents[1] / "golden" / "system_prompt.txt"


def test_system_prompt_matches_golden_file() -> None:
    assert GOLDEN.read_text() == SYSTEM_PROMPT
