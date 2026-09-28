"""Create LLM clients from environment variables.

    ANTHROPIC_API_KEY    the key (never put it in code or files
                         that are committed)
    HALDEN_MODEL         model for answer generation
    HALDEN_MODEL_SMALL   cheaper model for query rewriting, routing

Returns None when no key is set, so examples can fall back to an
offline mode.
"""

import os
from typing import Literal

from halden.adapters.anthropic_llm import AnthropicLLM
from halden.ports.llm import LLMClient


def llm_from_env(
    role: Literal["answer", "small"] = "answer",
) -> LLMClient | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    var = "HALDEN_MODEL" if role == "answer" else "HALDEN_MODEL_SMALL"
    model = os.environ.get(var)
    if not model:
        raise SystemExit(f"Set {var} to a model name")
    return AnthropicLLM(key, model)
