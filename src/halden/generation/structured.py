"""Get validated, typed objects out of a probabilistic model."""

import json
import re
from collections.abc import Sequence

from pydantic import BaseModel, ValidationError

from halden.ports.llm import LLMClient, Message, TextBlock

FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class StructuredOutputError(Exception):
    """The model never produced output matching the schema."""


def _schema_instruction(model: type[BaseModel]) -> str:
    schema = json.dumps(model.model_json_schema())
    return (
        "Respond with a single JSON object and nothing else. "
        f"It must validate against this JSON Schema:\n{schema}"
    )


async def complete_structured[T: BaseModel](
    llm: LLMClient,
    output_type: type[T],
    messages: Sequence[Message],
    *,
    system: str = "",
    max_attempts: int = 2,
) -> T:
    """Ask for JSON, validate it, and re-ask once with the errors.

    The model proposes; Pydantic decides. Nothing downstream ever
    sees an unvalidated object.
    """
    full_system = f"{system}\n\n{_schema_instruction(output_type)}"
    history = list(messages)
    last_error = ""
    for _ in range(max_attempts):
        response = await llm.complete(
            history, system=full_system.strip()
        )
        raw = FENCE.sub("", response.text).strip()
        try:
            return output_type.model_validate_json(raw)
        except ValidationError as exc:
            last_error = exc.json(include_url=False)
        history += [
            Message(role="assistant", content=[TextBlock(text=raw)]),
            Message.user(
                "That JSON was invalid. Errors:\n"
                f"{last_error}\nReturn only corrected JSON."
            ),
        ]
    raise StructuredOutputError(last_error)
