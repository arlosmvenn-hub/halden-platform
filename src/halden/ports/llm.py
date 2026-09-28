"""The LLM port: a provider-neutral contract for chat models.

Everything else in the platform talks to this interface, never to a
vendor SDK. Adapters in ``halden.adapters`` translate to and from
each provider's wire format.
"""

from collections.abc import AsyncIterator, Sequence
from typing import Annotated, Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field


# [start:blocks]
class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ToolUseBlock(BaseModel):
    """The model asking the application to run a tool."""

    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any]


class ToolResultBlock(BaseModel):
    """The application reporting a tool's outcome to the model."""

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str
    is_error: bool = False


ContentBlock = Annotated[
    TextBlock | ToolUseBlock | ToolResultBlock,
    Field(discriminator="type"),
]


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: list[ContentBlock]

    @classmethod
    def user(cls, text: str) -> "Message":
        return cls(role="user", content=[TextBlock(text=text)])


# [end:blocks]
# [start:response]
class ToolSpec(BaseModel):
    """A tool the model may request. The schema is JSON Schema."""

    name: str
    description: str
    input_schema: dict[str, Any]


class Usage(BaseModel):
    input_tokens: int = 0  # uncached input only (Anthropic's meaning)
    output_tokens: int = 0
    cache_read_input_tokens: int = 0  # served from the prompt cache
    cache_creation_input_tokens: int = 0  # written to the cache

    @property
    def total_input_tokens(self) -> int:
        return (
            self.input_tokens
            + self.cache_read_input_tokens
            + self.cache_creation_input_tokens
        )


StopReason = Literal[
    "end_turn", "tool_use", "max_tokens", "refusal", "other"
]


class LLMResponse(BaseModel):
    content: list[ContentBlock]
    stop_reason: StopReason
    usage: Usage
    model: str

    @property
    def text(self) -> str:
        return "".join(
            b.text for b in self.content if isinstance(b, TextBlock)
        )

    @property
    def tool_calls(self) -> list[ToolUseBlock]:
        return [b for b in self.content if isinstance(b, ToolUseBlock)]


# [end:response]
# [start:protocol]
class LLMError(Exception):
    """A failed model call. ``retryable`` guides the caller."""

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


class LLMClient(Protocol):
    async def complete(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        tools: Sequence[ToolSpec] = (),
        max_tokens: int = 1024,
        temperature: float | None = None,
    ) -> LLMResponse: ...


# [end:protocol]


# [start:streamer]
@runtime_checkable
class LLMStreamer(Protocol):
    """Streaming generation: yields text deltas as they arrive, then
    exactly one final LLMResponse with the full text, stop reason,
    and token usage. Used for answers, where the user watches."""

    def stream(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float | None = None,
    ) -> AsyncIterator[str | LLMResponse]: ...


# [end:streamer]
