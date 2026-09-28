"""A scripted LLMClient for tests, examples, and offline runs."""

from collections.abc import AsyncIterator, Sequence

from halden.ports.llm import (
    LLMResponse,
    Message,
    TextBlock,
    ToolSpec,
    Usage,
)


def text_response(text: str) -> LLMResponse:
    return LLMResponse(
        content=[TextBlock(text=text)],
        stop_reason="end_turn",
        usage=Usage(input_tokens=0, output_tokens=0),
        model="scripted",
    )


class ScriptedLLM:
    """Returns pre-written responses in order and records each call.

    Deterministic by design: tests assert on what the application
    *sent* to the model, which is the part we control.
    """

    def __init__(self, script: Sequence[LLMResponse | str]) -> None:
        self._script = [
            text_response(s) if isinstance(s, str) else s
            for s in script
        ]
        self.calls: list[dict[str, object]] = []

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        tools: Sequence[ToolSpec] = (),
        max_tokens: int = 1024,
        temperature: float | None = None,
    ) -> LLMResponse:
        self.calls.append(
            {
                "messages": list(messages),
                "system": system,
                "tools": list(tools),
            }
        )
        if not self._script:
            raise AssertionError("ScriptedLLM ran out of responses")
        return self._script.pop(0)

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float | None = None,
    ) -> AsyncIterator[str | LLMResponse]:
        """Stream the next scripted response word by word."""
        response = await self.complete(
            messages, system=system, max_tokens=max_tokens
        )
        words = response.text.split(" ")
        for i, word in enumerate(words):
            yield word if i == 0 else " " + word
        yield response
