"""LLMClient adapter for the Anthropic Messages API over HTTP."""

import json
from collections.abc import AsyncIterator, Sequence
from typing import Any, cast

import httpx

from halden.ports.llm import (
    ContentBlock,
    LLMError,
    LLMResponse,
    Message,
    StopReason,
    TextBlock,
    ToolSpec,
    ToolUseBlock,
    Usage,
)

API_VERSION = "2023-06-01"
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}
KNOWN_STOPS: set[str] = {
    "end_turn",
    "tool_use",
    "max_tokens",
    "refusal",
}


def _stop_reason(raw: str | None) -> StopReason:
    """Map provider stop reasons onto the port's small fixed set."""
    return cast(StopReason, raw) if raw in KNOWN_STOPS else "other"


USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)


def _usage_counts(raw: dict[str, Any]) -> dict[str, int]:
    return {f: int(raw.get(f) or 0) for f in USAGE_FIELDS}


class AnthropicLLM:
    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str = "https://api.anthropic.com",
        timeout_s: float = 60.0,
        http: httpx.AsyncClient | None = None,
        prompt_cache: bool = False,
    ) -> None:
        self._model = model
        self._prompt_cache = prompt_cache
        self._http = http or httpx.AsyncClient(
            base_url=base_url, timeout=timeout_s
        )
        self._headers = {
            "x-api-key": api_key,
            "anthropic-version": API_VERSION,
        }

    # [start:cache]
    def _system(self, system: str) -> str | list[dict[str, Any]]:
        """With prompt caching on, mark the end of the system prompt
        as a cache breakpoint: everything up to it (tools, system)
        is cached for five minutes and re-read at a tenth of the
        input price. The question, after it, is never cached."""
        if not self._prompt_cache:
            return system
        return [
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    # [end:cache]
    # [start:complete]
    async def complete(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        tools: Sequence[ToolSpec] = (),
        max_tokens: int = 1024,
        temperature: float | None = None,
    ) -> LLMResponse:
        body: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": [m.model_dump() for m in messages],
        }
        # Some newer models reject sampling parameters outright,
        # so send temperature only when the caller asks for it.
        if temperature is not None:
            body["temperature"] = temperature
        if system:
            body["system"] = self._system(system)
        if tools:
            body["tools"] = [t.model_dump() for t in tools]

        try:
            resp = await self._http.post(
                "/v1/messages", json=body, headers=self._headers
            )
        except httpx.TransportError as exc:
            raise LLMError(str(exc), retryable=True) from exc

        if resp.status_code != 200:
            raise LLMError(
                f"HTTP {resp.status_code}: {resp.text[:300]}",
                retryable=resp.status_code in RETRYABLE_STATUS,
            )
        return self._parse(resp.json())

    # [end:complete]
    # [start:stream]
    async def stream(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float | None = None,
    ) -> AsyncIterator[str | LLMResponse]:
        """Server-sent events: message_start carries input usage,
        content_block_delta carries text, message_delta carries the
        stop reason and output usage."""
        body: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": [m.model_dump() for m in messages],
            "stream": True,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if system:
            body["system"] = self._system(system)
        parts: list[str] = []
        usage: dict[str, int] = {}
        stop = "other"
        try:
            async with self._http.stream(
                "POST", "/v1/messages", json=body, headers=self._headers
            ) as resp:
                if resp.status_code != 200:
                    text = (await resp.aread()).decode()[:300]
                    raise LLMError(
                        f"HTTP {resp.status_code}: {text}",
                        retryable=resp.status_code in RETRYABLE_STATUS,
                    )
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    event = json.loads(line[5:])
                    kind = event.get("type")
                    if kind == "message_start":
                        u = event["message"].get("usage", {})
                        usage.update(_usage_counts(u))
                    elif kind == "content_block_delta":
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta":
                            parts.append(delta["text"])
                            yield delta["text"]
                    elif kind == "message_delta":
                        stop = event["delta"].get("stop_reason") or stop
                        u = event.get("usage", {})
                        usage["output_tokens"] = u.get(
                            "output_tokens", 0
                        )
                    elif kind == "error":
                        err = event.get("error", {})
                        raise LLMError(
                            str(err.get("message", err)),
                            retryable=err.get("type")
                            in ("overloaded_error", "api_error"),
                        )
        except httpx.TransportError as exc:
            raise LLMError(str(exc), retryable=True) from exc
        yield LLMResponse(
            content=[TextBlock(text="".join(parts))],
            stop_reason=_stop_reason(stop),
            usage=Usage(**usage),
            model=self._model,
        )

    # [end:stream]
    def _parse(self, data: dict[str, Any]) -> LLMResponse:
        blocks: list[ContentBlock] = []
        for raw in data["content"]:
            if raw["type"] == "text":
                blocks.append(TextBlock(text=raw["text"]))
            elif raw["type"] == "tool_use":
                blocks.append(
                    ToolUseBlock(
                        id=raw["id"],
                        name=raw["name"],
                        input=raw["input"],
                    )
                )
            # Other block types (e.g. thinking) are ignored here.
        stop_reason = _stop_reason(data.get("stop_reason"))
        usage = data.get("usage", {})
        return LLMResponse(
            content=blocks,
            stop_reason=stop_reason,
            usage=Usage(**_usage_counts(usage)),
            model=data.get("model", self._model),
        )

    async def aclose(self) -> None:
        await self._http.aclose()
