"""Timeouts and retries around any LLM client."""

import asyncio
import logging
import random
from collections.abc import AsyncIterator, Sequence

from halden.ports.llm import (
    LLMClient,
    LLMError,
    LLMResponse,
    LLMStreamer,
    Message,
    ToolSpec,
)

log = logging.getLogger(__name__)


# [start:resilient]
class ResilientLLM:
    """Adds a per-attempt timeout and retries with exponential
    backoff and full jitter. Only retryable errors are retried:
    a malformed request fails fast instead of failing slowly.

    Full jitter (sleep a random time up to the backoff ceiling)
    spreads retries out, so a fleet of clients hit by the same
    overload does not retry in synchronized waves.
    """

    def __init__(
        self,
        inner: LLMClient,
        *,
        attempts: int = 3,
        timeout_s: float = 30.0,
        base_delay_s: float = 0.5,
        max_delay_s: float = 8.0,
    ) -> None:
        self._inner = inner
        self._attempts = attempts
        self._timeout_s = timeout_s
        self._base = base_delay_s
        self._max = max_delay_s

    def _delay(self, attempt: int) -> float:
        ceiling = min(self._max, self._base * 2 ** (attempt - 1))
        return random.uniform(0, ceiling)

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        tools: Sequence[ToolSpec] = (),
        max_tokens: int = 1024,
        temperature: float | None = None,
    ) -> LLMResponse:
        for attempt in range(1, self._attempts + 1):
            try:
                async with asyncio.timeout(self._timeout_s):
                    return await self._inner.complete(
                        messages,
                        system=system,
                        tools=tools,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
            except (LLMError, TimeoutError) as exc:
                retryable = (
                    isinstance(exc, TimeoutError) or exc.retryable
                )
                if not retryable or attempt == self._attempts:
                    raise
                delay = self._delay(attempt)
                log.warning(
                    "LLM attempt %d failed (%s); retry in %.1fs",
                    attempt,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
        raise AssertionError("unreachable")

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float | None = None,
    ) -> AsyncIterator[str | LLMResponse]:
        """Retry only until the first token arrives: once text has
        reached the user, a silent retry would repeat it."""
        if not isinstance(self._inner, LLMStreamer):
            raise TypeError("inner client cannot stream")
        for attempt in range(1, self._attempts + 1):
            started = False
            try:
                async with asyncio.timeout(self._timeout_s):
                    async for item in self._inner.stream(
                        messages,
                        system=system,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    ):
                        started = True
                        yield item
                return
            except (LLMError, TimeoutError) as exc:
                retryable = (
                    isinstance(exc, TimeoutError) or exc.retryable
                )
                if (
                    started
                    or not retryable
                    or attempt == self._attempts
                ):
                    raise
                await asyncio.sleep(self._delay(attempt))


# [end:resilient]
