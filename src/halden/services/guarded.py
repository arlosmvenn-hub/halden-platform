"""Wrappers around the ask use case: an answer cache in front,
a token budget behind it. Each implements AskUseCase, so they
stack in any order the composition root chooses."""

import uuid
from collections.abc import AsyncIterator
from typing import Any

from halden.config.settings import Settings
from halden.generation.grounded import PROMPT_VERSION
from halden.observability.cost import cost_usd
from halden.ports.llm import Usage
from halden.security.identity import Identity
from halden.security.output import sanitize
from halden.services.ask import Answer, AskUseCase
from halden.store.cache import AnswerCache, cache_key
from halden.store.usage import UsageLedger


def fingerprint(settings: Settings) -> str:
    """The pipeline settings that change answers."""
    return "|".join(
        [
            settings.model or "",
            PROMPT_VERSION,
            settings.retrieval_mode,
            settings.reranker_model or "",
            str(settings.context_budget_tokens),
        ]
    )


# [start:cached]
class CachedAsk:
    def __init__(
        self, inner: AskUseCase, cache: AnswerCache, settings: Settings
    ) -> None:
        self._inner = inner
        self._cache = cache
        self._fp = fingerprint(settings)

    async def ask(
        self,
        question: str,
        identity: Identity,
        conversation_id: uuid.UUID | None = None,
    ) -> Answer:
        if conversation_id is not None:  # depends on history: skip
            return await self._inner.ask(
                question, identity, conversation_id
            )
        generation = await self._cache.generation()
        who = identity
        key = cache_key(
            who.tenant, question, who.groups, generation, self._fp
        )
        hit = await self._cache.get(key, who.groups, who.tenant)
        if hit is not None:
            return hit
        answer = await self._inner.ask(question, identity)
        if answer.citation_check_ok:  # never cache a withheld answer
            await self._cache.put(key, who.groups, who.tenant, answer)
        return answer

    def ask_stream(
        self, question: str, identity: Identity
    ) -> AsyncIterator[dict[str, Any]]:
        return self._inner.ask_stream(question, identity)


# [end:cached]


# [start:budgeted]
class BudgetedAsk:
    def __init__(
        self, inner: AskUseCase, ledger: UsageLedger, model: str
    ) -> None:
        self._inner = inner
        self._ledger = ledger
        self._model = model

    async def ask(
        self,
        question: str,
        identity: Identity,
        conversation_id: uuid.UUID | None = None,
    ) -> Answer:
        await self._ledger.check(identity.user_id)  # may raise
        answer = await self._inner.ask(
            question, identity, conversation_id
        )
        await self._record(
            identity, answer.input_tokens, answer.output_tokens
        )
        return answer

    async def ask_stream(
        self, question: str, identity: Identity
    ) -> AsyncIterator[dict[str, Any]]:
        await self._ledger.check(identity.user_id)
        async for event in self._inner.ask_stream(question, identity):
            if event["type"] == "done":
                await self._record(
                    identity,
                    event.get("input_tokens", 0),
                    event.get("output_tokens", 0),
                )
            yield event

    async def _record(
        self, identity: Identity, tokens_in: int, tokens_out: int
    ) -> None:
        usage = Usage(input_tokens=tokens_in, output_tokens=tokens_out)
        await self._ledger.record(
            identity.user_id,
            tokens_in,
            tokens_out,
            cost_usd(self._model, usage) or 0.0,
        )


# [end:budgeted]


# [start:sanitized]
class SanitizedAsk:
    """Strip images and unknown links from every answer. Streamed
    tokens can't be fixed after the fact, so the 'done' event
    carries the sanitized full answer and clients must render
    streamed text as plain text until it arrives (Chapter 26)."""

    def __init__(
        self, inner: AskUseCase, hosts: frozenset[str]
    ) -> None:
        self._inner = inner
        self._hosts = hosts

    async def ask(
        self,
        question: str,
        identity: Identity,
        conversation_id: uuid.UUID | None = None,
    ) -> Answer:
        answer = await self._inner.ask(
            question, identity, conversation_id
        )
        clean = sanitize(answer.answer, self._hosts)
        if not clean.removed:
            return answer
        return answer.model_copy(
            update={"answer": clean.text, "sanitized": True}
        )

    async def ask_stream(
        self, question: str, identity: Identity
    ) -> AsyncIterator[dict[str, Any]]:
        parts: list[str] = []
        async for event in self._inner.ask_stream(question, identity):
            if event["type"] == "token":
                parts.append(event["text"])
            elif event["type"] == "done":
                clean = sanitize("".join(parts), self._hosts)
                event = {
                    **event,
                    "answer": clean.text,
                    "sanitized": bool(clean.removed),
                }
            yield event


# [end:sanitized]
