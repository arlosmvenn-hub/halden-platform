"""Tracing at the seams: wrappers around each port, and a traced
AskService. No business logic changes to be observable."""

import json
import time
import uuid
from collections.abc import AsyncIterator, Sequence
from typing import Any

from opentelemetry.trace import SpanKind, Status, StatusCode

from halden.generation.context import Context
from halden.generation.grounded import PROMPT_VERSION
from halden.observability.cost import cost_usd
from halden.observability.telemetry import (
    ask_duration,
    input_tokens,
    llm_cost,
    llm_duration,
    output_tokens,
    pseudonym,
    redact,
    tracer,
)
from halden.ports.embedder import Embedder, Matrix, Vector
from halden.ports.llm import (
    LLMClient,
    LLMError,
    LLMResponse,
    LLMStreamer,
    Message,
    ToolSpec,
)
from halden.ports.reranker import Reranker
from halden.retrieval.pg_retriever import RetrievedChunk
from halden.security.identity import Identity
from halden.services.ask import Answer, AskService


# [start:llm]
class TracedLLM:
    """One CLIENT span per model call, named and attributed per the
    OpenTelemetry GenAI semantic conventions, plus metrics."""

    def __init__(
        self,
        inner: LLMClient,
        model: str,
        *,
        provider: str = "anthropic",
        capture_content: bool = False,
    ) -> None:
        self._inner = inner
        self._model = model
        self._base = {
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": provider,
            "gen_ai.request.model": model,
        }
        self._capture = capture_content

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        tools: Sequence[ToolSpec] = (),
        max_tokens: int = 1024,
        temperature: float | None = None,
    ) -> LLMResponse:
        with tracer.start_as_current_span(
            f"chat {self._model}",
            kind=SpanKind.CLIENT,
            attributes={
                **self._base,
                "gen_ai.request.max_tokens": max_tokens,
            },
        ) as span:
            t0 = time.perf_counter()
            try:
                resp = await self._inner.complete(
                    messages,
                    system=system,
                    tools=tools,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except LLMError as exc:
                self._failed(span, exc, t0)
                raise
            self._record(span, resp, t0, messages)
            return resp

    def _failed(self, span: Any, exc: LLMError, t0: float) -> None:
        kind = "retryable" if exc.retryable else "permanent"
        span.set_attribute("error.type", kind)
        span.set_status(Status(StatusCode.ERROR, kind))
        llm_duration.record(
            time.perf_counter() - t0,
            {**self._base, "error.type": kind},
        )

    def _record(
        self,
        span: Any,
        resp: LLMResponse,
        t0: float,
        messages: Sequence[Message],
    ) -> None:
        u = resp.usage
        span.set_attributes(
            {
                "gen_ai.response.model": resp.model,
                "gen_ai.response.finish_reasons": [resp.stop_reason],
                "gen_ai.usage.input_tokens": u.total_input_tokens,
                "gen_ai.usage.output_tokens": u.output_tokens,
                "gen_ai.usage.cache_read.input_tokens": (
                    u.cache_read_input_tokens
                ),
                "gen_ai.usage.cache_write.input_tokens": (
                    u.cache_creation_input_tokens
                ),
            }
        )
        cost = cost_usd(self._model, u)
        if cost is not None:
            span.set_attribute("halden.cost_usd", round(cost, 6))
            llm_cost.add(cost, self._base)
        if self._capture:  # opt-in, redacted, truncated
            span.set_attribute(
                "gen_ai.input.messages",
                redact(messages[-1].model_dump_json()),
            )
            span.set_attribute(
                "gen_ai.output.messages", redact(resp.text)
            )
        llm_duration.record(time.perf_counter() - t0, self._base)
        input_tokens.add(u.total_input_tokens, self._base)
        output_tokens.add(u.output_tokens, self._base)

    # [end:llm]

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float | None = None,
    ) -> AsyncIterator[str | LLMResponse]:
        if not isinstance(self._inner, LLMStreamer):
            raise TypeError("wrapped LLM cannot stream")
        with tracer.start_as_current_span(
            f"chat {self._model}",
            kind=SpanKind.CLIENT,
            attributes={**self._base, "gen_ai.request.stream": True},
        ) as span:
            t0 = time.perf_counter()
            first = True
            try:
                async for item in self._inner.stream(
                    messages,
                    system=system,
                    max_tokens=max_tokens,
                    temperature=temperature,
                ):
                    if isinstance(item, str) and first:
                        span.set_attribute(
                            "gen_ai.response.time_to_first_chunk",
                            time.perf_counter() - t0,
                        )
                        first = False
                    if isinstance(item, LLMResponse):
                        self._record(span, item, t0, messages)
                    yield item
            except LLMError as exc:
                self._failed(span, exc, t0)
                raise


# [start:embedder]
class TracedEmbedder:
    def __init__(self, inner: Embedder) -> None:
        self._inner = inner

    @property
    def model_id(self) -> str:
        return self._inner.model_id

    @property
    def dimensions(self) -> int:
        return self._inner.dimensions

    async def embed_documents(self, texts: Sequence[str]) -> Matrix:
        with tracer.start_as_current_span(
            f"embeddings {self.model_id}",
            attributes={
                "gen_ai.operation.name": "embeddings",
                "gen_ai.request.model": self.model_id,
                "halden.batch_size": len(texts),
            },
        ):
            return await self._inner.embed_documents(texts)

    async def embed_query(self, text: str) -> Vector:
        with tracer.start_as_current_span(
            f"embeddings {self.model_id}",
            attributes={
                "gen_ai.operation.name": "embeddings",
                "gen_ai.request.model": self.model_id,
            },
        ):
            return await self._inner.embed_query(text)


# [end:embedder]


class TracedReranker:
    def __init__(self, inner: Reranker) -> None:
        self._inner = inner

    @property
    def model_id(self) -> str:
        return self._inner.model_id

    async def score(
        self, query: str, texts: Sequence[str]
    ) -> list[float]:
        with tracer.start_as_current_span(
            f"rerank {self.model_id}",
            attributes={"halden.candidates": len(texts)},
        ):
            return await self._inner.score(query, texts)


# [start:service]
class TracedAskService(AskService):
    """Spans for each stage AskService runs. It overrides the
    stages; AskService itself contains no telemetry code."""

    @property
    def _salt(self) -> str:
        return self._settings.telemetry_salt.get_secret_value()

    @property
    def _capture(self) -> bool:
        return self._settings.telemetry_capture_content

    async def retrieve(
        self, question: str, identity: Identity
    ) -> list[RetrievedChunk]:
        with tracer.start_as_current_span(
            "retrieval chunks",
            kind=SpanKind.CLIENT,
            attributes={
                "gen_ai.operation.name": "retrieval",
                "gen_ai.data_source.id": "chunks",
                "halden.retrieval.mode": self._settings.retrieval_mode,
            },
        ) as span:
            hits = await super().retrieve(question, identity)
            # IDs and scores, never text. Chunk IDs here derive
            # from file names; if yours can reveal content (e.g.
            # "dismissal-jsmith.pdf"), hash them too.
            span.set_attribute(
                "gen_ai.retrieval.documents",
                json.dumps(
                    [
                        {"id": h.chunk_id, "score": round(h.score, 4)}
                        for h in hits
                    ]
                ),
            )
            if self._capture:
                span.set_attribute(
                    "gen_ai.retrieval.query.text", redact(question)
                )
            return hits

    def context(self, hits: list[RetrievedChunk]) -> Context:
        with tracer.start_as_current_span("build_context") as span:
            ctx = super().context(hits)
            span.set_attributes(
                {
                    "halden.context.sources": len(ctx.sources),
                    "halden.context.tokens": ctx.tokens,
                    "halden.context.dropped": len(ctx.dropped),
                }
            )
            return ctx

    async def ask(
        self,
        question: str,
        identity: Identity,
        conversation_id: uuid.UUID | None = None,
    ) -> Answer:
        with tracer.start_as_current_span(
            "ask",
            attributes={
                "halden.user": pseudonym(identity.user_id, self._salt),
                "gen_ai.prompt.name": "grounded-answer",
                "gen_ai.prompt.version": PROMPT_VERSION,
            },
        ) as span:
            t0 = time.perf_counter()
            try:
                ans = await super().ask(
                    question, identity, conversation_id
                )
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR))
                ask_duration.record(
                    time.perf_counter() - t0, {"outcome": "error"}
                )
                raise
            outcome = "abstained" if ans.abstained else "answered"
            span.set_attributes(
                {
                    "halden.outcome": outcome,
                    "halden.citations": len(ans.citations),
                    "halden.citation_check_ok": ans.citation_check_ok,
                }
            )
            if conversation_id is not None:
                span.set_attribute(
                    "gen_ai.conversation.id", str(conversation_id)
                )
            ask_duration.record(
                time.perf_counter() - t0, {"outcome": outcome}
            )
            return ans

    async def ask_stream(
        self, question: str, identity: Identity
    ) -> AsyncIterator[dict[str, Any]]:
        # One task iterates the whole stream (Starlette does), so
        # the span can stay current across the yields.
        with tracer.start_as_current_span(
            "ask_stream",
            attributes={
                "halden.user": pseudonym(identity.user_id, self._salt),
                "gen_ai.prompt.version": PROMPT_VERSION,
            },
        ) as span:
            async for event in super().ask_stream(question, identity):
                if event["type"] == "done":
                    span.set_attribute(
                        "halden.citation_check_ok",
                        bool(event["citation_check_ok"]),
                    )
                yield event


# [end:service]
