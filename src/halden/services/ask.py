"""The question-answering use case, independent of HTTP."""

import time
import uuid
from collections.abc import AsyncIterator
from typing import Any, Protocol

from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel, Field

from halden.config.settings import Settings
from halden.generation.context import Context, build_context
from halden.generation.grounded import (
    ABSTAIN,
    SYSTEM_PROMPT,
    build_messages,
    check_citations,
)
from halden.ports.embedder import Embedder
from halden.ports.llm import (
    LLMClient,
    LLMResponse,
    LLMStreamer,
    Message,
    TextBlock,
)
from halden.ports.reranker import Reranker
from halden.retrieval.pg_retriever import PgRetriever, RetrievedChunk
from halden.retrieval.query import condense_question
from halden.retrieval.rerank import rerank
from halden.security.identity import Identity
from halden.store.conversations import ConversationStore
from halden.store.scope import reader_scope


# [start:models]
class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    conversation_id: uuid.UUID | None = None


class Citation(BaseModel):
    label: int
    chunk_id: str
    title: str
    heading_path: list[str]


class Answer(BaseModel):
    answer: str
    abstained: bool
    citations: list[Citation]
    citation_check_ok: bool
    input_tokens: int = 0
    output_tokens: int = 0
    timings_ms: dict[str, float] = Field(default_factory=dict)
    search_query: str = ""  # what retrieval actually searched for
    conversation_id: uuid.UUID | None = None
    cached: bool = False  # served from the answer cache (Ch 25)
    sanitized: bool = False  # links or images removed (Ch 26)


class ConversationNotFound(Exception):
    pass


class AskUseCase(Protocol):
    """What the API needs. AskService implements it; so do the
    caching and budget wrappers of Chapter 25."""

    async def ask(
        self,
        question: str,
        identity: Identity,
        conversation_id: uuid.UUID | None = None,
    ) -> Answer: ...

    def ask_stream(
        self, question: str, identity: Identity
    ) -> AsyncIterator[dict[str, Any]]: ...


# [end:models]


class _Timer:
    def __init__(self) -> None:
        self.ms: dict[str, float] = {}
        self._t = time.perf_counter()

    def lap(self, name: str) -> None:
        now = time.perf_counter()
        self.ms[name] = round((now - self._t) * 1000, 1)
        self._t = now


def _citations(ctx: Context, labels: list[int]) -> list[Citation]:
    by_label = {s.label: s for s in ctx.sources}
    return [
        Citation(
            label=n,
            chunk_id=by_label[n].chunk_id,
            title=by_label[n].title,
            heading_path=by_label[n].heading_path,
        )
        for n in labels
        if n in by_label
    ]


# [start:service]
class AskService:
    def __init__(
        self,
        pool: AsyncConnectionPool,
        embedder: Embedder,
        llm: LLMClient,
        settings: Settings,
        reranker: Reranker | None = None,
        conversations: ConversationStore | None = None,
        rewriter: LLMClient | None = None,
    ) -> None:
        self._conversations = conversations
        self._rewriter = rewriter or llm
        self._pool = pool
        self._embedder = embedder
        self._llm = llm
        self._settings = settings
        self._reranker = reranker

    async def retrieve(
        self, question: str, identity: Identity
    ) -> list[RetrievedChunk]:
        async with (
            self._pool.connection() as conn,
            reader_scope(conn, identity.groups, identity.tenant),
        ):
            retriever = PgRetriever(conn, self._embedder)
            if self._settings.retrieval_mode == "dense":
                hits = await retriever.dense(
                    question, identity.groups, k=12
                )
            else:
                hits = await retriever.hybrid(
                    question,
                    identity.groups,
                    k=12,
                    candidates=self._settings.retrieval_candidates,
                )
        if self._reranker is not None:
            hits = await rerank(self._reranker, question, hits, 8)
        return hits

    def context(self, hits: list[RetrievedChunk]) -> Context:
        return build_context(
            hits,
            budget_tokens=self._settings.context_budget_tokens,
            untrusted_systems=self._settings.untrusted_sources,
        )

    async def _standalone(
        self,
        question: str,
        identity: Identity,
        conversation_id: uuid.UUID | None,
    ) -> str:
        """The query to search with: the question itself, or, in a
        conversation, a condensed standalone version (Chapter 13)."""
        if conversation_id is None or self._conversations is None:
            return question
        history = await self._conversations.history(
            conversation_id, identity.user_id
        )
        if history is None:
            raise ConversationNotFound(str(conversation_id))
        return await condense_question(
            self._rewriter, history, question
        )

    async def ask(
        self,
        question: str,
        identity: Identity,
        conversation_id: uuid.UUID | None = None,
    ) -> Answer:
        answer = await self._ask(question, identity, conversation_id)
        if conversation_id is not None and self._conversations:
            await self._conversations.append(
                conversation_id, question, answer.answer
            )
        return answer

    async def _ask(
        self,
        question: str,
        identity: Identity,
        conversation_id: uuid.UUID | None,
    ) -> Answer:
        t = _Timer()
        query = await self._standalone(
            question, identity, conversation_id
        )
        t.lap("condense")
        ctx = self.context(await self.retrieve(query, identity))
        t.lap("retrieve")
        meta: dict[str, Any] = {
            "search_query": query,
            "conversation_id": conversation_id,
        }
        if not ctx.sources:  # nothing permitted matched: no LLM call
            return Answer(
                answer=ABSTAIN,
                abstained=True,
                citations=[],
                citation_check_ok=True,
                timings_ms=t.ms,
                **meta,
            )
        messages = build_messages(query, ctx)
        resp = await self._generate(messages)
        check = check_citations(resp.text, ctx, lenient=True)
        tokens_in = resp.usage.total_input_tokens
        tokens_out = resp.usage.output_tokens
        if check.unknown:  # invented sources: retry once with feedback
            messages += [
                Message(
                    role="assistant",
                    content=[TextBlock(text=resp.text)],
                ),
                Message.user(
                    f"You cited sources {check.unknown}, which do "
                    "not exist. Answer again citing only the "
                    "numbered sources provided."
                ),
            ]
            resp = await self._generate(messages)
            check = check_citations(resp.text, ctx, lenient=True)
            tokens_in += resp.usage.total_input_tokens
            tokens_out += resp.usage.output_tokens
        t.lap("generate")
        if check.unknown:  # still invented: do not show it
            return Answer(
                answer=ABSTAIN,
                abstained=True,
                citations=[],
                citation_check_ok=False,
                input_tokens=tokens_in,
                output_tokens=tokens_out,
                timings_ms=t.ms,
                **meta,
            )
        return Answer(
            answer=resp.text,
            abstained=check.abstained,
            citations=_citations(ctx, check.cited),
            citation_check_ok=check.ok,
            input_tokens=tokens_in,
            output_tokens=tokens_out,
            timings_ms=t.ms,
            **meta,
        )

    async def _generate(self, messages: list[Message]) -> LLMResponse:
        return await self._llm.complete(
            messages,
            system=SYSTEM_PROMPT,
            max_tokens=self._settings.answer_max_tokens,
        )

    # [end:service]

    # [start:stream]
    async def ask_stream(
        self, question: str, identity: Identity
    ) -> AsyncIterator[dict[str, Any]]:
        """Events: sources, token*, done. Citations are validated
        only after the full answer exists, so the client renders
        citation links from the 'done' event, never from tokens."""
        if not isinstance(self._llm, LLMStreamer):
            raise TypeError("configured LLM cannot stream")
        ctx = self.context(await self.retrieve(question, identity))
        yield {"type": "sources", "count": len(ctx.sources)}
        if not ctx.sources:
            yield {"type": "token", "text": ABSTAIN}
            yield {
                "type": "done",
                "abstained": True,
                "citations": [],
                "citation_check_ok": True,
            }
            return
        final: LLMResponse | None = None
        async for item in self._llm.stream(
            build_messages(question, ctx),
            system=SYSTEM_PROMPT,
            max_tokens=self._settings.answer_max_tokens,
        ):
            if isinstance(item, str):
                yield {"type": "token", "text": item}
            else:
                final = item
        assert final is not None
        check = check_citations(final.text, ctx, lenient=True)
        yield {
            "type": "done",
            "abstained": check.abstained,
            "citations": [
                c.model_dump()
                for c in _citations(
                    ctx,
                    [n for n in check.cited if n not in check.unknown],
                )
            ],
            "citation_check_ok": check.ok,
            "input_tokens": final.usage.total_input_tokens,
            "output_tokens": final.usage.output_tokens,
            "warning": (
                "answer cited sources that do not exist"
                if check.unknown
                else None
            ),
        }

    # [end:stream]
