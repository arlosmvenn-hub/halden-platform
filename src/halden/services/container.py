"""The composition root: the one place that wires adapters to ports."""

from dataclasses import dataclass
from datetime import timedelta

from psycopg_pool import AsyncConnectionPool

from halden.adapters.anthropic_llm import AnthropicLLM
from halden.adapters.cross_encoder import CrossEncoderReranker
from halden.adapters.local_embedder import SentenceTransformerEmbedder
from halden.adapters.resilient import ResilientLLM
from halden.config.settings import Settings
from halden.observability.instrumented import (
    TracedAskService,
    TracedEmbedder,
    TracedLLM,
    TracedReranker,
)
from halden.observability.telemetry import setup_telemetry
from halden.ports.embedder import Embedder
from halden.ports.llm import LLMClient
from halden.ports.reranker import Reranker
from halden.services.ask import AskService, AskUseCase
from halden.services.guarded import (
    BudgetedAsk,
    CachedAsk,
    SanitizedAsk,
)
from halden.store.cache import AnswerCache
from halden.store.conversations import ConversationStore
from halden.store.db import create_pool, migrate
from halden.store.scope import writer_scope
from halden.store.usage import UsageLedger


@dataclass
class Services:
    settings: Settings
    pool: AsyncConnectionPool
    embedder: Embedder
    llm: LLMClient
    ask: AskUseCase
    conversations: ConversationStore
    reranker: Reranker | None = None


# [start:build]
async def build_services(
    settings: Settings, *, install_exporter: bool = True
) -> Services:
    """Create every long-lived component once, at startup.

    Models load here, not on the first request, so a readiness
    probe can wait for them and no user pays the load time.
    """
    if settings.anthropic_api_key is None or settings.model is None:
        raise RuntimeError(
            "ANTHROPIC_API_KEY and HALDEN_MODEL required"
        )
    traced = settings.telemetry != "off"
    if traced and install_exporter:
        _install_exporter(settings)
    pool = create_pool(settings)
    await pool.open(wait=True, timeout=30)
    if settings.auto_migrate:
        async with pool.connection() as conn:
            await conn.set_autocommit(True)
            await migrate(conn, settings.migrations_dir)
    await check_index_model(pool, settings)  # Chapter 27
    embedder: Embedder = SentenceTransformerEmbedder(
        settings.embedding_model,
        query_prefix=settings.embedding_query_prefix,
    )
    reranker: Reranker | None = (
        CrossEncoderReranker(settings.reranker_model)
        if settings.reranker_model
        else None
    )
    if traced:  # Chapter 24: trace at the seams
        embedder = TracedEmbedder(embedder)
        reranker = TracedReranker(reranker) if reranker else None
    llm = _model(settings, settings.model)
    rewriter = (
        _model(settings, settings.model_small)
        if settings.model_small
        else None
    )
    conversations = ConversationStore(pool)
    service = TracedAskService if traced else AskService
    ask: AskUseCase = service(
        pool, embedder, llm, settings, reranker, conversations, rewriter
    )
    hosts = frozenset(settings.allowed_link_hosts)
    ask = SanitizedAsk(ask, hosts)  # Chapter 26
    if settings.daily_token_budget:  # Chapter 25
        ledger = UsageLedger(pool, settings.daily_token_budget)
        ask = BudgetedAsk(ask, ledger, settings.model)
    if settings.answer_cache_ttl_s:  # outermost: hits cost nothing
        ttl = timedelta(seconds=settings.answer_cache_ttl_s)
        ask = CachedAsk(ask, AnswerCache(pool, ttl), settings)
    return Services(
        settings, pool, embedder, llm, ask, conversations, reranker
    )


def _model(settings: Settings, name: str) -> LLMClient:
    """Provider adapter, traced per attempt when telemetry is on,
    inside the retry wrapper, so each retry is its own span."""
    assert settings.anthropic_api_key is not None
    client: LLMClient = AnthropicLLM(
        settings.anthropic_api_key.get_secret_value(),
        name,
        timeout_s=settings.llm_timeout_s,
    )
    if settings.telemetry != "off":
        client = TracedLLM(
            client,
            name,
            capture_content=settings.telemetry_capture_content,
        )
    return ResilientLLM(
        client,
        attempts=settings.llm_max_attempts,
        timeout_s=settings.llm_timeout_s,
    )


# [end:build]


# [start:check]
async def check_index_model(
    pool: AsyncConnectionPool, settings: Settings
) -> None:
    """Refuse to start if the index was built with a different
    embedding model: every query would silently return garbage.
    An instance that fails here never becomes ready, so a bad
    deploy stalls instead of serving wrong answers."""
    async with pool.connection() as conn, writer_scope(conn):
        cur = await conn.execute(
            "SELECT DISTINCT model_id FROM chunks LIMIT 5"
        )
        found = {row[0] for row in await cur.fetchall()}
    if found and found != {settings.embedding_model}:
        raise RuntimeError(
            f"index {settings.index_schema!r} holds vectors from"
            f" {', '.join(sorted(found))};\n  the configured model"
            f" is {settings.embedding_model}"
        )


# [end:check]


def _install_exporter(settings: Settings) -> None:
    from opentelemetry.sdk.trace.export import (
        ConsoleSpanExporter,
        SpanExporter,
    )

    exporter: SpanExporter
    if settings.telemetry == "otlp":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # noqa: E501
            OTLPSpanExporter,
        )

        exporter = OTLPSpanExporter(endpoint=settings.otlp_endpoint)
    else:
        exporter = ConsoleSpanExporter()
    setup_telemetry(
        "halden-api", exporter, sample_ratio=settings.trace_sample_ratio
    )
