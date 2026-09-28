"""HTTP front door: authentication, validation, errors, streaming."""

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from opentelemetry.propagate import extract
from opentelemetry.trace import SpanKind

from halden.config.settings import Settings
from halden.observability.telemetry import tracer
from halden.ports.llm import LLMError
from halden.security.api_keys import authenticate
from halden.security.identity import Identity
from halden.services.ask import Answer, AskRequest, ConversationNotFound
from halden.services.container import Services, build_services
from halden.store.usage import BudgetExceeded

log = logging.getLogger(__name__)
ServiceFactory = Callable[[Settings], Awaitable[Services]]


class Unauthorized(Exception):
    pass


# [start:errors]
def problem(
    status: int, title: str, detail: str, request: Request
) -> JSONResponse:
    """RFC 9457 problem details: a stable, documented error shape
    that never includes stack traces or internal messages."""
    return JSONResponse(
        status_code=status,
        media_type="application/problem+json",
        content={
            "type": "about:blank",
            "title": title,
            "status": status,
            "detail": detail,
            "request_id": getattr(request.state, "request_id", None),
        },
    )


# [end:errors]


def create_app(
    settings: Settings | None = None,
    factory: ServiceFactory = build_services,
) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        services = await factory(settings)
        app.state.services = services
        yield
        await services.pool.close()

    app = FastAPI(title="Ask Halden", lifespan=lifespan)

    # [start:middleware]
    @app.middleware("http")
    async def request_id(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = rid
        # A SERVER span per request, continuing the caller's trace
        # if it sent a W3C traceparent header. A no-op until
        # telemetry is configured.
        with tracer.start_as_current_span(
            f"{request.method} {request.url.path}",
            context=extract(request.headers),
            kind=SpanKind.SERVER,
            attributes={
                "http.request.method": request.method,
                "url.path": request.url.path,
                "halden.request_id": rid,
            },
        ) as span:
            response = await call_next(request)
            span.set_attribute(
                "http.response.status_code", response.status_code
            )
        response.headers["x-request-id"] = rid
        return response

    # [end:middleware]

    @app.exception_handler(RequestValidationError)
    async def invalid(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return problem(
            422,
            "Invalid request",
            "; ".join(e["msg"] for e in exc.errors()),
            request,
        )

    @app.exception_handler(LLMError)
    async def llm_down(request: Request, exc: LLMError) -> JSONResponse:
        log.warning("LLM error: %s", exc)
        resp = problem(
            503,
            "Model temporarily unavailable",
            "Please retry shortly.",
            request,
        )
        resp.headers["retry-after"] = "5"
        return resp

    @app.exception_handler(Unauthorized)
    async def unauthorized(
        request: Request, exc: Unauthorized
    ) -> JSONResponse:
        resp = problem(
            401, "Unauthorized", "A valid API key is required.", request
        )
        resp.headers["www-authenticate"] = "Bearer"
        return resp

    @app.exception_handler(ConversationNotFound)
    async def no_conversation(
        request: Request, exc: ConversationNotFound
    ) -> JSONResponse:
        return problem(
            404, "Not found", "No such conversation.", request
        )

    @app.exception_handler(BudgetExceeded)
    async def over_budget(
        request: Request, exc: BudgetExceeded
    ) -> JSONResponse:
        now = datetime.now(UTC)
        midnight = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        resp = problem(
            429,
            "Daily budget used",
            "Your daily question budget is used up.",
            request,
        )
        wait = int((midnight - now).total_seconds()) + 1
        resp.headers["retry-after"] = str(wait)
        return resp

    @app.exception_handler(TimeoutError)
    async def too_slow(
        request: Request, exc: TimeoutError
    ) -> JSONResponse:
        return problem(
            504,
            "Timed out",
            "The answer took too long to produce.",
            request,
        )

    def services(request: Request) -> Services:
        s: Services = request.app.state.services
        return s

    # [start:auth]
    async def identity(request: Request) -> Identity:
        if settings.auth_mode == "dev":
            return Identity(
                user_id=settings.dev_user, groups=settings.dev_groups
            )
        scheme, _, token = request.headers.get(
            "authorization", ""
        ).partition(" ")
        who = None
        if scheme.lower() == "bearer" and token:
            who = await authenticate(services(request).pool, token)
        if who is None:
            raise Unauthorized()
        return who

    # [end:auth]

    Svc = Annotated[Services, Depends(services)]
    Who = Annotated[Identity, Depends(identity)]

    # [start:routes]
    @app.post("/v1/ask", response_model=Answer)
    async def ask(body: AskRequest, svc: Svc, who: Who) -> Answer:
        async with asyncio.timeout(settings.request_timeout_s):
            return await svc.ask.ask(
                body.question, who, body.conversation_id
            )

    @app.post("/v1/ask/stream")
    async def ask_stream(
        body: AskRequest, svc: Svc, who: Who
    ) -> StreamingResponse:
        async def events() -> AsyncIterator[str]:
            try:
                async for event in svc.ask.ask_stream(
                    body.question, who
                ):
                    yield (
                        f"event: {event['type']}\n"
                        f"data: {json.dumps(event)}\n\n"
                    )
            except LLMError:
                # Headers are already sent: report in-band.
                err: dict[str, Any] = {
                    "type": "error",
                    "detail": "model unavailable",
                }
                yield f"event: error\ndata: {json.dumps(err)}\n\n"

        return StreamingResponse(
            events(), media_type="text/event-stream"
        )

    @app.post("/v1/conversations", status_code=201)
    async def new_conversation(svc: Svc, who: Who) -> dict[str, str]:
        cid = await svc.conversations.create(who.user_id)
        return {"conversation_id": str(cid)}

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}  # the process is alive

    @app.get("/readyz")
    async def readyz(svc: Svc) -> dict[str, str]:
        async with svc.pool.connection() as conn:  # can we serve?
            await conn.execute("SELECT 1")
        return {"status": "ready"}

    # [end:routes]
    return app
