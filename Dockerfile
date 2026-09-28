# syntax=docker/dockerfile:1
# One image, two processes: the API (default command) and the
# ingestion worker (command: python scripts/worker.py).
#   docker build -t halden .
ARG PYTHON_IMAGE=python:3.12-slim

# [start:build]
FROM ${PYTHON_IMAGE} AS build
RUN pip install --no-cache-dir "uv>=0.9"
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
# Dependencies first: this layer is reused until the lock changes.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project \
        --extra local-models --extra otel
COPY src ./src
RUN uv sync --frozen --no-dev --extra local-models --extra otel
# Bake the open-weight models in: no downloads at startup.
COPY scripts/fetch_models.py ./scripts/
ENV HF_HOME=/app/hf
RUN .venv/bin/python scripts/fetch_models.py
# [end:build]

# [start:runtime]
FROM ${PYTHON_IMAGE}
RUN useradd --create-home --uid 10001 halden
WORKDIR /app
COPY --from=build --chown=halden /app /app
COPY --chown=halden migrations ./migrations
COPY --chown=halden scripts ./scripts
ENV PATH=/app/.venv/bin:$PATH \
    HF_HOME=/app/hf \
    HF_HUB_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
    HALDEN_MIGRATIONS_DIR=/app/migrations
USER halden
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=60s \
    CMD python -c "import urllib.request as u; \
u.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"
# Uvicorn stops accepting on SIGTERM and gives in-flight requests
# up to 30 s; the orchestrator's grace period must be longer.
CMD ["uvicorn", "--factory", "halden.api.app:create_app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--timeout-graceful-shutdown", "30"]
# [end:runtime]
