# Halden Knowledge Platform

Companion code for the book **RAG Beyond the Demo: Building, Evaluating, and Operating Retrieval-Augmented Generation in Python** by Arlo S.M. Venn. A companion book on Model Context Protocol (MCP) servers will build on this repository.

The repository grows chapter by chapter into the *Halden Knowledge Platform*, the system behind **Ask Halden**, an assistant for the employees of Halden Instruments, a **fictional** industrial-sensor manufacturer. All company data in `data/` is invented.

> **Status:** complete for the book: Parts I–V (Chapters 1–27) and the closing Chapter 28. See [`docs/CHAPTERS.md`](docs/CHAPTERS.md) for where each chapter's code lives.

## What's inside

| Path | Contents |
|---|---|
| `src/halden/ports/` | Interfaces the platform depends on: `LLMClient`, `Embedder`, `Reranker` |
| `src/halden/adapters/` | Implementations: Anthropic Messages API, local embedding and reranking models, test doubles |
| `src/halden/ingestion/` | Parsing, normalization, change detection, chunking |
| `src/halden/retrieval/` | Dense, lexical, and hybrid retrieval on PostgreSQL + pgvector; fusion; reranking |
| `src/halden/generation/` | Structured output, context building, grounded prompts, citation checks, map-reduce |
| `src/halden/agent/` | Bounded tool-using agent loop, search tool |
| `src/halden/store/`, `services/`, `api/`, `security/` | Database access, job queue, answer cache, budgets, the question-answering service, FastAPI app, API keys, output sanitizing, erasure |
| `src/halden/eval/` | Evaluation: datasets, retrieval metrics, LLM judge, calibration, reports, CI gate |
| `src/halden/observability/` | OpenTelemetry tracing and metrics, cost per model call |
| `eval/` | Golden and synthetic datasets, baseline, prompt-injection fixtures |
| `migrations/` | Database schema, including row-level security |
| `examples/` | One runnable program per chapter experiment |
| `scripts/` | `verify.py` (all checks), `eval.py` (evaluation and gate), `worker.py` (ingestion worker), `build_index.py` (blue/green re-index), `load_demo_index.py` (sample data) |
| `Dockerfile`, `docker-compose.yml` | Container image for API and worker; local database and full stack |
| `tests/` | Unit tests, and integration tests against a real database and real models |

## Requirements

- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- Docker (for the local PostgreSQL + pgvector database)
- About 2 GB of disk for the open-weight models, which download automatically on first use

No API key is needed for the tests or the deterministic examples. Examples marked "live model" in the chapter guide print setup instructions when no key is set.

## Quick start

```bash
uv sync --all-extras               # install dependencies
docker compose up -d               # start PostgreSQL + pgvector
uv run python scripts/verify.py    # lint, type-check, run all tests

uv run python scripts/load_demo_index.py       # Part II-III demo data
uv run python -m examples.ch10_compare_retrievers

uv run python scripts/migrate.py               # Part IV application
for s in manuals policies contracts tickets; do
  uv run python scripts/ingest.py data/sources/$s $s
done
uv run python scripts/create_user.py alice "Alice" everyone
ANTHROPIC_API_KEY=... HALDEN_MODEL=... \
  uv run uvicorn --factory halden.api.app:create_app
uv run python scripts/eval.py --baseline eval/baseline.json

# or the whole stack in containers
docker compose --profile app up --build
```

Examples read the database location from `HALDEN_DSN` and default to the Docker Compose database. See `.env.example` for all settings.

The database container creates the extension and an ordinary `halden` role on first start (`docker/initdb/`). The application never connects as a superuser, because superusers bypass row-level security. If you started the database from an earlier version of this repository, recreate it once with `docker compose down -v && docker compose up -d`.

## Running the chapter examples

```bash
uv run python examples/ch02_tool_round_trip.py        # offline
uv run python examples/ch04_semantic_search.py        # real model
uv run python examples/ch08_chunking_experiment.py
uv run python -m examples.ch09_ann_recall             # needs the database
uv run python -m examples.ch11_rerank
```

To run the Chapter 2 example against the live Anthropic API, set `ANTHROPIC_API_KEY` and `HALDEN_MODEL` in your environment. Never commit keys.

## How this code is verified

Every code listing and every printed result in the book is extracted from this repository by a build script, and the build fails if any output was not produced by an actual run. `scripts/verify.py` runs formatting, linting (72-character lines, for e-readers), strict type checking, and the test suite. CI runs the same checks, the retrieval evaluation gate, and the examples against a real pgvector database on every push, builds the container image, and runs the full evaluation with a live model nightly. The verification report is in [`VERIFICATION.md`](VERIFICATION.md).

## License

The code in this repository is released under the [MIT License](LICENSE).
Copyright (c) 2026 Arlo S.M. Venn.

The book's text is not part of this repository and is not covered by
this license.
