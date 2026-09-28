# Chapter guide

Where each chapter's code lives and how to run it.

| Ch | Topic | Code | Run |
|---|---|---|---|
| 2 | LLM fundamentals, tool calling | `ports/llm.py`, `adapters/anthropic_llm.py`, `adapters/fake_llm.py` | `uv run python examples/ch02_tool_round_trip.py` |
| 3 | Structured output | `generation/structured.py`, `models/triage.py` | `uv run pytest tests/unit/test_part1.py` |
| 4 | Embeddings, brute-force search | `ports/embedder.py`, `adapters/local_embedder.py`, `retrieval/brute_force.py` | `uv run python examples/ch04_semantic_search.py` |
| 7 | Ingestion, parsing, normalization | `models/document.py`, `ingestion/markdown_parser.py`, `ingestion/normalize.py`, `ingestion/sync.py` | `uv run pytest tests/unit/test_ingestion.py` |
| 8 | Chunking | `ingestion/chunking.py` | `uv run python examples/ch08_chunking_experiment.py` |
| 9 | Vector indexing | `migrations/001_chunks.sql` | `uv run python -m examples.ch09_ann_recall`, `uv run python -m examples.ch09_filtered_ann` |
| 10 | Dense, lexical, hybrid retrieval | `retrieval/pg_retriever.py`, `retrieval/fusion.py` | `uv run python scripts/load_demo_index.py` then `uv run python -m examples.ch10_compare_retrievers` |
| 11 | Reranking | `ports/reranker.py`, `adapters/cross_encoder.py`, `retrieval/rerank.py` | `uv run python -m examples.ch11_rerank` |
| 12 | Context building, citation checks | `generation/context.py`, `generation/grounded.py` | `uv run python -m examples.ch12_context_and_citations` |
| 13 | Query condensation, multi-query, HyDE, routing | `retrieval/query.py`, `retrieval/routing.py`, `config/llm_factory.py` | `python -m examples.ch13_condense`, `examples.ch13_multi_query`, `examples.ch13_route` (live model) |
| 14 | Parent-child and graph retrieval | `retrieval/structure.py` | `uv run python -m examples.ch14_structure` |
| 15 | Agent loop, map-reduce | `agent/loop.py`, `agent/search_tool.py`, `generation/mapreduce.py` | `python -m examples.ch15_agent_vs_pipeline`, `examples.ch15_long_context` (live model) |
| 16 | Project layout, settings, pool, migrations | `config/settings.py`, `store/db.py`, `services/container.py` | `uv run python scripts/migrate.py` |
| 17 | Ingestion pipeline | `ingestion/pdf_parser.py`, `ingestion/service.py`, `ingestion/worker.py`, `ingestion/folder_connector.py`, `store/jobs.py`, `store/documents.py` | `uv run python -m examples.ch17_ingest_demo`; `uv run python scripts/ingest.py data/sources/manuals manuals` |
| 18 | Q&A service and API | `services/ask.py`, `adapters/resilient.py`, `api/app.py` | `HALDEN_AUTH_MODE=dev uv run uvicorn --factory halden.api.app:create_app` (live model) |
| 19 | API keys, row-level security, memory | `security/api_keys.py`, `store/scope.py`, `store/conversations.py`, `migrations/003_*`, `docker/initdb/01-app-role.sql` | `uv run python scripts/create_user.py`; `python -m examples.ch19_permissions_and_memory` (live model) |
| 20 | Configurable hybrid + rerank benchmark | `services/ask.py` (`retrieval_mode`) | `uv run python -m examples.ch20_benchmark` |
| 21 | Test suite | `tests/` | `uv run python scripts/verify.py` |
| 23 | Evaluation | `eval/` (package), `eval/datasets/`, `scripts/eval.py`, `scripts/make_synthetic.py` | `uv run python scripts/eval.py`; `uv run python -m examples.ch23_retrieval_eval`; `--mode full` and `examples.ch23_calibrate_judge` (live model) |
| 24 | Observability | `observability/telemetry.py`, `observability/instrumented.py`, `observability/cost.py` | `HALDEN_TELEMETRY=console ...`; `python -m examples.ch24_trace_waterfall` (live model) |
| 25 | Caching, latency, cost | `store/cache.py`, `store/usage.py`, `services/guarded.py`, `migrations/004_*` | `uv run python -m examples.ch25_semantic_cache`, `examples.ch25_retrieval_latency`, `examples.ch25_cost_model`; `examples.ch25_prompt_caching` (live model) |
| 26 | Security, privacy, tenancy | `security/output.py`, `security/forget.py`, `migrations/005_*`, `eval/injection/` | `uv run pytest tests/integration/test_tenancy.py tests/integration/test_forget.py`; `uv run python -m examples.ch26_trust_labels`; `python -m examples.ch26_injection_test` (live model) |
| 27 | Deployment | `Dockerfile`, `docker-compose.yml`, `scripts/worker.py`, `scripts/build_index.py`, `.github/workflows/ci.yml` | `docker compose --profile app up --build`; `uv run python scripts/build_index.py green BAAI/bge-base-en-v1.5` |
| 28 | Reference architecture (summary; no new code) | `services/container.py` (composition root) | `uv run python scripts/verify.py` |

Before Chapters 18–27, ingest all four sources: `for s in manuals policies contracts tickets; do uv run python scripts/ingest.py data/sources/$s $s; done`

Examples marked "live model" need `ANTHROPIC_API_KEY`, `HALDEN_MODEL`, and `HALDEN_MODEL_SMALL`; without them they print setup instructions and exit.

Paths under "Code" are relative to `src/halden/` unless they start with another top-level folder. Chapters 1, 5, and 6 are conceptual and have no code.
