# Code verification report

Generated: 2026-09-28 · re-run in book Session 2 with Python 3.13.13 · PostgreSQL 16.15 · pgvector 0.8.6

| Level | Status | Detail |
|---|---|---|
| V1 Static | PASS | ruff format, ruff check (line length 72), mypy --strict |
| V2 Unit | PASS | pytest: 153 passed in 45.17s (unit, property-based, integration, security, tenancy, telemetry, erasure) |
| V3 Integration | PASS (Parts I–V) | Real PostgreSQL + pgvector: schema migration, dense/lexical/hybrid retrieval, ACL filtering, model_id guard (tests run in throwaway schemas). Anthropic adapter verified against recorded API exchanges. All example programs executed by scripts/build_book.py and their output captured into the manuscript. |
| V4 Real models | PASS (Parts I–V) | BAAI/bge-small-en-v1.5 (embeddings), BAAI/bge-base-en-v1.5 (Chapter 27 blue/green index), cross-encoder/ms-marco-MiniLM-L6-v2 (reranking), via sentence-transformers 6.1 |
| V5 Live provider | PASS (Parts I–V) | Anthropic Messages API: claude-haiku-4-5 (query rewriting, routing, map step, synthetic questions, judge calibration), claude-sonnet-5 (answers, agent), claude-opus-5-5 (evaluation judge). Parts III–V examples recorded 2026-09-27/28; API run end to end under uvicorn and curl; prompt caching verified from usage counts |
| Container | PARTIAL | Dockerfile steps executed in a clean directory (locked install without dev dependencies, model download, API started from the result with models offline: /healthz, /readyz, a live /v1/ask, clean SIGTERM shutdown). The image itself could not be built in the build environment (container registries unreachable); CI builds and smoke-tests it on every push. Workflow checked with actionlint; Compose file with `docker compose config`. |

## Defects found by verification (all fixed)
1. Newer Claude models reject `temperature`; adapter always sent it (V5).
2. sentence-transformers 6.x renamed `get_sentence_embedding_dimension` (V4).
3. Recursive chunker dropped sentence-final periods at chunk boundaries (property test).
4. Header/footer stripper removes body lines that differ only by digits (documented limitation, test pins it).
5. Experiment harness: psycopg auto-prepared statements kept a cached plan, so `enable_indexscan` changes had no effect (documented in Chapter 9).

6. Map-reduce combining step silently truncated at its output limit (stop_reason not checked); now detected and reported.
7. Book build let deterministic examples see the API key, switching one to live mode; the builder now strips the key from non-live runs.
8. One-page PDF kept its running header as title (repeated-line detection needs 2+ pages); fixed with position-based margin filtering.
9. Parser fixes never reached indexed documents (file hash unchanged). The pipeline-version fix was applied by hand during Part IV but never committed; see 15.
10. Margin filter re-sorted text fragments, scrambling a table row; fixed by keeping extractor order.
11. Hard per-document context cap discarded the only chunk with the answer; replaced with a soft cap plus backfill.
12. Strict citation check rejected valid answers; lenient mode added with regression tests from recorded answers.
13. An automatic lint fix deleted a re-exported name used by examples; an import test now covers every example.
14. The folder connector deleted Part II demo documents sharing the same tables; demo data moved to its own schema.
15. The pipeline version (9) was missing from the connector, so the index still held scrambled PDF table rows. Found by Chapter 23's evidence-label check; fixed, with a regression test.
16. `missed_abstain` reported 1.0 in retrieval-only runs (no answer treated as "answered"); fixed before the baseline was written.
17. Judge calibration found a wrong reference label and two judge-setup errors (no document titles, no question); all fixed (Chapter 23).
18. Automatic (top-level) prompt caching wrote a new cache entry on every independent question, costing 21% more; replaced with an explicit breakpoint after the system prompt (Chapter 25).
19. The first prompt-injection detector flagged correct answers that quoted the injected text; replaced (Chapter 26).
20. An embedding-model/index mismatch silently degraded retrieval to keyword-only (the model_id filter returns no dense results); the API now refuses to start (Chapter 27).
21. Not fixed, documented: 13 of 56 synthetic questions don't name the product they ask about (Chapter 23).
22. Docker Compose and CI created the application's database user as a superuser, which bypasses row-level security even with FORCE. The RLS tests passed only because the build environment's role was ordinary. Compose and CI now create an ordinary `halden` role (`docker/initdb/01-app-role.sql`), and a test fails if the application's role can bypass RLS (Chapter 19).

Build: `uv run python scripts/build_book.py --final` fails on any unverified output. Verify: `uv run python scripts/verify.py`.
