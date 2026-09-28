-- 002_ingestion.sql: job queue and source versions.

-- [start:jobs]
CREATE TABLE jobs (
    id           bigserial PRIMARY KEY,
    kind         text NOT NULL,
    payload      jsonb NOT NULL,
    dedupe_key   text NOT NULL,
    status       text NOT NULL DEFAULT 'queued'
                 CHECK (status IN
                     ('queued', 'running', 'done', 'dead')),
    attempts     int NOT NULL DEFAULT 0,
    max_attempts int NOT NULL DEFAULT 5,
    run_after    timestamptz NOT NULL DEFAULT now(),
    locked_at    timestamptz,
    last_error   text,
    created_at   timestamptz NOT NULL DEFAULT now(),
    updated_at   timestamptz NOT NULL DEFAULT now()
);

-- At most one pending job per (kind, dedupe_key): enqueueing the
-- same work twice while it is queued or running is a no-op.
CREATE UNIQUE INDEX jobs_pending_dedupe ON jobs (kind, dedupe_key)
    WHERE status IN ('queued', 'running');
CREATE INDEX jobs_ready ON jobs (run_after, id)
    WHERE status = 'queued';
-- [end:jobs]

-- The connector's change marker (e.g. a file hash), distinct from
-- content_hash, which covers parsed content and permissions.
ALTER TABLE documents ADD COLUMN source_version text;
