-- 004: index generation, answer cache, per-user daily usage.

-- [start:generation]
-- A counter that moves whenever any chunk changes. Cache keys
-- include it, so every cached answer is invalidated by re-indexing
-- without the ingestion code having to remember to do it.
CREATE TABLE index_state (
    id         boolean PRIMARY KEY DEFAULT true CHECK (id),
    generation bigint NOT NULL DEFAULT 0
);
INSERT INTO index_state DEFAULT VALUES;

CREATE FUNCTION bump_index_generation() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    UPDATE index_state SET generation = generation + 1;
    RETURN NULL;
END $$;

CREATE TRIGGER chunks_changed
AFTER INSERT OR UPDATE OR DELETE OR TRUNCATE ON chunks
FOR EACH STATEMENT EXECUTE FUNCTION bump_index_generation();
-- [end:generation]

-- [start:cache]
CREATE TABLE answer_cache (
    key        text PRIMARY KEY,  -- sha256, see store/cache.py
    groups     text[] NOT NULL,   -- checked again on every read
    answer     jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
-- [end:cache]

-- [start:usage]
CREATE TABLE usage_daily (
    user_id       text NOT NULL,
    day           date NOT NULL,
    input_tokens  bigint NOT NULL DEFAULT 0,
    output_tokens bigint NOT NULL DEFAULT 0,
    cost_usd      numeric(12, 6) NOT NULL DEFAULT 0,
    PRIMARY KEY (user_id, day)
);
-- [end:usage]
