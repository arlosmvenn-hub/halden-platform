-- 001_chunks.sql: documents, chunks, vectors, full text.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    doc_id        text PRIMARY KEY,
    title         text NOT NULL,
    source_system text NOT NULL,
    source_uri    text NOT NULL,
    content_hash  text NOT NULL,
    acl_groups    text[] NOT NULL,
    language      text NOT NULL DEFAULT 'en',
    indexed_at    timestamptz NOT NULL DEFAULT now()
);

-- [start:chunks]
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     text PRIMARY KEY,
    doc_id       text NOT NULL
                 REFERENCES documents ON DELETE CASCADE,
    ordinal      int  NOT NULL,
    heading_path text[] NOT NULL,
    kind         text NOT NULL,
    text         text NOT NULL,
    embed_text   text NOT NULL,
    -- Copied from the document so filters need no join.
    acl_groups   text[] NOT NULL,
    -- The model that produced the vector; one model per column.
    model_id     text NOT NULL,
    embedding    vector(384) NOT NULL,
    -- Lexical search (Chapter 10), maintained by Postgres.
    tsv          tsvector GENERATED ALWAYS AS
                 (to_tsvector('english', embed_text)) STORED
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_acl_gin
    ON chunks USING gin (acl_groups);
-- [end:chunks]
