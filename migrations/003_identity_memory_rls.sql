-- 003: users, API keys, conversation memory, row-level security.

CREATE TABLE users (
    user_id      text PRIMARY KEY,
    display_name text NOT NULL,
    created_at   timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE user_groups (
    user_id    text NOT NULL REFERENCES users ON DELETE CASCADE,
    group_name text NOT NULL,
    PRIMARY KEY (user_id, group_name)
);

-- [start:keys]
-- Keys look like hk_<key_id>_<secret>. Only a hash of the secret
-- is stored; the plaintext is shown once, at creation.
CREATE TABLE api_keys (
    key_id      text PRIMARY KEY,
    user_id     text NOT NULL REFERENCES users ON DELETE CASCADE,
    secret_hash text NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now(),
    revoked_at  timestamptz
);
-- [end:keys]

CREATE TABLE conversations (
    conversation_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         text NOT NULL REFERENCES users ON DELETE CASCADE,
    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE messages (
    id              bigserial PRIMARY KEY,
    conversation_id uuid NOT NULL
                    REFERENCES conversations ON DELETE CASCADE,
    role            text NOT NULL CHECK (role IN ('user', 'assistant')),
    text            text NOT NULL,
    created_at      timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX messages_by_conversation ON messages (conversation_id, id);

-- [start:rls]
-- Defense in depth: even a query that forgets its permission
-- filter cannot read chunks outside the caller's groups. The
-- groups come from a per-transaction setting the application sets
-- from the authenticated identity. Unset means no groups: fail
-- closed. FORCE applies the policies to the table owner too.
ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunks FORCE ROW LEVEL SECURITY;

CREATE POLICY chunks_reader ON chunks FOR SELECT USING (
    acl_groups && coalesce(
        nullif(current_setting('halden.user_groups', true), '')::text[],
        '{}'::text[])
);

-- Ingestion transactions declare themselves as writers.
CREATE POLICY chunks_writer ON chunks
    USING (current_setting('halden.writer', true) = 'on')
    WITH CHECK (current_setting('halden.writer', true) = 'on');
-- [end:rls]
