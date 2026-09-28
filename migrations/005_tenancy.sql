-- 005: tenants. Shared tables, a tenant column, row-level security.

-- [start:columns]
ALTER TABLE users ADD COLUMN tenant_id text NOT NULL DEFAULT 'halden';
ALTER TABLE documents
    ADD COLUMN tenant_id text NOT NULL DEFAULT 'halden';
ALTER TABLE chunks ADD COLUMN tenant_id text NOT NULL DEFAULT 'halden';
ALTER TABLE answer_cache
    ADD COLUMN tenant_id text NOT NULL DEFAULT 'halden';

-- Existing rows now belong to Halden. New rows take the tenant of
-- the transaction that writes them; with no tenant declared,
-- current_setting() raises and the insert fails: fail closed.
ALTER TABLE documents
    ALTER COLUMN tenant_id SET DEFAULT current_setting('halden.tenant');
ALTER TABLE chunks
    ALTER COLUMN tenant_id SET DEFAULT current_setting('halden.tenant');
-- [end:columns]

-- [start:policies]
CREATE FUNCTION halden_tenant() RETURNS text LANGUAGE sql STABLE
AS $$ SELECT current_setting('halden.tenant', true) $$;

CREATE FUNCTION halden_groups() RETURNS text[] LANGUAGE sql STABLE
AS $$ SELECT coalesce(nullif(current_setting('halden.user_groups',
    true), '')::text[], '{}'::text[]) $$;

CREATE FUNCTION halden_writer() RETURNS boolean LANGUAGE sql STABLE
AS $$ SELECT current_setting('halden.writer', true) = 'on' $$;

DROP POLICY chunks_reader ON chunks;
DROP POLICY chunks_writer ON chunks;
CREATE POLICY chunks_reader ON chunks FOR SELECT USING (
    tenant_id = halden_tenant() AND acl_groups && halden_groups());
CREATE POLICY chunks_writer ON chunks
    USING (halden_writer() AND tenant_id = halden_tenant())
    WITH CHECK (halden_writer() AND tenant_id = halden_tenant());

ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents FORCE ROW LEVEL SECURITY;
CREATE POLICY documents_reader ON documents FOR SELECT USING (
    tenant_id = halden_tenant() AND acl_groups && halden_groups());
CREATE POLICY documents_writer ON documents
    USING (halden_writer() AND tenant_id = halden_tenant())
    WITH CHECK (halden_writer() AND tenant_id = halden_tenant());
-- [end:policies]
