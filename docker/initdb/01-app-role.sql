-- Runs once, as the database superuser, when the database is new
-- (Docker Compose mounts this folder into the Postgres image, and
-- CI runs it with psql). The application must not connect as a
-- superuser: superusers bypass row-level security, even FORCE
-- ROW LEVEL SECURITY (Chapter 19).
CREATE EXTENSION IF NOT EXISTS vector;
CREATE ROLE halden LOGIN PASSWORD 'halden';  -- local development only
ALTER DATABASE halden OWNER TO halden;
