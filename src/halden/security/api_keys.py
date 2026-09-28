"""API keys: create, store as hashes, authenticate."""

import hashlib
import hmac
import secrets

from psycopg_pool import AsyncConnectionPool

from halden.security.identity import Identity

PREFIX = "hk_"


# [start:keys]
def _hash(secret: str) -> str:
    # A fast hash is appropriate here only because the secret is
    # 256 random bits: unguessable, unlike a human password, which
    # needs a slow, salted hash (bcrypt, scrypt, argon2).
    return hashlib.sha256(secret.encode()).hexdigest()


async def create_key(pool: AsyncConnectionPool, user_id: str) -> str:
    """Return the plaintext key. It is never stored or shown again."""
    key_id = secrets.token_hex(6)
    secret = secrets.token_urlsafe(32)
    async with pool.connection() as conn:
        await conn.execute(
            "INSERT INTO api_keys (key_id, user_id, secret_hash)"
            " VALUES (%s, %s, %s)",
            (key_id, user_id, _hash(secret)),
        )
    return f"{PREFIX}{key_id}_{secret}"


async def authenticate(
    pool: AsyncConnectionPool, presented: str
) -> Identity | None:
    """Identity for a valid, unrevoked key; None otherwise.

    Every failure looks the same to the caller, so responses do
    not reveal whether a key ID exists.
    """
    if not presented.startswith(PREFIX):
        return None
    key_id, _, secret = presented[len(PREFIX) :].partition("_")
    async with pool.connection() as conn:
        row = await (
            await conn.execute(
                "SELECT k.user_id, k.secret_hash,"
                " coalesce(array_agg(g.group_name)"
                "   FILTER (WHERE g.group_name IS NOT NULL), '{}'),"
                " u.tenant_id"
                " FROM api_keys k JOIN users u USING (user_id)"
                " LEFT JOIN user_groups g USING (user_id)"
                " WHERE k.key_id = %s AND k.revoked_at IS NULL"
                " GROUP BY k.user_id, k.secret_hash, u.tenant_id",
                (key_id,),
            )
        ).fetchone()
    # Constant-time comparison: timing must not leak hash prefixes.
    if row is None or not hmac.compare_digest(row[1], _hash(secret)):
        return None
    return Identity(user_id=row[0], groups=list(row[2]), tenant=row[3])


# [end:keys]


async def create_user(
    pool: AsyncConnectionPool,
    user_id: str,
    name: str,
    groups: list[str],
) -> None:
    async with pool.connection() as conn, conn.transaction():
        await conn.execute(
            "INSERT INTO users (user_id, display_name) VALUES (%s, %s)"
            " ON CONFLICT (user_id) DO UPDATE"
            " SET display_name = EXCLUDED.display_name",
            (user_id, name),
        )
        await conn.execute(
            "DELETE FROM user_groups WHERE user_id = %s", (user_id,)
        )
        for g in groups:
            await conn.execute(
                "INSERT INTO user_groups VALUES (%s, %s)", (user_id, g)
            )
