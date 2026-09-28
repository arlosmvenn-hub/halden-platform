"""All configuration, loaded once from the environment."""

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]


# [start:settings]
class Settings(BaseSettings):
    """Read from HALDEN_* environment variables (and .env).

    Secrets are SecretStr: they print as '**********', so a settings
    object can be logged without leaking the key.
    """

    model_config = SettingsConfigDict(
        env_prefix="HALDEN_", env_file=".env", extra="ignore"
    )

    dsn: str = "postgresql://halden:halden@localhost:5432/halden"
    # Which copy of the index to read and write (Chapter 27's
    # blue/green re-index). Deploy it together with embedding_model.
    index_schema: str = "public"
    db_pool_min: int = 1
    db_pool_max: int = 10

    anthropic_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "ANTHROPIC_API_KEY", "HALDEN_ANTHROPIC_API_KEY"
        ),
    )
    model: str | None = None  # answers (HALDEN_MODEL)
    model_small: str | None = None  # rewriting, routing

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_query_prefix: str = (
        "Represent this sentence for searching relevant passages: "
    )
    reranker_model: str | None = None  # off unless measured (Ch 11)
    retrieval_mode: Literal["dense", "hybrid"] = "hybrid"

    chunk_max_tokens: int = 120
    retrieval_candidates: int = 40
    context_budget_tokens: int = 1500
    answer_max_tokens: int = 800

    llm_timeout_s: float = 30.0
    llm_max_attempts: int = 3
    request_timeout_s: float = 60.0

    migrations_dir: Path = REPO_ROOT / "migrations"
    auto_migrate: bool = (
        False  # dev convenience; run scripts/migrate.py in prod
    )

    # Security (Chapter 26). Sources written outside Halden are
    # labeled untrusted in the prompt; answers may link only here.
    untrusted_sources: list[str] = Field(
        default_factory=lambda: ["tickets"]
    )
    allowed_link_hosts: list[str] = Field(
        default_factory=lambda: ["halden.example"]
    )

    # Cost controls (Chapter 25). None switches a feature off.
    answer_cache_ttl_s: int | None = None
    daily_token_budget: int | None = None  # per user

    # Telemetry (Chapter 24). "console" prints spans; "otlp" sends
    # them to a collector (install the "otel" extra).
    telemetry: Literal["off", "console", "otlp"] = "off"
    trace_sample_ratio: float = 1.0
    telemetry_capture_content: bool = False  # question/answer text
    telemetry_salt: SecretStr = SecretStr("rotate-me")
    otlp_endpoint: str = "http://localhost:4318/v1/traces"

    # "api_key" requires a valid key per request (Chapter 19).
    # "dev" trusts a fixed identity: local development only.
    auth_mode: Literal["api_key", "dev"] = "api_key"
    dev_user: str = "dev"
    dev_groups: list[str] = Field(default_factory=lambda: ["everyone"])


# [end:settings]
