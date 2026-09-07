from __future__ import annotations

import os
import re
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "application.toml"
CONFIG_PATH_ENV = "MULTIAGENT_CONFIG_PATH"

# Matches ${VAR} style environment variable references inside string
# values. Resolved in Settings.from_toml so secrets can live in the
# process environment instead of the committed config file.
_ENV_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _resolve_env_references(value: Any) -> Any:
    """Recursively substitute ${VAR} references in loaded config values.

    An unresolvable reference raises immediately (fail-fast) rather than
    silently degrading to an empty string.
    """
    if isinstance(value, str):
        def _substitute(match: re.Match[str]) -> str:
            var_name = match.group(1)
            if var_name not in os.environ:
                raise ValueError(
                    f"Environment variable '{var_name}' referenced in config "
                    "is not set. Export it before loading settings."
                )
            return os.environ[var_name]

        return _ENV_REFERENCE.sub(_substitute, value)
    if isinstance(value, dict):
        return {key: _resolve_env_references(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_env_references(item) for item in value]
    return value


class AppSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "multiagent"
    env: str = "development"
    debug: bool = True


class APISettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    prefix: str = "/api/v1"


class OrchestratorSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: str = "simple"
    max_graph_steps: int = Field(default=20, ge=1, le=500)


class DatabaseSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    driver: str = "postgresql+psycopg"
    host: str = "127.0.0.1"
    port: int = Field(default=5432, ge=1, le=65535)
    name: str = "multiagent_db"
    user: str = "multiagent_user"
    password: str = "replace_me"
    echo: bool = False
    pool_size: int = Field(default=10, ge=1, le=100)
    max_overflow: int = Field(default=20, ge=0, le=100)
    connect_timeout_seconds: int = Field(
        default=3,
        ge=1,
        le=60,
        description="Connection timeout used by probes and pooled connections.",
    )

    def build_url(self, *, mask_password: bool = False) -> str:
        password = "***" if mask_password else self.password
        return (
            f"{self.driver}://{self.user}:{password}"
            f"@{self.host}:{self.port}/{self.name}"
        )

    def build_connect_args(self) -> dict[str, object]:
        return {"connect_timeout": self.connect_timeout_seconds}


class RedisSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str = "127.0.0.1"
    port: int = Field(default=6379, ge=1, le=65535)
    db: int = Field(default=0, ge=0, le=32)
    username: str = ""
    password: str = ""
    decode_responses: bool = True
    connect_timeout_seconds: float = Field(
        default=5.0,
        ge=0.1,
        le=60.0,
        description="TCP connect timeout for Redis probes and runtime clients.",
    )
    socket_timeout_seconds: float = Field(
        default=5.0,
        ge=0.1,
        le=60.0,
        description="Read/write timeout for Redis probes and runtime clients.",
    )

    def build_url(
        self,
        *,
        database_index: int | None = None,
        mask_password: bool = False,
    ) -> str:
        db = self.db if database_index is None else database_index
        auth_part = ""
        if self.username:
            username = quote(self.username, safe="")
            if self.password:
                password = "***" if mask_password else quote(self.password, safe="")
                auth_part = f"{username}:{password}@"
            else:
                auth_part = f"{username}@"
        elif self.password:
            password = "***" if mask_password else quote(self.password, safe="")
            auth_part = f":{password}@"
        return f"redis://{auth_part}{self.host}:{self.port}/{db}"

    def build_client_kwargs(
        self,
        *,
        database_index: int | None = None,
    ) -> dict[str, object]:
        db = self.db if database_index is None else database_index
        return {
            "host": self.host,
            "port": self.port,
            "db": db,
            "username": self.username or None,
            "password": self.password or None,
            "decode_responses": self.decode_responses,
            "socket_connect_timeout": self.connect_timeout_seconds,
            "socket_timeout": self.socket_timeout_seconds,
        }


class CelerySettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    broker_db: int = Field(default=1, ge=0, le=32)
    result_db: int = Field(default=2, ge=0, le=32)
    task_default_queue: str = "multiagent"
    worker_pool: str = Field(
        default="",
        description=(
            "Optional Celery worker pool override. Leave empty to use the "
            "platform default (Windows -> solo, others -> prefork)."
        ),
    )


class LLMProviderSettings(BaseModel):
    """Alternative provider entry-point for multi-model routing.

    A named provider can supply its own api_base and api_key (usually via an
    environment variable reference) so agents or end users can switch models
    per task without editing the default [llm] block.
    """

    model_config = ConfigDict(extra="forbid")

    api_base: str = Field(default="", description="Provider API base URL.")
    api_key: str = Field(
        default="",
        description="Provider API key, typically a ${ENV_VAR} reference.",
    )
    model: str | None = Field(
        default=None,
        description="Default model for this provider; falls back to [llm].model.",
    )


class ModelPricingSettings(BaseModel):
    """Per-model token pricing used for cost estimation ($ per 1M tokens)."""

    model_config = ConfigDict(extra="forbid")

    input_per_million: float = Field(default=0.0, ge=0.0)
    output_per_million: float = Field(default=0.0, ge=0.0)


class LLMSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str = "openai"
    model: str = "gpt-4.1-mini"
    api_base: str = "https://api.openai.com/v1"
    api_key: str = "replace_me"
    timeout_seconds: int = Field(default=60, ge=1, le=600)
    max_retries: int = Field(
        default=2,
        ge=0,
        le=10,
        description="Retry attempts for transient LLM failures.",
    )
    retry_backoff_seconds: float = Field(
        default=1.0,
        ge=0.0,
        le=60.0,
        description="Exponential backoff multiplier between retries.",
    )
    retry_max_seconds: float = Field(
        default=8.0,
        ge=1.0,
        le=300.0,
        description="Upper bound for exponential backoff.",
    )
    providers: dict[str, LLMProviderSettings] = Field(
        default_factory=dict,
        description="Named alternative providers for multi-model routing.",
    )
    pricing: dict[str, ModelPricingSettings] = Field(
        default_factory=dict,
        description="Per-model token pricing ($ per 1M tokens) for cost estimation.",
    )


class PGVectorSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    embedding_dimension: int = Field(default=1536, ge=1)
    collection: str = "knowledge_chunks"


class MilvusSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = Field(default=19530, ge=1, le=65535)
    user: str = ""
    password: str = ""
    database: str = "default"
    collection: str = "knowledge_chunks"


class KnowledgeDocument(BaseModel):
    """One static knowledge document for the in-memory retriever."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1, max_length=128)
    title: str = ""
    content: str = Field(..., min_length=1)
    tags: list[str] = Field(default_factory=list)
    source: str = "builtin"


class KnowledgeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    retriever: Literal["inmemory"] = "inmemory"
    top_k: int = Field(default=5, ge=1, le=50)
    min_score: float = Field(default=0.05, ge=0.0, le=1.0)
    documents: list[KnowledgeDocument] = Field(
        default_factory=lambda: [
            KnowledgeDocument(
                id="kb-platform-overview",
                title="Platform Overview",
                content=(
                    "The enterprise multi-agent platform provides a reusable "
                    "agent framework with registries, orchestration and audit "
                    "for production workloads."
                ),
                tags=["platform", "overview"],
            ),
            KnowledgeDocument(
                id="kb-platform-orchestration",
                title="Orchestration Backends",
                content=(
                    "Orchestration supports a simple sequential backend and a "
                    "langgraph backend; both execute tool calls inside the "
                    "workflow and record runtime audit logs."
                ),
                tags=["orchestration"],
            ),
            KnowledgeDocument(
                id="kb-platform-memory",
                title="Memory and Review",
                content=(
                    "Layered memory keeps user, project and global scopes with "
                    "recall priority user -> project -> global, while human "
                    "review gates sensitive workflow outcomes."
                ),
                tags=["memory", "review"],
            ),
        ]
    )
    pgvector: PGVectorSettings = Field(default_factory=PGVectorSettings)
    milvus: MilvusSettings = Field(default_factory=MilvusSettings)


class ObservabilitySettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    service_name: str = "multiagent-service"
    langfuse_host: str = ""
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""


class ToolHttpSettings(BaseModel):
    """Execution policy for the real outbound http_request tool."""

    model_config = ConfigDict(extra="forbid")

    timeout_seconds: float = Field(default=15.0, ge=1.0, le=300.0)
    user_agent: str = "multiagent-agent/0.1"
    allowed_methods: list[str] = Field(
        default_factory=lambda: ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"],
    )
    max_response_bytes: int = Field(
        default=1_048_576,
        ge=1024,
        le=104_857_600,
        description="Upper bound on response body size captured into the result.",
    )
    blocked_hosts: list[str] = Field(
        default_factory=list,
        description=(
            "Hosts refused by the tool (SSRF guard). Exact host matches are "
            "blocked; cloud metadata addresses are always blocked."
        ),
    )
    follow_redirects: bool = True


class ToolSettings(BaseModel):
    """Configuration for the tool layer (real vs placeholder execution)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    http: ToolHttpSettings = Field(default_factory=ToolHttpSettings)


class MemorySettings(BaseModel):
    """Configuration for the session-context and long-term memory system."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    session_history_limit: int = Field(
        default=20,
        ge=1,
        le=200,
        description="Max previous turns injected into the next request context.",
    )
    recall_limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Max recalled memory entries injected per scope.",
    )
    auto_extract: bool = True
    project_key: str = Field(
        default="default",
        description="scope_key used for project-scoped facts.",
    )
    merge_duplicates: bool = Field(
        default=True,
        description=(
            "Merge new memory entries whose normalized content already "
            "exists in the same (memory_type, scope, scope_key) instead of "
            "appending duplicates."
        ),
    )
    merge_similarity_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum difflib ratio (0..1) for merging near-duplicate "
            "wording. 0 disables fuzzy merging; exact normalized matches "
            "always merge regardless of this value."
        ),
    )


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    app: AppSettings = Field(default_factory=AppSettings)
    api: APISettings = Field(default_factory=APISettings)
    orchestrator: OrchestratorSettings = Field(default_factory=OrchestratorSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    celery: CelerySettings = Field(default_factory=CelerySettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    knowledge: KnowledgeSettings = Field(default_factory=KnowledgeSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    tools: ToolSettings = Field(default_factory=ToolSettings)

    @classmethod
    def from_toml(cls, path: Path) -> "Settings":
        with path.open("rb") as file:
            raw_config = tomllib.load(file)
        resolved_config = _resolve_env_references(raw_config)
        return cls.model_validate(resolved_config)


def resolve_config_path(config_path: str | Path | None = None) -> Path:
    if config_path is not None:
        return Path(config_path).resolve()

    env_path = os.getenv(CONFIG_PATH_ENV)
    if env_path:
        return Path(env_path).resolve()

    return DEFAULT_CONFIG_PATH


def load_settings(config_path: str | Path | None = None) -> Settings:
    path = resolve_config_path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return Settings.from_toml(path)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()
