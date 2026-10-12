from __future__ import annotations

import os
import re
import secrets
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
    table_schema: str = Field(
        default="",
        description=(
            "Optional PostgreSQL schema unqualified tables resolve to. Empty "
            "keeps the server default (public). The test suite points this at "
            "a dedicated schema so test data never touches development data."
        ),
    )

    def build_url(self, *, mask_password: bool = False) -> str:
        password = "***" if mask_password else self.password
        url = (
            f"{self.driver}://{self.user}:{password}"
            f"@{self.host}:{self.port}/{self.name}"
        )
        if self.table_schema:
            # Render as a psycopg `options` parameter. public is deliberately
            # NOT in the search path: with a fallback, unqualified lookups of
            # `alembic_version` (and any table the suite forgets to create)
            # silently resolve against development tables in public. With no
            # fallback, a missing table fails loudly inside the test schema.
            # The '=' inside the option value is percent-encoded so it
            # survives as one query parameter.
            search_path = quote(self.table_schema, safe="")
            url = f"{url}?options=-csearch_path%3D{search_path}"
        return url

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
    visibility_timeout_seconds: int = Field(
        default=900,
        ge=60,
        le=86400,
        description=(
            "Redis broker visibility timeout: a message taken by a worker and "
            "never acknowledged is re-delivered to another worker after this "
            "many seconds. MUST be greater than the longest single task "
            "runtime, otherwise a healthy long task gets executed twice."
        ),
    )
    worker_heartbeat_interval_seconds: int = Field(
        default=30,
        ge=5,
        le=600,
        description=(
            "Interval of the worker heartbeat thread that publishes liveness "
            "to Redis. It runs outside the task thread, so a worker stuck "
            "inside a task stops heartbeating and becomes detectable."
        ),
    )
    worker_heartbeat_stale_seconds: int = Field(
        default=300,
        ge=30,
        le=3600,
        description=(
            "Heartbeat age after which a worker is considered stalled. Must be "
            "greater than the longest single task runtime plus the heartbeat "
            "interval, so that a legitimately busy worker is not misjudged."
        ),
    )
    queue_stall_seconds: int = Field(
        default=300,
        ge=60,
        le=7200,
        description=(
            "How long queued messages may wait with no healthy worker "
            "heartbeat before the API watchdog spawns a replacement worker."
        ),
    )
    worker_auto_recover: bool = Field(
        default=True,
        description=(
            "Allow the API watchdog to spawn a replacement Celery worker when "
            "the queue is backing up and no worker is heartbeating."
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

    # Deprecated backend: Phase-2 selected Milvus as the single vector store
    # and dropped pgvector as a second vector backend. Kept for config/API
    # compatibility (resources/dashboard still read the key), but disabled so
    # no pgvector-based retrieval is ever exercised.
    enabled: bool = False
    embedding_dimension: int = Field(default=1536, ge=1)
    collection: str = "knowledge_chunks"


class MilvusSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = Field(default=19530, ge=1, le=65535)
    user: str = ""
    # Leave empty for a no-auth deployment, or set it at runtime via a ${VAR}
    # reference so the secret never lives in the committed config.
    password: str = ""
    database: str = "default"
    collection: str = "knowledge_chunks"


class GraphEntitySettings(BaseModel):
    """Semantic entity extraction (the graph's LLM layer).

    One LLM call covers a whole batch of clauses from one document, and the
    answer is validated against a closed type list.  ``provider``/``model``
    name the primary leg; ``fallback_*`` describe the leg used when the primary
    stops answering (an exhausted quota is the expected case, not an
    exception).  Leaving ``fallback_provider`` empty disables degradation
    entirely, i.e. the first failure propagates.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    provider: str = "zhipu"
    # Named the way the rest of the config names models (LiteLLM provider
    # prefix included, as in [llm].model and [llm.pricing]).
    model: str = "zai/glm-5.3"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    # A reasoning model spends this budget on its trace too, so the cap must
    # cover thinking + JSON.  Hitting it is not silent: finish_reason="length"
    # splits the batch and re-asks.
    max_tokens: int = Field(default=8192, ge=256)
    timeout_seconds: int = Field(default=180, ge=1, le=600)
    # Documents in flight at once.  The provider's concurrency limit is the
    # real ceiling; the fallback provider's limit is independent of it.
    concurrency: int = Field(default=4, ge=1, le=32)
    # Batch split thresholds: the clause count bounds the answer length (every
    # ref must come back), the character budget bounds the prompt.
    max_clauses_per_call: int = Field(default=20, ge=1, le=200)
    max_chars_per_call: int = Field(default=12000, ge=500, le=200_000)
    max_split_depth: int = Field(default=3, ge=0, le=6)
    # Provider-specific request fields for the primary leg, forwarded verbatim
    # by LiteLLM (see ``LLMRequest.extra_body``).  Used in practice to bound
    # GLM-5.3's reasoning effort: it always deliberates and cannot be told to
    # stop, and at the default effort the reasoning trace consumes the whole
    # ``max_tokens`` budget so the JSON answer is never written.
    extra_body: dict[str, Any] = Field(default_factory=dict)
    # Degradation leg.  Used only when the primary leg fails; its own
    # provider-specific switches belong here rather than on the primary.
    fallback_provider: str = ""
    fallback_model: str = ""
    fallback_extra_body: dict[str, Any] = Field(default_factory=dict)


class GraphExpandSettings(BaseModel):
    """Multi-hop expansion used as a third recall leg (see graph/expand.py).

    Read side of the graph: the vector and BM25 legs can only reach clauses
    whose words match the query, and both of the graph's clause-to-clause paths
    (``CITES``, and the two-hop ``MENTIONS`` bridge) are unreachable lexically.

    ``enabled`` defaults to False so that turning it on is an explicit,
    measurable decision rather than a silent change to production retrieval;
    the two ``use_*`` switches exist so a run can attribute an effect to the
    structural layer or the entity layer instead of to "the graph" as a whole.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    # Seeds are the best-ranked clause ids of the other recall legs.  Capped
    # because every extra seed multiplies the Cypher fan-out, and the top of
    # each leg is where the relevant clauses are.
    seed_top_n: int = Field(default=10, ge=1, le=100)
    # Per-seed cap inside Cypher.  Required, not cosmetic: a clause mentioning
    # 增值税 reaches every other clause that mentions it.
    neighbors_per_seed: int = Field(default=8, ge=1, le=100)
    # Total cap on the leg's output.  The RRF pool is shared with the lexical
    # legs, so an unbounded graph leg would displace them instead of adding to
    # them.
    max_results: int = Field(default=20, ge=1, le=200)
    # Minimum entities two clauses must share for the MENTIONS bridge.  1 admits
    # any shared subject, which on a corpus this size includes very generic
    # entities; raising it is the cheap precision knob to test.
    min_shared_entities: int = Field(default=1, ge=1, le=20)
    use_cites: bool = True
    use_mentions: bool = True


class GraphSettings(BaseModel):
    """Neo4j knowledge-graph connection (Graph RAG expansion engine).

    Neo4j is deployed separately from the app host (typically a dedicated
    VM/container).  Only ``uri``/``user``/``password``/``database`` are read;
    the retrieval code performs multi-hop Cypher expansion seeded from vector
    hits, so it never depends on a specific graph schema up front.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    # Bolt URI of the remote Neo4j instance, e.g. "bolt://192.168.1.50:7687".
    # Point this at the other host that runs your Neo4j deployment.
    uri: str = "bolt://127.0.0.1:7687"
    user: str = "neo4j"
    # Prefer a ${NEO4J_PASSWORD} environment-variable reference in the config
    # so the plaintext default below never leaks to production.
    password: str = "replace_me"
    database: str = "neo4j"
    # Labels actually written by Neo4jGraphStore for the document skeleton:
    # the chunk node label and the document->chunk containment relationship.
    # The structural (Clause/CITES/...) and entity (Entity/MENTIONS) layers are
    # *not* configurable -- they are the graph contract, see neo4j.py.
    chunk_entity_label: str = "KnowledgeChunk"
    relation_label: str = "CONTAINS"
    entity: GraphEntitySettings = Field(default_factory=GraphEntitySettings)
    expand: GraphExpandSettings = Field(default_factory=GraphExpandSettings)


class KnowledgeIngestSettings(BaseModel):
    """Document ingestion pipeline config for the fiscal/tax regulatory corpus.

    The corpus is word-doc dominated (.docx), plus PDF and Excel (.xlsx).
    ``primary_extension`` drives which parser the ingestion task picks first.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    # Supported inputs for the parse -> chunk -> embed -> (index|graph) pipeline.
    primary_extension: str = "docx"
    accepted_extensions: list[str] = Field(
        default_factory=lambda: [".docx", ".pdf", ".xlsx"],
    )
    # Native file dialogs or a schedule can point the ingestion watcher here.
    source_dir: str = ""
    chunk_size_chars: int = Field(default=900, ge=100, le=8000)
    chunk_overlap_chars: int = Field(default=120, ge=0, le=2000)
    # Two-level chunking: ``chunk_size_chars`` bounds the *parent* (a whole
    # "第X条" clause), while this bounds the retrievable *child* -- a sentence
    # window inside the clause.  Smaller children rank more precisely while the
    # caller still receives the whole clause.  Must not exceed
    # ``chunk_size_chars`` (the chunker rejects that instead of silently
    # degenerating to one child per clause).
    child_window_chars: int = Field(default=400, ge=100, le=8000)
    # Mongo-like collection for graph-write batching; passthrough only.
    batch_size: int = Field(default=16, ge=1, le=256)


class KnowledgeDocument(BaseModel):
    """One static knowledge document for the in-memory retriever."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1, max_length=128)
    title: str = ""
    content: str = Field(..., min_length=1)
    tags: list[str] = Field(default_factory=list)
    source: str = "builtin"


class KnowledgeEvaluationSettings(BaseModel):
    """RAGAS evaluation of the retrieval + generation chain.

    The judge is deliberately independent of ``[llm]``: scoring a DeepSeek-
    answered query with a DeepSeek judge shares the same blind spots, so the
    two roles are configured separately.  A named provider contributes only
    endpoint + credentials (``ZHIPU_API_KEY`` arrives that way) — the judge
    model name lives here so switching judges never touches secrets.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    judge_provider: str = "zhipu"
    # Judge model, named the way the rest of the config names models (LiteLLM
    # provider prefix included, as in [llm].model and [llm.pricing]).  The
    # RAGAS scorer talks to the provider's OpenAI-compatible endpoint directly,
    # so it strips the prefix before sending the id on.  GLM-5.3 is chosen for
    # its structured-output compliance, which is what RAGAS metric prompts need
    # (a judge that fails to emit valid JSON yields a parse error, not a score).
    judge_model: str = "zai/glm-5.3"
    # temperature 0 keeps re-runs comparable: a score change should mean the
    # system changed, not the judge.
    judge_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    judge_timeout_seconds: int = Field(default=180, ge=1, le=600)
    # How the judge adapter asks the judge to limit deliberation.  This is
    # provider-specific and therefore configurable: GLM-5.3 always reasons
    # (``thinking`` only accepts "enabled") so it is capped with the OpenAI
    # ``reasoning_effort`` field, while DeepSeek takes a ``thinking`` switch in
    # ``judge_extra_body`` and would either reject or ignore ``reasoning_effort``.
    # Empty means "send neither field" -- the adapter must never hand one
    # vendor's knob to another vendor's endpoint.
    judge_reasoning_effort: str = "low"
    # Provider-specific request fields for the judge, forwarded to the
    # OpenAI-compatible endpoint verbatim.  Same rationale as
    # ``answer_extra_body``: the field that turns deliberation off belongs in
    # config, not hardcoded in the scorer.
    judge_extra_body: dict[str, Any] = Field(default_factory=dict)
    # Judge calls in flight at once.  Each sample costs several judge calls, so
    # this is the run's throughput/cost knob; the provider's rate limit is the
    # real ceiling.
    concurrency: int = Field(default=2, ge=1, le=16)
    # The generation leg under test.  Empty values fall back to [llm].
    answer_provider: str = ""
    answer_model: str = ""
    answer_top_k: int = Field(default=5, ge=1, le=20)
    answer_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    answer_max_tokens: int = Field(default=1200, ge=1)
    # Provider-specific request fields for the generation leg, forwarded to the
    # provider verbatim (see ``LLMRequest.extra_body``).  The generation leg does
    # not need deliberation, and DeepSeek's thinking chain shares ``max_tokens``
    # with the answer, so the switch that turns it off belongs here rather than
    # hardcoded in the answerer: swapping the generation model must not send a
    # DeepSeek-only field to another vendor's endpoint.
    answer_extra_body: dict[str, Any] = Field(default_factory=dict)
    max_context_chars: int = Field(default=12000, ge=500)


class KnowledgeChatSettings(BaseModel):
    """Production answer policy, independent of evaluation/judge settings."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=3000, ge=1)
    max_context_chars: int = Field(default=24000, ge=500)
    timeout_seconds: int = Field(default=120, ge=1, le=600)
    provider_extra_body: dict[str, dict[str, Any]] = Field(default_factory=dict)


class KnowledgeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Active retrieval backend. "inmemory" is the P1 baseline; "milvus",
    # "graph" and "hybrid" are wired into the ingestion/retrieval changes of
    # phase P2 (Milvus vector recall + Neo4j multi-hop graph expansion).
    retriever: Literal["inmemory", "milvus", "graph", "hybrid"] = "inmemory"
    chunk_snapshot: str = "data/knowledge/processed/chunks.jsonl"
    chat: KnowledgeChatSettings = Field(default_factory=KnowledgeChatSettings)
    top_k: int = Field(default=5, ge=1, le=50)
    min_score: float = Field(default=0.05, ge=0.0, le=1.0)
    # Local embedding / rerankers (sentence-transformers). Embedding dim must
    # match the Milvus collection dimension (EMBEDDING_DIM in milvus store).
    embedding_model: str = "BAAI/bge-base-zh-v1.5"
    embedding_dim: int = Field(default=768, ge=1)
    reranker_model: str = "BAAI/bge-reranker-base"
    # Hybrid fusion: how many candidates to take from the vector leg and the
    # BM25 leg before merging + reranking.
    hybrid_candidates: int = Field(default=8, ge=1, le=50)
    # Pre-fusion (RRF) stage: merge the recall legs by reciprocal rank before
    # the cross-encoder runs.  ``rrf_k`` is the standard RRF smoothing constant;
    # ``prefusion_top_n`` caps how many fused candidates reach the (expensive)
    # cross-encoder, so keep it >= top_k.  It is a no-op while the fused pool is
    # smaller than the cap (e.g. small samples).
    rrf_k: int = Field(default=60, ge=1)
    prefusion_top_n: int = Field(default=20, ge=1, le=200)
    # Diversity guard on the pre-rerank pool: at most this many chunks per
    # source document.  A long statute (民法典 ≈ 1.2e3 chunks) otherwise
    # occupies every pool slot and the cross-encoder can only choose among its
    # own fragments, so precision on mixed-corpus queries collapses.  The pool
    # is backfilled from the skipped candidates when the corpus genuinely lacks
    # diversity, so this never shrinks the pool below ``prefusion_top_n``.
    max_chunks_per_document: int = Field(default=3, ge=1, le=50)
    # Supplementary recall for what the quota above rejects: candidates that
    # passed both recall legs (so the retriever did judge them relevant) and were
    # dropped only because their source document had already used its slots.
    # They are returned after the primary top_k and flagged
    # (``QUOTA_OVERFLOW_KEY``) so a consumer can offer them as hints: they never
    # reach the cross-encoder, so treating them as equal evidence would silently
    # undo the diversity guard.  The case they exist for is a question whose
    # answer spans several clauses of one statute, where the article holding the
    # answer loses to its own neighbours.  0 keeps the channel off.
    quota_overflow_top_n: int = Field(default=0, ge=0, le=20)
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
    graph: GraphSettings = Field(default_factory=GraphSettings)
    ingest: KnowledgeIngestSettings = Field(default_factory=KnowledgeIngestSettings)
    evaluation: KnowledgeEvaluationSettings = Field(
        default_factory=KnowledgeEvaluationSettings
    )


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


class AuthSettings(BaseModel):
    """Login/session configuration for the console permission system."""

    model_config = ConfigDict(extra="forbid")

    session_secret: str = Field(default_factory=lambda: secrets.token_urlsafe(32), min_length=32)
    session_cookie_name: str = "ma_session"
    session_ttl_seconds: int = Field(default=7 * 24 * 3600, ge=60)


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
    auth: AuthSettings = Field(default_factory=AuthSettings)

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
