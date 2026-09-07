from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.core import Settings, get_settings
from app.infra.database import (
    build_database_runtime,
    get_database_connection_config,
)
from app.infra.knowledge import get_vectorstore_configs
from app.infra.llm import get_litellm_config
from app.infra.redis import (
    get_celery_redis_urls,
    get_redis_connection_config,
    probe_redis_connection,
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ResourceStatus(StrEnum):
    READY = "ready"
    DEGRADED = "degraded"
    CLOSED = "closed"


class ResourceHandle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    status: ResourceStatus = Field(default=ResourceStatus.READY)
    config: dict[str, Any] = Field(default_factory=dict)
    message: str = ""


class AppResources(BaseModel):
    model_config = ConfigDict(extra="forbid")

    database: ResourceHandle
    database_engine: Any | None = None
    database_session_factory: Any | None = None
    redis: ResourceHandle
    llm: ResourceHandle
    vectorstores: dict[str, ResourceHandle]
    started_at: datetime = Field(default_factory=utc_now)
    stopped_at: datetime | None = None

    def health_summary(self) -> dict[str, str]:
        summary = {
            "database": self.database.status.value,
            "redis": self.redis.status.value,
            "llm": self.llm.status.value,
        }
        summary.update(
            {
                f"vectorstore:{name}": resource.status.value
                for name, resource in self.vectorstores.items()
            }
        )
        return summary


def initialize_app_resources(settings: Settings | None = None) -> AppResources:
    current_settings = settings or get_settings()
    vectorstore_configs = get_vectorstore_configs(current_settings)
    database_runtime = build_database_runtime(current_settings)
    redis_probe = probe_redis_connection(current_settings)
    database_status = (
        ResourceStatus.READY if database_runtime["probe_ok"] else ResourceStatus.DEGRADED
    )
    redis_status = (
        ResourceStatus.READY if redis_probe["reachable"] else ResourceStatus.DEGRADED
    )
    redis_message = (
        f"Redis probe succeeded in {redis_probe['latency_ms']} ms."
        if redis_probe["reachable"]
        else (
            "Redis probe failed: "
            f"{redis_probe['error_type']}: {redis_probe['error_message']}"
        )
    )

    return AppResources(
        database=ResourceHandle(
            name="database",
            status=database_status,
            config={
                **get_database_connection_config(current_settings),
                "sqlalchemy_available": database_runtime["sqlalchemy_available"],
            },
            message=database_runtime["probe_message"],
        ),
        database_engine=database_runtime["engine"],
        database_session_factory=database_runtime["session_factory"],
        redis=ResourceHandle(
            name="redis",
            status=redis_status,
            config={
                **get_redis_connection_config(current_settings),
                **get_celery_redis_urls(current_settings),
                "probe": redis_probe,
            },
            message=redis_message,
        ),
        llm=ResourceHandle(
            name="llm",
            status=ResourceStatus.READY,
            config=get_litellm_config(current_settings),
            message="LiteLLM runtime config prepared.",
        ),
        vectorstores={
            name: ResourceHandle(
                name=name,
                status=ResourceStatus.READY,
                config=config,
                message=(
                    "Vector store config prepared for runtime initialization."
                ),
            )
            for name, config in vectorstore_configs.items()
        },
    )


def close_app_resources(resources: AppResources) -> AppResources:
    if resources.database_engine is not None:
        resources.database_engine.dispose()
    resources.database.status = ResourceStatus.CLOSED
    resources.redis.status = ResourceStatus.CLOSED
    resources.llm.status = ResourceStatus.CLOSED
    for resource in resources.vectorstores.values():
        resource.status = ResourceStatus.CLOSED
    resources.stopped_at = utc_now()
    return resources
