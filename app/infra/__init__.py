"""Infrastructure access layer."""

from app.infra.celery_broker import BrokerBacklogSnapshot, snapshot_broker_backlog
from app.infra.database import get_database_connection_config
from app.infra.knowledge import get_vectorstore_configs
from app.infra.llm import get_litellm_config
from app.infra.redis import (
    build_broker_client,
    get_celery_redis_urls,
    get_redis_connection_config,
    is_redis_reachable,
    probe_redis_connection,
)
from app.infra.resources import (
    AppResources,
    ResourceHandle,
    ResourceStatus,
    close_app_resources,
    initialize_app_resources,
)
from app.infra.worker_health import (
    WorkerHeartbeat,
    read_worker_heartbeats,
    start_worker_heartbeat,
    stop_worker_heartbeat,
)

__all__ = [
    "AppResources",
    "BrokerBacklogSnapshot",
    "ResourceHandle",
    "ResourceStatus",
    "WorkerHeartbeat",
    "build_broker_client",
    "close_app_resources",
    "get_celery_redis_urls",
    "get_database_connection_config",
    "get_litellm_config",
    "get_redis_connection_config",
    "get_vectorstore_configs",
    "initialize_app_resources",
    "is_redis_reachable",
    "probe_redis_connection",
    "read_worker_heartbeats",
    "snapshot_broker_backlog",
    "start_worker_heartbeat",
    "stop_worker_heartbeat",
]
