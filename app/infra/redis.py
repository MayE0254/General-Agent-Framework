from __future__ import annotations

import sys
from time import perf_counter

from app.core import Settings, get_settings


def probe_redis_connection(settings: Settings | None = None) -> dict[str, object]:
    """Return a structured Redis probe result for diagnostics and health views."""
    import redis as redis_client

    current_settings = settings or get_settings()
    redis_settings = current_settings.redis
    started_at = perf_counter()
    client = None

    try:
        client = redis_client.Redis(**redis_settings.build_client_kwargs())
        reachable = bool(client.ping())
        return {
            "reachable": reachable,
            "latency_ms": round((perf_counter() - started_at) * 1000, 2),
            "error_type": "",
            "error_message": "",
        }
    except Exception as exc:
        return {
            "reachable": False,
            "latency_ms": round((perf_counter() - started_at) * 1000, 2),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
    finally:
        if client is not None:
            client.close()


def is_redis_reachable(settings: Settings | None = None) -> bool:
    """Probe the configured Redis server with a short connect timeout.

    Used by the async task API: submission is refused (not silently queued
    locally) when the broker is unreachable.
    """
    return bool(probe_redis_connection(settings).get("reachable"))


def get_redis_connection_config(settings: Settings | None = None) -> dict[str, object]:
    current_settings = settings or get_settings()
    redis_settings = current_settings.redis

    return {
        "url": redis_settings.build_url(),
        "decode_responses": redis_settings.decode_responses,
        "connect_timeout_seconds": redis_settings.connect_timeout_seconds,
        "socket_timeout_seconds": redis_settings.socket_timeout_seconds,
    }


def get_celery_redis_urls(settings: Settings | None = None) -> dict[str, str]:
    current_settings = settings or get_settings()
    redis_settings = current_settings.redis
    celery_settings = current_settings.celery

    def build_url(database_index: int) -> str:
        return redis_settings.build_url(database_index=database_index)

    return {
        "broker_url": build_url(celery_settings.broker_db),
        "result_backend": build_url(celery_settings.result_db),
        "task_default_queue": celery_settings.task_default_queue,
        "worker_pool": _resolve_worker_pool(current_settings),
    }


def _resolve_worker_pool(settings: Settings) -> str:
    configured = settings.celery.worker_pool.strip().lower()
    if configured:
        return configured
    if sys.platform.startswith("win"):
        return "solo"
    return "prefork"
