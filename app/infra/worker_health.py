"""Worker liveness registry backed by the Celery broker Redis DB.

Celery's own ``inspect ping`` answers only while a worker can still run its
event loop, which leaves two blind spots: a worker wedged inside a task may
either keep replying "pong" or stop replying in a way that is identical to
"no worker configured at all". Neither state is visible from the API, which is
how a message ended up taken by a stale process and never acknowledged while
nothing reported a problem.

This module keeps an independent, TTL-based heartbeat per worker process,
written from a daemon thread that is *not* the task thread. A worker busy with
a long (but healthy) task keeps beating; a worker stuck inside a task stops.
Readers (the API watchdog and ``/system/workers``) therefore get one bit that
Celery cannot provide: "is any worker actually alive right now?".
"""

from __future__ import annotations

import json
import os
import socket
import threading
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.core import Settings, get_settings
from app.infra.redis import build_broker_client, get_celery_redis_urls

HEARTBEAT_KEY_PREFIX = "multiagent:worker:heartbeat:"

_heartbeat_lock = threading.Lock()
_heartbeat_thread: threading.Thread | None = None
_heartbeat_stop = threading.Event()
_heartbeat_worker_id = ""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def build_worker_id(hostname: str | None = None, pid: int | None = None) -> str:
    resolved_host = hostname or socket.gethostname()
    resolved_pid = os.getpid() if pid is None else pid
    return f"{resolved_host}:{resolved_pid}"


def heartbeat_key(worker_id: str) -> str:
    return f"{HEARTBEAT_KEY_PREFIX}{worker_id}"


class WorkerHeartbeat(BaseModel):
    model_config = ConfigDict(extra="forbid")

    worker_id: str
    hostname: str = ""
    pid: int = 0
    pool: str = ""
    started_at: datetime = Field(default_factory=utc_now)
    beat_at: datetime = Field(default_factory=utc_now)

    def age_seconds(self, now: datetime | None = None) -> float:
        reference = now or utc_now()
        return max(0.0, (reference - self.beat_at).total_seconds())

    def is_stale(self, stale_seconds: float, now: datetime | None = None) -> bool:
        return self.age_seconds(now) > stale_seconds


def read_worker_heartbeats(
    settings: Settings | None = None,
    *,
    client: Any | None = None,
) -> list[WorkerHeartbeat]:
    """Return every heartbeat currently present in the broker DB.

    Unreadable or malformed entries are skipped rather than raising: a broken
    heartbeat must never be able to take down the diagnostic view that exists
    to detect broken workers.
    """
    current_settings = settings or get_settings()
    owned_client = client is None
    broker_client = client if client is not None else build_broker_client(current_settings)

    try:
        keys = list(broker_client.scan_iter(match=f"{HEARTBEAT_KEY_PREFIX}*", count=100))
        if not keys:
            return []
        payloads = broker_client.mget(keys)
    finally:
        if owned_client:
            broker_client.close()

    heartbeats: list[WorkerHeartbeat] = []
    for payload in payloads:
        if not payload:
            continue
        try:
            raw = json.loads(payload) if isinstance(payload, (str, bytes)) else payload
            heartbeats.append(WorkerHeartbeat.model_validate(raw))
        except Exception:  # pragma: no cover - defensive parse boundary
            continue
    heartbeats.sort(key=lambda item: item.beat_at)
    return heartbeats


def start_worker_heartbeat(settings: Settings | None = None) -> str:
    """Start the daemon heartbeat thread and return this worker's id.

    Idempotent: calling it again from the same process keeps the running thread
    so a re-entrant ``worker_ready`` cannot spawn duplicates.
    """
    global _heartbeat_thread, _heartbeat_worker_id

    current_settings = settings or get_settings()
    worker_id = build_worker_id()

    with _heartbeat_lock:
        if _heartbeat_thread is not None and _heartbeat_thread.is_alive():
            return _heartbeat_worker_id
        _heartbeat_worker_id = worker_id
        _heartbeat_stop.clear()
        _heartbeat_thread = threading.Thread(
            target=_heartbeat_loop,
            args=(current_settings, worker_id, _heartbeat_stop),
            name="multiagent-worker-heartbeat",
            daemon=True,
        )
        thread = _heartbeat_thread

    thread.start()
    return worker_id


def stop_worker_heartbeat() -> None:
    """Stop the heartbeat thread and remove this worker's key."""
    global _heartbeat_thread, _heartbeat_worker_id

    with _heartbeat_lock:
        thread = _heartbeat_thread
        worker_id = _heartbeat_worker_id
        _heartbeat_thread = None

    if thread is None:
        return

    _heartbeat_stop.set()
    thread.join(timeout=5.0)

    if not worker_id:
        return
    try:
        client = build_broker_client()
    except Exception:  # pragma: no cover - shutdown best effort
        return
    try:
        client.delete(heartbeat_key(worker_id))
    except Exception:  # pragma: no cover - shutdown best effort
        pass
    finally:
        client.close()


def _heartbeat_loop(settings: Settings, worker_id: str, stop: threading.Event) -> None:
    interval = settings.celery.worker_heartbeat_interval_seconds
    # Keep the key around a little past the staleness threshold so a dead
    # worker is observable as "stale since T" instead of silently vanishing.
    ttl = settings.celery.worker_heartbeat_stale_seconds + interval
    hostname = socket.gethostname()
    pid = os.getpid()
    pool = get_celery_redis_urls(settings)["worker_pool"]
    started_at = utc_now()

    client: Any | None = None
    try:
        while not stop.is_set():
            try:
                if client is None:
                    client = build_broker_client(settings)
                payload = WorkerHeartbeat(
                    worker_id=worker_id,
                    hostname=hostname,
                    pid=pid,
                    pool=pool,
                    started_at=started_at,
                    beat_at=utc_now(),
                ).model_dump_json()
                client.set(heartbeat_key(worker_id), payload, ex=ttl)
            except Exception:
                # A transient broker outage must not kill the thread: drop the
                # connection so the next tick reconnects from scratch.
                if client is not None:
                    try:
                        client.close()
                    except Exception:
                        pass
                    client = None
            stop.wait(interval)
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
