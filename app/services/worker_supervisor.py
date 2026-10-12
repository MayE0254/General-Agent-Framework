"""Celery worker supervision: stall detection and self-healing.

Background: a message was once taken from the queue by a leftover worker
process, which then wedged. Because the task ran with ``acks_late`` on a
single-process ``solo`` pool, the message stayed in the broker's *unacked*
hash; and because Redis only re-delivers unacked messages during a live
worker's periodic restore sweep, no healthy process was left to notice. The
job was simply gone, and every Celery ``inspect``-based view said "no workers"
without saying "something is stuck".

This module removes the "nobody notices" part:

* :func:`summarize_worker_health` combines the independent heartbeats
  (written outside the task thread, see ``app.infra.worker_health``) with the
  broker backlog and unacked age (``app.infra.celery_broker``).
* :class:`WorkerWatchdog` runs that summary on a timer inside the API process
  and spawns a replacement worker when the queue is clearly stalled and no
  worker is heartbeating — which is also what makes the broker restore the
  stuck unacked message.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, ConfigDict, Field

from app.core import Settings, get_settings
from app.infra.celery_broker import BrokerBacklogSnapshot, snapshot_broker_backlog
from app.infra.worker_health import WorkerHeartbeat, read_worker_heartbeats

logger = logging.getLogger(__name__)

_PROXY_ENV_KEYS = {"http_proxy", "https_proxy", "all_proxy"}
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# A stall window may spawn at most this many workers before the watchdog backs
# off and keeps reporting only. Prevents an unbounded spawn loop when a worker
# cannot start at all (bad config, missing broker, port conflicts).
_MAX_SPAWN_ATTEMPTS = 3


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def spawn_detached_worker() -> int:
    """Start a detached Celery worker the same way ``scripts/start_all.ps1`` does."""
    command = [
        sys.executable,
        "-m",
        "celery",
        "-A",
        "app.tasks.celery_app",
        "worker",
        "-l",
        "info",
    ]
    # Drop proxy env vars: a stale proxy inherited from the parent process
    # breaks every outbound LLM call made by workflow tasks.
    env = {
        key: value
        for key, value in os.environ.items()
        if key.lower() not in _PROXY_ENV_KEYS
    }
    log_dir = _PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    creation_flags = 0
    if sys.platform == "win32":
        creation_flags = (
            subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        )
    with open(log_dir / "worker.out.log", "ab") as out, open(
        log_dir / "worker.err.log", "ab"
    ) as err:
        process = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
            command,
            cwd=str(_PROJECT_ROOT),
            stdout=out,
            stderr=err,
            stdin=subprocess.DEVNULL,
            env=env,
            creationflags=creation_flags,
        )
    return process.pid


class WorkerHealthSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str = "ready"
    message: str = ""
    heartbeats: list[WorkerHeartbeat] = Field(default_factory=list)
    live_worker_ids: list[str] = Field(default_factory=list)
    stale_worker_ids: list[str] = Field(default_factory=list)
    backlog: BrokerBacklogSnapshot = Field(default_factory=BrokerBacklogSnapshot)
    stall_detected: bool = False


def summarize_worker_health(settings: Settings | None = None) -> WorkerHealthSummary:
    """Combine heartbeats and broker backlog into one supervised view."""
    current_settings = settings or get_settings()
    stale_seconds = current_settings.celery.worker_heartbeat_stale_seconds

    try:
        heartbeats = read_worker_heartbeats(current_settings)
    except Exception as exc:  # noqa: BLE001 - a health view must never raise
        heartbeats = []
        logger.warning("worker heartbeat read failed: %s", exc)

    backlog = snapshot_broker_backlog(current_settings)

    live = [item.worker_id for item in heartbeats if not item.is_stale(stale_seconds)]
    stale = [item.worker_id for item in heartbeats if item.is_stale(stale_seconds)]
    stall_detected = is_stalled(backlog, current_settings) if not live else False

    if not backlog.reachable:
        status = "degraded"
        message = f"无法读取 broker 队列状态：{backlog.error}"
    elif live:
        status = "ready"
        message = f"{len(live)} 个 worker 心跳正常。"
    elif stale:
        status = "stalled"
        message = (
            f"{len(stale)} 个 worker 心跳已过期（>{stale_seconds}s），"
            "当前无活跃消费者。"
        )
    else:
        status = "offline"
        message = "未发现任何 worker 心跳记录。"

    if stall_detected:
        status = "stalled"

    return WorkerHealthSummary(
        status=status,
        message=message,
        heartbeats=heartbeats,
        live_worker_ids=live,
        stale_worker_ids=stale,
        backlog=backlog,
        stall_detected=stall_detected,
    )


def is_stalled(
    backlog: BrokerBacklogSnapshot,
    settings: Settings | None = None,
) -> bool:
    """True when messages are waiting with no consumer able to make progress.

    Covers both halves of the original incident: queued messages nobody took,
    and a taken-but-never-acknowledged message that only a live worker can
    restore (the broker itself never re-delivers on a timer).
    """
    if not backlog.reachable:
        return False
    if backlog.ready > 0:
        return True
    if backlog.unacked == 0:
        return False

    current_settings = settings or get_settings()
    visibility_timeout = current_settings.celery.visibility_timeout_seconds
    age = backlog.oldest_unacked_age_seconds
    if age is None:
        return False
    # Within the visibility timeout the message belongs to a worker that may
    # still be working on it; past it, only a live restore sweep can save it.
    return age > visibility_timeout


class WorkerWatchdogState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    running: bool = False
    started_at: datetime = Field(default_factory=utc_now)
    last_check_at: datetime | None = None
    checks: int = 0
    recoveries: int = 0
    spawn_attempts: int = 0
    last_action: str = ""
    last_action_at: datetime | None = None
    last_message: str = ""
    spawned_pids: list[int] = Field(default_factory=list)


class WorkerWatchdog:
    """Periodic stall check that can spawn a replacement worker."""

    def __init__(
        self,
        settings: Settings,
        *,
        spawn: Callable[[], int] = spawn_detached_worker,
    ) -> None:
        self._settings = settings
        self._spawn = spawn
        self._task: asyncio.Task[None] | None = None
        self._stopping = asyncio.Event()
        self._stall_since: datetime | None = None
        self.state = WorkerWatchdogState()

    @property
    def interval_seconds(self) -> float:
        configured = self._settings.celery.queue_stall_seconds
        return float(max(10, min(60, configured // 3)))

    def start(self) -> WorkerWatchdog:
        if self._task is None:
            self.state.running = True
            self._task = asyncio.create_task(self.run(), name="worker-watchdog")
        return self

    async def stop(self) -> None:
        self._stopping.set()
        task = self._task
        self._task = None
        self.state.running = False
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def run(self) -> None:
        interval = self.interval_seconds
        while not self._stopping.is_set():
            try:
                summary = await asyncio.to_thread(summarize_worker_health, self._settings)
                self._react(summary)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - the loop must survive
                logger.warning("worker watchdog tick failed: %s", exc)
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=interval)
            except asyncio.TimeoutError:
                continue

    def _react(self, summary: WorkerHealthSummary) -> None:
        now = utc_now()
        self.state.checks += 1
        self.state.last_check_at = now

        if summary.live_worker_ids:
            if self._stall_since is not None:
                logger.info(
                    "worker watchdog: consumer restored (%s)",
                    ", ".join(summary.live_worker_ids),
                )
            self._stall_since = None
            self.state.spawn_attempts = 0
            if self.state.last_action != "healthy":
                self._record("healthy", now, "检测到活跃 worker 心跳。")
            return

        if not summary.stall_detected:
            self._stall_since = None
            return

        if self._stall_since is None:
            self._stall_since = now
            self._record(
                "stall_detected",
                now,
                "队列积压且无活跃 worker 心跳"
                f"（ready={summary.backlog.ready}, unacked={summary.backlog.unacked}）。",
            )
            return

        stalled_seconds = (now - self._stall_since).total_seconds()
        if stalled_seconds < self._settings.celery.queue_stall_seconds:
            return

        if not self._settings.celery.worker_auto_recover:
            self._stall_since = now
            self._record(
                "auto_recover_disabled",
                now,
                "worker_auto_recover=false，仅上报不自愈。",
            )
            return

        if self.state.spawn_attempts >= _MAX_SPAWN_ATTEMPTS:
            self._stall_since = now
            self._record(
                "spawn_limit_reached",
                now,
                f"已连续尝试拉起 {_MAX_SPAWN_ATTEMPTS} 次 worker 仍未恢复，停止自愈。",
            )
            return

        self._stall_since = now
        self.state.spawn_attempts += 1
        try:
            pid = self._spawn()
        except Exception as exc:  # noqa: BLE001 - spawn failure must be reported
            self._record("spawn_failed", now, f"拉起 worker 失败：{exc}")
            return

        self.state.recoveries += 1
        self.state.spawned_pids.append(pid)
        logger.warning(
            "worker watchdog: no live consumer for %.0fs, spawned worker pid=%s",
            stalled_seconds,
            pid,
        )
        self._record("spawned_worker", now, f"已拉起新 worker (pid={pid})。")

    def _record(self, action: str, at: datetime, message: str) -> None:
        self.state.last_action = action
        self.state.last_action_at = at
        self.state.last_message = message


def start_worker_watchdog(settings: Settings | None = None) -> WorkerWatchdog:
    current_settings = settings or get_settings()
    return WorkerWatchdog(current_settings).start()
