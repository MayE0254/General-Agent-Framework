"""Unit tests for worker supervision: heartbeats, backlog, watchdog, dedupe.

These cover the guardrails added after a leftover worker took a beat message,
hung inside a task, and left it unacked with no live consumer to notice.
"""

from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace
import time
import weakref

import pytest

from app.core import Settings
from app.infra.celery_broker import (
    BrokerBacklogSnapshot,
    BrokerQueueState,
    snapshot_broker_backlog,
)
from app.infra.worker_health import (
    WorkerHeartbeat,
    heartbeat_key,
    read_worker_heartbeats,
)
from app.services.worker_supervisor import (
    WorkerHealthSummary,
    WorkerWatchdog,
    is_stalled,
    utc_now,
)
from app.tasks.celery_app import build_celery_app
from app.tasks import workflow_tasks


# --- celery wiring ----------------------------------------------------------


def test_broker_transport_options_bound_the_unacked_window() -> None:
    """A stuck message must be re-delivered in minutes, not kombu's 1h default."""
    settings = Settings()
    settings.celery.visibility_timeout_seconds = 600

    celery_app = build_celery_app(settings)

    assert celery_app.conf.broker_transport_options == {"visibility_timeout": 600}
    assert celery_app.conf.worker_prefetch_multiplier == 1
    assert celery_app.conf.task_acks_late is True
    assert celery_app.conf.task_reject_on_worker_lost is True


# --- broker backlog ---------------------------------------------------------


class _FakeBroker:
    """Minimal stand-in for the broker Redis client."""

    def __init__(
        self,
        *,
        depths: dict[str, int] | None = None,
        priority_keys: list[str] | None = None,
        unacked: int = 0,
        oldest_unacked_age: float | None = None,
    ) -> None:
        self.depths = depths or {}
        self.priority_keys = priority_keys or []
        self.unacked = unacked
        self.oldest_unacked_age = oldest_unacked_age
        self.closed = False

    def llen(self, key: str) -> int:
        return self.depths.get(key, 0)

    def scan_iter(self, match: str | None = None, count: int | None = None):
        return iter(self.priority_keys)

    def hlen(self, key: str) -> int:
        return self.unacked

    def zrange(self, key: str, start: int, end: int, withscores: bool = False):
        if self.oldest_unacked_age is None:
            return []
        return [("delivery-tag", time.time() - self.oldest_unacked_age)]

    def close(self) -> None:
        self.closed = True


class _BrokenBroker(_FakeBroker):
    def scan_iter(self, match: str | None = None, count: int | None = None):
        raise RuntimeError("broker down")


def test_snapshot_reports_queue_depth_and_unacked_age() -> None:
    settings = Settings()
    settings.celery.task_default_queue = "multiagent"

    snapshot = snapshot_broker_backlog(
        settings,
        client=_FakeBroker(
            depths={"multiagent": 3},
            unacked=2,
            oldest_unacked_age=1200.0,
        ),
    )

    assert snapshot.reachable is True
    assert snapshot.queue_name == "multiagent"
    assert snapshot.ready == 3
    assert snapshot.unacked == 2
    assert snapshot.oldest_unacked_age_seconds == pytest.approx(1200.0, abs=5)
    assert snapshot.has_backlog is True


def test_snapshot_counts_priority_queue_keys() -> None:
    settings = Settings()
    settings.celery.task_default_queue = "multiagent"
    priority_key = "multiagent\x06\x162"

    snapshot = snapshot_broker_backlog(
        settings,
        client=_FakeBroker(
            depths={"multiagent": 1, priority_key: 4},
            priority_keys=[priority_key],
        ),
    )

    assert snapshot.ready == 5
    assert {item.name for item in snapshot.queues} == {"multiagent", priority_key}


def test_snapshot_reports_broker_failure_without_raising() -> None:
    snapshot = snapshot_broker_backlog(Settings(), client=_BrokenBroker())

    assert snapshot.reachable is False
    assert "broker down" in snapshot.error
    # An unreadable broker must never be mistaken for an empty queue.
    assert snapshot.has_backlog is False
    assert is_stalled(snapshot) is False


# --- stall detection --------------------------------------------------------


def _backlog(
    *,
    ready: int = 0,
    unacked: int = 0,
    oldest_age: float | None = None,
    reachable: bool = True,
) -> BrokerBacklogSnapshot:
    return BrokerBacklogSnapshot(
        reachable=reachable,
        queue_name="multiagent",
        ready=ready,
        unacked=unacked,
        oldest_unacked_age_seconds=oldest_age,
        queues=[BrokerQueueState(name="multiagent", ready=ready)],
    )


def test_is_stalled_detects_queued_and_abandoned_messages() -> None:
    settings = Settings()
    settings.celery.visibility_timeout_seconds = 900

    assert is_stalled(_backlog(ready=1), settings) is True

    # Taken by a worker that may still be working on it: not yet stalled.
    assert is_stalled(_backlog(unacked=1, oldest_age=30.0), settings) is False
    # Past the visibility timeout only a live restore sweep can save it.
    assert is_stalled(_backlog(unacked=1, oldest_age=1500.0), settings) is True

    assert is_stalled(_backlog(), settings) is False
    assert is_stalled(_backlog(unacked=1, oldest_age=None), settings) is False


# --- heartbeats -------------------------------------------------------------


def test_worker_ready_keeps_a_live_receiver() -> None:
    """worker_ready must keep a live receiver, or no heartbeat ever gets written.

    Celery connects receivers weakly by default. Both lifecycle handlers are
    nested functions, so with the default they were collected the moment
    ``_register_worker_lifecycle_signals`` returned: the signal fired into a
    dead weakref, the heartbeat thread never started, and both the launcher's
    self-check and the API watchdog reported "no live worker" while the worker
    was in fact healthy.
    """
    from celery.signals import worker_ready

    build_celery_app(Settings())

    alive = [
        receiver
        for _, receiver in worker_ready.receivers
        if not isinstance(receiver, weakref.ReferenceType) or receiver() is not None
    ]

    assert alive, "worker_ready has no live receiver: heartbeats would never start"


class _FakeHeartbeatRedis:
    def __init__(self, payloads: list[str | None]) -> None:
        self.payloads = payloads
        self.closed = False

    def scan_iter(self, match: str | None = None, count: int | None = None):
        return iter([f"{heartbeat_key('w:1')}", f"{heartbeat_key('w:2')}"])

    def mget(self, keys: list[str]):
        return self.payloads

    def close(self) -> None:
        self.closed = True


def test_read_worker_heartbeats_skips_malformed_entries() -> None:
    payload = WorkerHeartbeat(worker_id="w:1", pid=10, pool="solo").model_dump_json()

    heartbeats = read_worker_heartbeats(
        Settings(),
        client=_FakeHeartbeatRedis([payload, "not-json"]),
    )

    assert [item.worker_id for item in heartbeats] == ["w:1"]
    assert heartbeats[0].pid == 10


def test_heartbeat_staleness_uses_beat_age() -> None:
    stale = WorkerHeartbeat(
        worker_id="w:1",
        beat_at=utc_now() - timedelta(seconds=400),
    )
    fresh = WorkerHeartbeat(worker_id="w:2")

    assert stale.is_stale(300) is True
    assert stale.age_seconds() >= 400
    assert fresh.is_stale(300) is False


# --- watchdog ---------------------------------------------------------------


def _summary(*, stall: bool, live: list[str] | None = None) -> WorkerHealthSummary:
    return WorkerHealthSummary(
        stall_detected=stall,
        live_worker_ids=live or [],
        backlog=_backlog(ready=1 if stall else 0),
    )


def test_watchdog_spawns_only_after_the_stall_window() -> None:
    settings = Settings()
    settings.celery.queue_stall_seconds = 300
    settings.celery.worker_auto_recover = True
    spawned: list[int] = []
    watchdog = WorkerWatchdog(settings, spawn=lambda: (spawned.append(1), 4242)[1])

    watchdog._react(_summary(stall=True))
    assert spawned == []
    assert watchdog.state.last_action == "stall_detected"

    watchdog._stall_since = utc_now() - timedelta(
        seconds=settings.celery.queue_stall_seconds + 1
    )
    watchdog._react(_summary(stall=True))

    assert spawned == [1]
    assert watchdog.state.recoveries == 1
    assert watchdog.state.spawned_pids == [4242]
    assert watchdog.state.last_action == "spawned_worker"


def test_watchdog_stops_spawning_after_the_attempt_limit() -> None:
    settings = Settings()
    settings.celery.queue_stall_seconds = 300
    spawned: list[int] = []
    watchdog = WorkerWatchdog(settings, spawn=lambda: (spawned.append(1), 1)[1])

    for _ in range(5):
        watchdog._stall_since = utc_now() - timedelta(
            seconds=settings.celery.queue_stall_seconds + 1
        )
        watchdog._react(_summary(stall=True))

    assert len(spawned) == 3
    assert watchdog.state.spawn_attempts == 3
    assert watchdog.state.last_action == "spawn_limit_reached"


def test_watchdog_does_not_spawn_when_auto_recover_is_disabled() -> None:
    settings = Settings()
    settings.celery.queue_stall_seconds = 300
    settings.celery.worker_auto_recover = False
    spawned: list[int] = []
    watchdog = WorkerWatchdog(settings, spawn=lambda: (spawned.append(1), 1)[1])

    watchdog._stall_since = utc_now() - timedelta(
        seconds=settings.celery.queue_stall_seconds + 1
    )
    watchdog._react(_summary(stall=True))

    assert spawned == []
    assert watchdog.state.last_action == "auto_recover_disabled"


def test_watchdog_reports_healthy_and_resets_attempts() -> None:
    settings = Settings()
    watchdog = WorkerWatchdog(settings, spawn=lambda: 1)
    watchdog.state.spawn_attempts = 3

    watchdog._react(_summary(stall=False, live=["host:1"]))

    assert watchdog.state.spawn_attempts == 0
    assert watchdog.state.last_action == "healthy"
    assert watchdog._stall_since is None


def test_watchdog_reports_spawn_failures() -> None:
    settings = Settings()
    settings.celery.queue_stall_seconds = 300

    def _boom() -> int:
        raise OSError("no such executable")

    watchdog = WorkerWatchdog(settings, spawn=_boom)
    watchdog._stall_since = utc_now() - timedelta(
        seconds=settings.celery.queue_stall_seconds + 1
    )
    watchdog._react(_summary(stall=True))

    assert watchdog.state.last_action == "spawn_failed"
    assert "no such executable" in watchdog.state.last_message
    assert watchdog.state.recoveries == 0


# --- duplicate delivery guard ----------------------------------------------


def test_duplicate_request_id_is_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("sqlalchemy")
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session, sessionmaker

    from app.models import Base, WorkflowRunRecord

    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)

    with session_factory() as session:
        session.add(
            WorkflowRunRecord(
                id="run-1",
                request_id="content-ops-20260916-001",
                initial_agent_id="content_ops_agent",
                status="completed",
                execution_path=["content_ops_agent"],
                errors=[],
                summary="done",
                metadata_payload={"terminal_reason": "natural_completion"},
            )
        )
        session.commit()

    stack = SimpleNamespace(
        resources=SimpleNamespace(database_session_factory=session_factory)
    )
    monkeypatch.setattr(workflow_tasks, "_get_runtime", lambda: stack)

    found = workflow_tasks._find_finished_run("content-ops-20260916-001")

    assert found is not None
    assert found["duplicate"] is True
    assert found["status"] == "completed"
    assert found["terminal_reason"] == "natural_completion"
    assert workflow_tasks._find_finished_run("content-ops-20260916-999") is None


def test_finished_run_lookup_without_database_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        workflow_tasks,
        "_get_runtime",
        lambda: SimpleNamespace(resources=SimpleNamespace(database_session_factory=None)),
    )

    assert workflow_tasks._find_finished_run("req-preview-mode") is None
