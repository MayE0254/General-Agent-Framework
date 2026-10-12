"""Celery application wiring.

The broker/result URLs come from the ``[redis]`` + ``[celery]`` config blocks
(``get_celery_redis_urls``), so the async layer never hard-codes connection
strings. API and worker processes share the same module-level ``celery_app``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from celery import Celery
from celery.schedules import crontab

from app.business import business_beat_entries, business_task_modules
from app.infra import get_celery_redis_urls

if TYPE_CHECKING:
    from app.core import Settings

_BASE_TASK_MODULES = ["app.tasks.workflow_tasks", "app.tasks.review_tasks"]

# Guards against registering the lifecycle signal handlers twice when
# build_celery_app() is called more than once in the same process.
_SIGNALS_REGISTERED = False


def _register_worker_lifecycle_signals() -> None:
    """Start/stop the out-of-band heartbeat when this process runs as a worker.

    The heartbeat lives in its own thread (see app.infra.worker_health) so a
    worker wedged inside a task stops reporting itself as alive. The API
    process imports the same module but the worker-only signals never fire
    there, which keeps a single wiring point for both roles.
    """
    global _SIGNALS_REGISTERED
    if _SIGNALS_REGISTERED:
        return
    _SIGNALS_REGISTERED = True

    from celery.signals import worker_ready, worker_shutting_down

    # weak=False on purpose: both handlers are nested functions, so Celery's
    # default weak registration left a dead weakref behind as soon as this
    # function returned ("receivers: 1, live: 0"). worker_ready then fired into
    # nothing, the heartbeat thread never started, and the launcher self-check
    # plus the API watchdog both read "no live worker" forever while the worker
    # was in fact healthy.
    @worker_ready.connect(weak=False)
    def _on_worker_ready(**_kwargs: object) -> None:
        from app.core import get_settings
        from app.infra.worker_health import start_worker_heartbeat

        start_worker_heartbeat(get_settings())

    @worker_shutting_down.connect(weak=False)
    def _on_worker_shutting_down(**_kwargs: object) -> None:
        from app.infra.worker_health import stop_worker_heartbeat

        stop_worker_heartbeat()


# Framework-level beat entries (in addition to business contributions).
_BASE_BEAT_ENTRIES = {
    # Auto-approve pending human reviews older than the timeout so
    # unattended workflows keep moving (see app.tasks.review_tasks).
    "review-auto-approve-overdue": {
        "task": "review.auto_approve_overdue",
        "schedule": crontab(minute="*/10"),
    },
}


def build_celery_app(settings: Settings | None = None) -> Celery:
    # Lazy import so that importing this module never touches configuration.
    from app.core import get_settings

    current_settings = settings or get_settings()
    urls = get_celery_redis_urls(current_settings)

    # Business packages contribute their task modules and beat entries only
    # when present (optional plugins — see app.business). A bare
    # infrastructure checkout runs the workflow tasks without any schedule.
    task_modules = list(_BASE_TASK_MODULES) + business_task_modules()

    celery_app = Celery("multiagent")
    celery_app.conf.update(
        broker_url=urls["broker_url"],
        result_backend=urls["result_backend"],
        # Redis never *pushes* an unacked message back on its own: only a live
        # worker's periodic restore sweep does (kombu calls
        # maybe_restore_messages every ~10s). With no live consumer the
        # message stays unacked forever, which is how a message got lost once
        # a stale worker grabbed it. Declaring the timeout explicitly (instead
        # of inheriting kombu's 1h default) bounds that window.
        broker_transport_options={
            "visibility_timeout": current_settings.celery.visibility_timeout_seconds,
        },
        task_default_queue=urls["task_default_queue"],
        worker_pool=urls["worker_pool"],
        # Fetch one message at a time: a single long task can no longer hold a
        # whole batch of messages hostage for its entire runtime.
        worker_prefetch_multiplier=1,
        # Ack only after the task returns, and requeue instead of dropping the
        # message when a worker dies mid-task.
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        include=task_modules,
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        task_track_started=True,
        task_ignore_result=False,
        broker_connection_retry_on_startup=True,
        beat_schedule={**_BASE_BEAT_ENTRIES, **business_beat_entries()},
        timezone="Asia/Shanghai",
    )
    _register_worker_lifecycle_signals()
    return celery_app


celery_app = build_celery_app()
