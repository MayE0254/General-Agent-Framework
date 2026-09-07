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

_BASE_TASK_MODULES = ["app.tasks.workflow_tasks"]


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
        task_default_queue=urls["task_default_queue"],
        worker_pool=urls["worker_pool"],
        include=task_modules,
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        task_track_started=True,
        task_ignore_result=False,
        broker_connection_retry_on_startup=True,
        beat_schedule=business_beat_entries(),
        timezone="Asia/Shanghai",
    )
    return celery_app


celery_app = build_celery_app()
