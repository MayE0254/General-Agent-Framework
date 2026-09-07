"""Business agent layer.

Each subpackage under ``app.business`` adapts one vertical business agent
(kept in its own repository) into the framework's registries: tools wrap the
business pipelines, a DOMAIN agent dispatches intents, and optional Celery
tasks expose the business' scheduled jobs on the framework's beat schedule.

Business packages are optional plugins. The framework discovers them here so
that a checkout without any business subpackage (e.g. the open-source
infrastructure repo) stays a pure, runnable infrastructure build:

- a subpackage that defines ``register_business(**registries)`` mounts its
  agents/tools into the runtime registries (see ``register_business_agents``);
- a subpackage with a ``tasks`` module contributes Celery task modules (see
  ``business_task_modules``) and, if that module defines ``beat_entries()``,
  entries for the framework beat schedule (see ``business_beat_entries``);
- any import failure inside a business package is logged and skipped — it
  never breaks the infrastructure process.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from typing import Any

logger = logging.getLogger(__name__)


def iter_business_packages() -> list[Any]:
    """Import every business subpackage, skipping the ones that fail."""
    packages: list[Any] = []
    for module_info in pkgutil.iter_modules(__path__):
        try:
            module = importlib.import_module(f"{__name__}.{module_info.name}")
        except Exception as exc:  # noqa: BLE001 - plugin isolation
            logger.warning(
                "Skipping business package %s: %s", module_info.name, exc
            )
            continue
        packages.append(module)
    return packages


def register_business_agents(
    *,
    agent_registry: Any,
    tool_registry: Any,
) -> list[str]:
    """Mount agents/tools from every business package exposing ``register_business``."""
    mounted: list[str] = []
    for module in iter_business_packages():
        register = getattr(module, "register_business", None)
        if register is None:
            continue
        register(agent_registry=agent_registry, tool_registry=tool_registry)
        mounted.append(module.__name__.rsplit(".", 1)[-1])
    return mounted


def business_task_modules() -> list[str]:
    """Return importable ``<business>.tasks`` module names for Celery include."""
    modules: list[str] = []
    for package in iter_business_packages():
        tasks_ref = f"{package.__name__}.tasks"
        try:
            importlib.import_module(tasks_ref)
        except Exception as exc:  # noqa: BLE001 - plugin isolation
            logger.warning(
                "Skipping business tasks module %s: %s", tasks_ref, exc
            )
            continue
        modules.append(tasks_ref)
    return modules


def business_beat_entries() -> dict[str, dict[str, Any]]:
    """Collect ``beat_entries()`` dicts from business ``tasks`` modules."""
    entries: dict[str, dict[str, Any]] = {}
    for package in iter_business_packages():
        tasks_ref = f"{package.__name__}.tasks"
        try:
            tasks_module = importlib.import_module(tasks_ref)
        except Exception as exc:  # noqa: BLE001 - plugin isolation
            logger.warning(
                "Skipping business tasks module %s: %s", tasks_ref, exc
            )
            continue
        provider = getattr(tasks_module, "beat_entries", None)
        if provider is None:
            continue
        entries.update(provider())
    return entries
