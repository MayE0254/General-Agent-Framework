"""Celery broker backlog snapshot (Redis transport).

The admin API previously only asked Celery ``inspect`` questions, which answer
from the worker's event loop. When the only consumer is a wedged process the
queue can hold messages for hours while every inspect-based view stays quiet.
These helpers read the broker keyspace directly so backlog and *unacked age*
are observable without any worker cooperating.

Kombu's Redis transport keeps:

* one list per queue (``<queue>`` and ``<queue>\\x06\\x16<priority>``), and
* a hash ``unacked`` (delivery tags) plus a sorted set ``unacked_index``
  scored with the delivery timestamp, holding messages that were taken but
  never acknowledged.
"""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.core import Settings, get_settings
from app.infra.redis import build_broker_client

UNACKED_HASH_KEY = "unacked"
UNACKED_INDEX_KEY = "unacked_index"

# kombu separates a queue name from its priority band with these bytes
# (``Channel.sep``); priority 0 uses the bare queue name.
_PRIORITY_SEPARATOR = "\x06\x16"


class BrokerQueueState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    ready: int = 0


class BrokerBacklogSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reachable: bool = True
    error: str = ""
    queue_name: str = ""
    ready: int = 0
    unacked: int = 0
    oldest_unacked_age_seconds: float | None = None
    queues: list[BrokerQueueState] = Field(default_factory=list)

    @property
    def has_backlog(self) -> bool:
        return self.ready > 0


def snapshot_broker_backlog(
    settings: Settings | None = None,
    *,
    client: Any | None = None,
) -> BrokerBacklogSnapshot:
    """Read queue depth plus unacked count/age from the broker DB.

    Failures are reported inside the snapshot (``reachable=False``) instead of
    raising, so callers can distinguish "broker unreachable" from "queue empty"
    without extra plumbing.
    """
    current_settings = settings or get_settings()
    queue_name = current_settings.celery.task_default_queue
    owned_client = client is None
    broker_client = client if client is not None else build_broker_client(current_settings)

    try:
        queues = _read_queue_states(broker_client, queue_name)
        unacked = int(broker_client.hlen(UNACKED_HASH_KEY) or 0)
        oldest_unacked_age = _oldest_unacked_age_seconds(broker_client)
    except Exception as exc:
        return BrokerBacklogSnapshot(
            reachable=False,
            error=f"{type(exc).__name__}: {exc}",
            queue_name=queue_name,
        )
    finally:
        if owned_client:
            broker_client.close()

    return BrokerBacklogSnapshot(
        reachable=True,
        queue_name=queue_name,
        ready=sum(item.ready for item in queues),
        unacked=unacked,
        oldest_unacked_age_seconds=oldest_unacked_age,
        queues=queues,
    )


def _read_queue_states(client: Any, queue_name: str) -> list[BrokerQueueState]:
    keys = [queue_name]
    keys.extend(
        client.scan_iter(match=f"{queue_name}{_PRIORITY_SEPARATOR}*", count=100)
    )

    states: list[BrokerQueueState] = []
    for key in keys:
        depth = int(client.llen(key) or 0)
        if depth or key == queue_name:
            states.append(BrokerQueueState(name=key, ready=depth))
    return states


def _oldest_unacked_age_seconds(client: Any) -> float | None:
    """Age of the message that has been unacked the longest, in seconds."""
    entries = client.zrange(UNACKED_INDEX_KEY, 0, 0, withscores=True)
    if not entries:
        return None
    _delivery_tag, delivered_at = entries[0]
    return max(0.0, time.time() - float(delivered_at))
