"""Routable-agent directory used by the RouterAgent.

``RouterAgent`` has no access to the ``AgentRegistry`` instance, so DOMAIN
agents cannot be discovered dynamically from inside its LLM prompt. This
module closes that gap: ``build_runtime_stack`` syncs DOMAIN agents from the
registry into this directory once per process, and the router treats every
registered DOMAIN agent as a routing candidate — no per-domain wiring needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.agents.registry.agent_registry import AgentRegistry

_DOMAIN_DESCRIPTIONS: dict[str, str] = {}


def sync_domain_routes(registry: AgentRegistry) -> list[str]:
    """Sync routable DOMAIN agents from the registry into this directory."""
    descriptions: dict[str, str] = {}
    for metadata in registry.list_metadata(agent_kind="domain"):
        descriptions[metadata.agent_id] = (
            metadata.description or metadata.agent_name
        )
    _DOMAIN_DESCRIPTIONS.clear()
    _DOMAIN_DESCRIPTIONS.update(descriptions)
    return sorted(_DOMAIN_DESCRIPTIONS)


def domain_route_entries() -> dict[str, str]:
    """Snapshot of routable DOMAIN agents (agent_id -> description)."""
    return dict(_DOMAIN_DESCRIPTIONS)
