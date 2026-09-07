from __future__ import annotations

from collections.abc import Callable

from app.agents.base.agent import BaseAgent
from app.agents.registry.agent_registry import AgentRegistry


def register_agent(
    registry: AgentRegistry,
) -> Callable[[type[BaseAgent]], type[BaseAgent]]:
    """Decorator used to keep registration close to agent declarations."""

    def decorator(agent_class: type[BaseAgent]) -> type[BaseAgent]:
        registry.register(agent_class)
        return agent_class

    return decorator
