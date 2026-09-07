from __future__ import annotations

from app.agents.registry.agent_registry import AgentRegistry
from app.agents.system.builtin_agents import (
    ExecutorAgent,
    KnowledgeAgent,
    PlannerAgent,
    ReviewerAgent,
    RouterAgent,
)


def create_default_registry() -> AgentRegistry:
    """Register the framework's default system agents."""
    registry = AgentRegistry()
    registry.register_many(
        [
            PlannerAgent,
            RouterAgent,
            KnowledgeAgent,
            ExecutorAgent,
            ReviewerAgent,
        ]
    )
    return registry
