from __future__ import annotations

from abc import ABC, abstractmethod

from app.agents.base.schemas import AgentContext, AgentMetadata, AgentResult


class BaseAgent(ABC):
    """Base protocol for all agents in the framework."""

    def __init__(self, metadata: AgentMetadata | None = None) -> None:
        self._metadata = metadata or self.build_metadata()

    @classmethod
    @abstractmethod
    def build_metadata(cls) -> AgentMetadata:
        """Return immutable metadata used by the registry and orchestrator."""

    @property
    def metadata(self) -> AgentMetadata:
        return self._metadata

    async def run(self, context: AgentContext) -> AgentResult:
        """Entry point used by the orchestrator."""
        result = await self._run(context)
        if result.agent_id != self.metadata.agent_id:
            raise ValueError(
                "Agent result agent_id does not match metadata: "
                f"{result.agent_id} != {self.metadata.agent_id}"
            )
        return result

    def can_use_tool(self, tool_name: str) -> bool:
        return tool_name in self.metadata.allowed_tools

    @abstractmethod
    async def _run(self, context: AgentContext) -> AgentResult:
        """Implement agent-specific logic."""
