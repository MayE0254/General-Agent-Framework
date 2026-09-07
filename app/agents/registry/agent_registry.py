from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar

from app.agents.base.agent import BaseAgent
from app.agents.base.schemas import AgentKind, AgentMetadata

AgentType = TypeVar("AgentType", bound=BaseAgent)


class AgentRegistry:
    """Central registry for agent classes and metadata."""

    def __init__(self) -> None:
        self._agent_classes: dict[str, type[BaseAgent]] = {}
        self._agent_metadata: dict[str, AgentMetadata] = {}

    def register(self, agent_class: type[AgentType]) -> type[AgentType]:
        metadata = agent_class.build_metadata()
        existing = self._agent_metadata.get(metadata.agent_id)
        if existing is not None:
            raise ValueError(f"Agent '{metadata.agent_id}' is already registered.")

        self._agent_classes[metadata.agent_id] = agent_class
        self._agent_metadata[metadata.agent_id] = metadata
        return agent_class

    def register_many(
        self, agent_classes: Iterable[type[AgentType]]
    ) -> list[type[AgentType]]:
        return [self.register(agent_class) for agent_class in agent_classes]

    def exists(self, agent_id: str) -> bool:
        return agent_id in self._agent_classes

    def get_metadata(self, agent_id: str) -> AgentMetadata:
        try:
            return self._agent_metadata[agent_id]
        except KeyError as exc:
            raise KeyError(f"Agent '{agent_id}' is not registered.") from exc

    def create(self, agent_id: str) -> BaseAgent:
        try:
            agent_class = self._agent_classes[agent_id]
        except KeyError as exc:
            raise KeyError(f"Agent '{agent_id}' is not registered.") from exc
        return agent_class()

    def list_metadata(
        self,
        *,
        agent_kind: AgentKind | None = None,
        tag: str | None = None,
    ) -> list[AgentMetadata]:
        items = list(self._agent_metadata.values())
        if agent_kind is not None:
            items = [item for item in items if item.agent_kind == agent_kind]
        if tag is not None:
            items = [item for item in items if tag in item.tags]
        return sorted(items, key=lambda item: item.agent_id)

    def registered_agent_ids(self) -> list[str]:
        return sorted(self._agent_classes.keys())
