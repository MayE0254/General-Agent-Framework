from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.tools.schemas import ToolContext, ToolMetadata, ToolResult


class BaseTool(ABC):
    """Base protocol for all tools exposed to agents.

    ``config`` carries tool-specific settings resolved from the application
    configuration (e.g. the ``[tools.http]`` block) or from a custom
    registration entry, so a single tool class can be reused with different
    endpoints/credentials without subclassing.
    """

    def __init__(
        self,
        metadata: ToolMetadata | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._metadata = metadata or self.build_metadata()
        self._config = config or {}

    @classmethod
    @abstractmethod
    def build_metadata(cls) -> ToolMetadata:
        """Return immutable metadata used by the tool registry."""

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    @property
    def config(self) -> dict[str, Any]:
        """Per-instance configuration resolved at registration time."""
        return self._config

    def _config_get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

    async def run(self, context: ToolContext, arguments: dict) -> ToolResult:
        result = await self._run(context, arguments)
        if result.tool_name != self.metadata.tool_name:
            raise ValueError(
                "Tool result tool_name does not match metadata: "
                f"{result.tool_name} != {self.metadata.tool_name}"
            )
        return result

    @abstractmethod
    async def _run(self, context: ToolContext, arguments: dict) -> ToolResult:
        """Implement tool-specific execution logic."""
