from __future__ import annotations

from collections.abc import Iterable
from typing import Any, TypeVar

from app.tools.base import BaseTool
from app.tools.schemas import ToolMetadata

ToolType = TypeVar("ToolType", bound=BaseTool)


class ToolRegistry:
    """Central registry for tool classes, per-tool config and metadata."""

    def __init__(self) -> None:
        self._tool_classes: dict[str, type[BaseTool]] = {}
        self._tool_metadata: dict[str, ToolMetadata] = {}
        self._tool_configs: dict[str, dict[str, Any]] = {}

    def register(
        self,
        tool_class: type[ToolType],
        config: dict[str, Any] | None = None,
    ) -> type[ToolType]:
        metadata = tool_class.build_metadata()
        existing = self._tool_metadata.get(metadata.tool_name)
        if existing is not None:
            raise ValueError(f"Tool '{metadata.tool_name}' is already registered.")

        self._tool_classes[metadata.tool_name] = tool_class
        self._tool_metadata[metadata.tool_name] = metadata
        self._tool_configs[metadata.tool_name] = config or {}
        return tool_class

    def register_many(
        self,
        tool_classes: Iterable[type[ToolType]],
        configs: dict[str, dict[str, Any]] | None = None,
    ) -> list[type[ToolType]]:
        registered = []
        for tool_class in tool_classes:
            registered.append(
                self.register(tool_class, (configs or {}).get(tool_class.build_metadata().tool_name))
            )
        return registered

    def exists(self, tool_name: str) -> bool:
        return tool_name in self._tool_classes

    def get_metadata(self, tool_name: str) -> ToolMetadata:
        try:
            return self._tool_metadata[tool_name]
        except KeyError as exc:
            raise KeyError(f"Tool '{tool_name}' is not registered.") from exc

    def get_config(self, tool_name: str) -> dict[str, Any]:
        try:
            return self._tool_configs[tool_name]
        except KeyError as exc:
            raise KeyError(f"Tool '{tool_name}' is not registered.") from exc

    def create(self, tool_name: str) -> BaseTool:
        try:
            tool_class = self._tool_classes[tool_name]
        except KeyError as exc:
            raise KeyError(f"Tool '{tool_name}' is not registered.") from exc
        return tool_class(config=self._tool_configs[tool_name])

    def registered_tool_names(self) -> list[str]:
        return sorted(self._tool_classes.keys())
