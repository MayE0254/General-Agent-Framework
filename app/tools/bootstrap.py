from __future__ import annotations

from typing import TYPE_CHECKING

from app.tools.registry import ToolRegistry
from app.tools.system import (
    DocumentParserTool,
    HttpRequestTool,
    KnowledgeSearchTool,
    MilvusSearchTool,
    PgVectorSearchTool,
)

if TYPE_CHECKING:
    from app.core.settings import KnowledgeSettings, ToolSettings


def create_default_tool_registry(
    tool_settings: ToolSettings | None = None,
    knowledge_settings: KnowledgeSettings | None = None,
) -> ToolRegistry:
    """Register the framework's default system tools.

    ``tool_settings`` carries the ``[tools]`` configuration block. When
    provided, the ``[tools.http]`` sub-block is bound as the config for the
    ``http_request`` tool so its execution policy (timeout, allowed methods,
    SSRF blocklist, response cap) comes from configuration instead of
    hard-coded defaults.

    ``knowledge_settings`` carries the ``[knowledge]`` block; its documents
    and scoring options are bound as the config for ``knowledge_search``.
    """
    registry = ToolRegistry()
    configs: dict[str, dict] = {}
    if tool_settings is not None:
        configs["http_request"] = tool_settings.http.model_dump()
    if knowledge_settings is not None:
        configs["knowledge_search"] = knowledge_settings.model_dump(
            exclude={"pgvector", "milvus"}
        )
    registry.register_many(
        [
            KnowledgeSearchTool,
            PgVectorSearchTool,
            MilvusSearchTool,
            HttpRequestTool,
            DocumentParserTool,
        ],
        configs=configs,
    )
    return registry
