from __future__ import annotations

import logging
import time
from urllib.parse import urlparse

import httpx

from app.knowledge import InMemoryKnowledgeRetriever
from app.tools.base import BaseTool
from app.tools.schemas import (
    ToolContext,
    ToolErrorCategory,
    ToolExecutionStatus,
    ToolMetadata,
    ToolResult,
)

logger = logging.getLogger(__name__)


class KnowledgeSearchTool(BaseTool):
    """Knowledge retrieval tool backed by the pluggable retriever layer.

    Documents come from the ``[knowledge]`` settings block via the tool
    registry config; with no documents configured the tool still answers
    with an empty hit list instead of fake hits.
    """

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config=config)
        self._retriever: InMemoryKnowledgeRetriever | None = None

    @classmethod
    def build_metadata(cls) -> ToolMetadata:
        return ToolMetadata(
            tool_name="knowledge_search",
            description="Searches configured knowledge documents for a query.",
            tags=["core", "knowledge"],
        )

    def _get_retriever(self) -> InMemoryKnowledgeRetriever:
        if self._retriever is None:
            self._retriever = InMemoryKnowledgeRetriever(
                self._config.get("documents") or [],
                min_score=float(self._config.get("min_score", 0.05)),
            )
        return self._retriever

    async def _run(self, context: ToolContext, arguments: dict) -> ToolResult:
        query = arguments.get("query", "")
        retriever = self._get_retriever()
        settings_top_k = int(self._config.get("top_k", 5))
        try:
            top_k = int(arguments.get("top_k", settings_top_k))
        except (TypeError, ValueError):
            top_k = settings_top_k
        hits = await retriever.retrieve(query, top_k=top_k)
        return ToolResult(
            tool_name=self.metadata.tool_name,
            status=ToolExecutionStatus.SUCCESS,
            content={
                "query": query,
                "hits": len(hits),
                "backend": retriever.name,
                "results": [hit.model_dump() for hit in hits],
            },
            message=f"knowledge search executed ({len(hits)} hits)",
        )


class PgVectorSearchTool(BaseTool):
    @classmethod
    def build_metadata(cls) -> ToolMetadata:
        return ToolMetadata(
            tool_name="pgvector_search",
            description="Placeholder pgvector retrieval tool.",
            tags=["core", "knowledge", "pgvector"],
        )

    async def _run(self, context: ToolContext, arguments: dict) -> ToolResult:
        return ToolResult(
            tool_name=self.metadata.tool_name,
            status=ToolExecutionStatus.SUCCESS,
            content={
                "query": arguments.get("query", ""),
                "backend": "pgvector",
                "matches": [],
            },
            message="pgvector placeholder executed",
        )


class MilvusSearchTool(BaseTool):
    @classmethod
    def build_metadata(cls) -> ToolMetadata:
        return ToolMetadata(
            tool_name="milvus_search",
            description="Placeholder Milvus retrieval tool.",
            tags=["core", "knowledge", "milvus"],
        )

    async def _run(self, context: ToolContext, arguments: dict) -> ToolResult:
        return ToolResult(
            tool_name=self.metadata.tool_name,
            status=ToolExecutionStatus.SUCCESS,
            content={
                "query": arguments.get("query", ""),
                "backend": "milvus",
                "matches": [],
            },
            message="milvus placeholder executed",
        )


class HttpRequestTool(BaseTool):
    """Real outbound HTTP tool for executor agents.

    Execution policy is resolved from the registered config (the
    ``[tools.http]`` block): timeout, allowed methods, response body size
    cap and an SSRF host blocklist. Arguments accepted by ``_run``:

    ``url`` (required), ``method`` (default GET), ``headers`` (dict),
    ``params`` (dict), ``json_body`` (dict), ``data`` (str),
    ``timeout_seconds`` (per-call override).
    """

    @classmethod
    def build_metadata(cls) -> ToolMetadata:
        return ToolMetadata(
            tool_name="http_request",
            description=(
                "Performs a real outbound HTTP request. Required argument "
                "'url'; optional 'method', 'headers', 'params', 'json_body', "
                "'data', 'timeout_seconds'."
            ),
            tags=["core", "integration"],
        )

    # Cloud metadata endpoints that must never be reachable (SSRF guard).
    # Exact host matches are refused before any connection is attempted.
    _ALWAYS_BLOCKED_HOSTS = {
        "169.254.169.254",  # AWS / GCP / Aliyun metadata
        "metadata.google.internal",
        "metadata.azure.internal",
    }

    async def _run(self, context: ToolContext, arguments: dict) -> ToolResult:
        url = arguments.get("url")
        if not isinstance(url, str) or not url.strip():
            return ToolResult(
                tool_name=self.metadata.tool_name,
                status=ToolExecutionStatus.FAILED,
                message="http_request requires a non-empty 'url' argument.",
                errors=["missing_url"],
                error_category=ToolErrorCategory.INPUT_VALIDATION,
                retryable=False,
                error_detail="missing required argument: url",
            )

        method = str(arguments.get("method", "GET")).upper()
        allowed_methods = [
            m.upper()
            for m in self._config_get(
                "allowed_methods",
                ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"],
            )
        ]
        if method not in allowed_methods:
            return ToolResult(
                tool_name=self.metadata.tool_name,
                status=ToolExecutionStatus.FAILED,
                message=(
                    f"HTTP method '{method}' is not allowed. Allowed: "
                    f"{', '.join(allowed_methods)}."
                ),
                errors=["method_not_allowed"],
                error_category=ToolErrorCategory.POLICY_DENIED,
                retryable=False,
                error_detail=f"method '{method}' is not in the configured allowlist",
            )

        blocked = self._resolve_blocked_hosts()
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        if not host or host in blocked:
            return ToolResult(
                tool_name=self.metadata.tool_name,
                status=ToolExecutionStatus.FAILED,
                message=f"HTTP host '{host}' is blocked by policy.",
                errors=["host_blocked"],
                error_category=ToolErrorCategory.POLICY_DENIED,
                retryable=False,
                error_detail=f"blocked host '{host}'",
            )

        timeout_seconds = float(
            arguments.get(
                "timeout_seconds",
                self._config_get("timeout_seconds", self.metadata.timeout_seconds),
            )
        )
        max_response_bytes = int(
            self._config_get("max_response_bytes", 1_048_576)
        )

        headers = dict(arguments.get("headers") or {})
        headers.setdefault(
            "User-Agent", self._config_get("user_agent", "multiagent-agent/0.1")
        )

        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(
                timeout=timeout_seconds,
                follow_redirects=bool(
                    self._config_get("follow_redirects", True)
                ),
            ) as client:
                request_kwargs: dict = {
                    "headers": headers,
                    "params": arguments.get("params") or None,
                }
                json_body = arguments.get("json_body")
                data = arguments.get("data")
                if json_body is not None:
                    request_kwargs["json"] = json_body
                elif data is not None:
                    request_kwargs["content"] = data
                response = await client.request(
                    method, url, **request_kwargs
                )
        except httpx.TimeoutException as exc:
            return ToolResult(
                tool_name=self.metadata.tool_name,
                status=ToolExecutionStatus.TIMEOUT,
                message=f"HTTP request to '{url}' timed out after {timeout_seconds}s.",
                errors=["http_timeout"],
                error_category=ToolErrorCategory.TIMEOUT,
                retryable=True,
                error_detail=str(exc),
                metrics={"latency_ms": round((time.perf_counter() - started) * 1000, 2)},
            )
        except httpx.HTTPError as exc:
            logger.warning(
                "http_request failed for %s %s: %s", method, url, exc
            )
            return ToolResult(
                tool_name=self.metadata.tool_name,
                status=ToolExecutionStatus.FAILED,
                message=f"HTTP request to '{url}' failed: {exc}",
                errors=["http_error"],
                error_category=ToolErrorCategory.EXTERNAL_DEPENDENCY,
                retryable=True,
                error_detail=f"{type(exc).__name__}: {exc}",
                metrics={"latency_ms": round((time.perf_counter() - started) * 1000, 2)},
            )

        body = response.content
        truncated = len(body) > max_response_bytes
        if truncated:
            body = body[:max_response_bytes]
        text_body = body.decode("utf-8", errors="replace")

        return ToolResult(
            tool_name=self.metadata.tool_name,
            status=ToolExecutionStatus.SUCCESS,
            content={
                "url": url,
                "method": method,
                "status_code": response.status_code,
                "headers": {
                    key: value
                    for key, value in response.headers.items()
                    if key.lower() in {"content-type", "location"}
                },
                "body": text_body,
                "truncated": truncated,
            },
            message=(
                f"HTTP {method} {url} -> {response.status_code} "
                f"({len(response.content)} bytes)"
            ),
            metrics={
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "status_code": response.status_code,
                "response_bytes": len(response.content),
            },
        )

    def _resolve_blocked_hosts(self) -> set[str]:
        blocked = set(self._ALWAYS_BLOCKED_HOSTS)
        configured = self._config_get("blocked_hosts", [])
        if isinstance(configured, list):
            blocked.update(
                str(item).strip().lower() for item in configured if str(item).strip()
            )
        return blocked


class DocumentParserTool(BaseTool):
    @classmethod
    def build_metadata(cls) -> ToolMetadata:
        return ToolMetadata(
            tool_name="document_parser",
            description="Placeholder document parsing tool for ingestion workflows.",
            tags=["core", "document"],
        )

    async def _run(self, context: ToolContext, arguments: dict) -> ToolResult:
        return ToolResult(
            tool_name=self.metadata.tool_name,
            status=ToolExecutionStatus.SUCCESS,
            content={
                "document_name": arguments.get("document_name"),
                "pages": arguments.get("pages", 0),
            },
            message="document parser placeholder executed",
        )
