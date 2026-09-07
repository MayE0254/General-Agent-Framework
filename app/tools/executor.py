from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.agents.base.agent import BaseAgent
from app.tools.registry import ToolRegistry
from app.tools.schemas import (
    ToolCallRequest,
    ToolContext,
    ToolErrorCategory,
    ToolExecutionStatus,
    ToolResult,
)

if TYPE_CHECKING:
    # Imported lazily to avoid a circular import at module load time:
    # app.tools -> app.services -> app.orchestrator -> app.tools.
    from app.services.runtime_record_service import RuntimeRecordService


class ToolExecutor:
    """Executes tools with registry lookup and agent permission checks."""

    def __init__(
        self,
        registry: ToolRegistry,
        runtime_record_service: RuntimeRecordService | None = None,
        tracer: Any | None = None,
    ) -> None:
        self._registry = registry
        self._runtime_record_service = runtime_record_service
        self._tracer = tracer

    async def execute(
        self,
        *,
        agent: BaseAgent,
        request: ToolCallRequest,
        context: ToolContext,
    ) -> ToolResult:
        if not agent.can_use_tool(request.tool_name):
            result = ToolResult(
                tool_name=request.tool_name,
                status=ToolExecutionStatus.DENIED,
                message=(
                    f"Agent '{agent.metadata.agent_id}' is not allowed to use "
                    f"tool '{request.tool_name}'."
                ),
                errors=["tool_permission_denied"],
                error_category=ToolErrorCategory.POLICY_DENIED,
                retryable=False,
                error_detail="agent permission denied by tool allowlist",
            )
            self._record_tool_call(
                tool_context=context,
                tool_request=request,
                tool_result=result,
            )
            return result

        try:
            tool = self._registry.create(request.tool_name)
        except KeyError as exc:
            result = ToolResult(
                tool_name=request.tool_name,
                status=ToolExecutionStatus.FAILED,
                message=f"Tool '{request.tool_name}' is not registered.",
                errors=["tool_not_registered"],
                error_category=ToolErrorCategory.CONFIGURATION,
                retryable=False,
                error_detail=str(exc),
                metrics={"exception_class": type(exc).__name__},
            )
            self._record_tool_call(
                tool_context=context,
                tool_request=request,
                tool_result=result,
            )
            return result
        span = None
        if self._tracer is not None:
            span = self._tracer.start_tool_span(
                request.tool_name,
                agent.metadata.agent_id,
                request,
            )
        try:
            result = await tool.run(context, request.arguments)
        except TimeoutError as exc:
            result = ToolResult(
                tool_name=request.tool_name,
                status=ToolExecutionStatus.TIMEOUT,
                message=f"Tool '{request.tool_name}' timed out.",
                errors=["tool_timeout"],
                error_category=ToolErrorCategory.TIMEOUT,
                retryable=True,
                error_detail=str(exc),
                metrics={"exception_class": type(exc).__name__},
            )
        except Exception as exc:
            if self._tracer is not None:
                self._tracer.capture_error(
                    exc,
                    tool_name=request.tool_name,
                    agent_id=agent.metadata.agent_id,
                )
            result = ToolResult(
                tool_name=request.tool_name,
                status=ToolExecutionStatus.FAILED,
                message=f"Tool '{request.tool_name}' failed with an internal exception.",
                errors=["tool_execution_exception"],
                error_category=ToolErrorCategory.EXECUTION_EXCEPTION,
                retryable=False,
                error_detail=f"{type(exc).__name__}: {exc}",
                metrics={"exception_class": type(exc).__name__},
            )
        if self._tracer is not None:
            self._tracer.finish_tool_span(span, result)
        self._record_tool_call(
            tool_context=context,
            tool_request=request,
            tool_result=result,
        )
        return result

    def _record_tool_call(
        self,
        *,
        tool_context: ToolContext,
        tool_request: ToolCallRequest,
        tool_result: ToolResult,
    ) -> None:
        if self._runtime_record_service is None:
            return
        self._runtime_record_service.save_tool_call_result(
            tool_context=tool_context,
            tool_request=tool_request,
            tool_result=tool_result,
        )
