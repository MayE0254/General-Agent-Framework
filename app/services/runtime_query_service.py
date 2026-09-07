from __future__ import annotations

from app.core import NotFoundError
from app.models import AgentRunLog, LLMCallLog, TaskRunRecord, ToolCallLog, WorkflowRunRecord
from app.repositories import RuntimeAuditRepository, WorkflowRunRepository


def _build_runtime_aggregates(
    *,
    task_run_count: int = 0,
    agent_run_count: int = 0,
    tool_call_count: int = 0,
    llm_call_count: int = 0,
    total_llm_tokens: int = 0,
    total_cost_usd: float = 0.0,
) -> dict[str, object]:
    return {
        "task_run_count": task_run_count,
        "agent_run_count": agent_run_count,
        "tool_call_count": tool_call_count,
        "llm_call_count": llm_call_count,
        "total_llm_tokens": total_llm_tokens,
        "total_cost_usd": round(total_cost_usd, 6),
    }


class RuntimeQueryService:
    """Query-side service for platform runtime audit records."""

    def __init__(
        self,
        repository: RuntimeAuditRepository,
        workflow_run_repository: WorkflowRunRepository | None = None,
    ) -> None:
        self._repository = repository
        self._workflow_run_repository = workflow_run_repository

    def list_task_runs(
        self,
        *,
        status: str | None = None,
        session_id: str | None = None,
        workflow_id: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, object]:
        items, total = self._repository.list_task_runs(
            status=status,
            session_id=session_id,
            workflow_id=workflow_id,
            limit=limit,
            offset=offset,
        )
        return {
            "items": items,
            "total": total,
            "limit": limit,
            "offset": offset,
            "filters": {
                "status": status,
                "session_id": session_id,
                "workflow_id": workflow_id,
            },
        }

    def get_task_run_detail(self, request_id: str) -> dict[str, object]:
        task_run = self._repository.get_task_run_by_request_id(request_id)
        if task_run is None:
            raise NotFoundError(
                "Task run was not found.",
                details={"request_id": request_id},
            )
        agent_runs = self._repository.list_agent_runs_by_request_id(request_id)
        tool_calls = self._repository.list_tool_calls_by_request_id(request_id)
        llm_calls = self._repository.list_llm_call_logs_by_request_id(request_id)

        return {
            "task_run": task_run,
            "agent_runs": agent_runs,
            "tool_calls": tool_calls,
            "llm_calls": llm_calls,
            "aggregates": _build_runtime_aggregates(
                task_run_count=1,
                agent_run_count=len(agent_runs),
                tool_call_count=len(tool_calls),
                llm_call_count=len(llm_calls),
                total_llm_tokens=sum(
                    item.total_tokens for item in llm_calls if item.status == "success"
                ),
                total_cost_usd=sum(item.cost_usd for item in llm_calls),
            ),
        }

    def list_agent_runs(
        self,
        *,
        request_id: str | None = None,
        agent_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, object]:
        items, total = self._repository.list_agent_runs(
            request_id=request_id,
            agent_id=agent_id,
            status=status,
            limit=limit,
            offset=offset,
        )
        return {
            "items": items,
            "total": total,
            "limit": limit,
            "offset": offset,
            "filters": {
                "request_id": request_id,
                "agent_id": agent_id,
                "status": status,
            },
        }

    def list_tool_calls(
        self,
        *,
        request_id: str | None = None,
        agent_id: str | None = None,
        tool_name: str | None = None,
        status: str | None = None,
        error_category: str | None = None,
        retryable: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, object]:
        items, total = self._repository.list_tool_calls(
            request_id=request_id,
            agent_id=agent_id,
            tool_name=tool_name,
            status=status,
            error_category=error_category,
            retryable=retryable,
            limit=limit,
            offset=offset,
        )
        return {
            "items": items,
            "total": total,
            "limit": limit,
            "offset": offset,
            "filters": {
                "request_id": request_id,
                "agent_id": agent_id,
                "tool_name": tool_name,
                "status": status,
                "error_category": error_category,
                "retryable": retryable,
            },
        }

    def list_llm_calls(
        self,
        *,
        request_id: str | None = None,
        agent_id: str | None = None,
        model: str | None = None,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, object]:
        items, total = self._repository.list_llm_call_logs(
            request_id=request_id,
            agent_id=agent_id,
            model=model,
            status=status,
            limit=limit,
            offset=offset,
        )
        return {
            "items": items,
            "total": total,
            "limit": limit,
            "offset": offset,
            "filters": {
                "request_id": request_id,
                "agent_id": agent_id,
                "model": model,
                "status": status,
            },
        }

    def close(self) -> None:
        self._repository.close()

    def list_workflow_runs(
        self,
        *,
        request_id: str | None = None,
        status: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, object]:
        if self._workflow_run_repository is None:
            raise NotFoundError(
                "Workflow run repository is not available.",
                details={"resource": "workflow_run_repository"},
            )
        items, total = self._workflow_run_repository.list_workflow_runs(
            request_id=request_id,
            status=status,
            limit=limit,
            offset=offset,
        )
        return {
            "items": items,
            "total": total,
            "limit": limit,
            "offset": offset,
            "filters": {
                "request_id": request_id,
                "status": status,
            },
        }

    def get_workflow_run_detail(self, request_id: str) -> dict[str, object]:
        if self._workflow_run_repository is None:
            raise NotFoundError(
                "Workflow run repository is not available.",
                details={"resource": "workflow_run_repository"},
            )
        workflow_run = self._workflow_run_repository.get_by_request_id(request_id)
        if workflow_run is None:
            raise NotFoundError(
                "Workflow run was not found.",
                details={"request_id": request_id},
            )
        agent_runs = self._repository.list_agent_runs_by_request_id(request_id)
        tool_calls = self._repository.list_tool_calls_by_request_id(request_id)
        llm_calls = self._repository.list_llm_call_logs_by_request_id(request_id)
        return {
            "workflow_run": workflow_run,
            "agent_runs": agent_runs,
            "llm_calls": llm_calls,
            "aggregates": _build_runtime_aggregates(
                task_run_count=1,
                agent_run_count=len(agent_runs),
                tool_call_count=len(tool_calls),
                llm_call_count=len(llm_calls),
                total_llm_tokens=sum(
                    item.total_tokens for item in llm_calls if item.status == "success"
                ),
                total_cost_usd=sum(item.cost_usd for item in llm_calls),
            ),
        }

    def list_sessions(
        self,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
        status: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, object]:
        items, total = self._repository.list_sessions(
            session_id=session_id,
            user_id=user_id,
            status=status,
            limit=limit,
            offset=offset,
        )
        return {
            "items": items,
            "total": total,
            "limit": limit,
            "offset": offset,
            "filters": {
                "session_id": session_id,
                "user_id": user_id,
                "status": status,
            },
        }

    def get_session_detail(self, session_id: str) -> dict[str, object]:
        session = self._repository.get_session_by_session_id(session_id)
        if session is None:
            raise NotFoundError(
                "Session was not found.",
                details={"session_id": session_id},
            )
        task_runs = self._repository.list_task_runs_by_session_id(session_id)
        agent_run_count = 0
        tool_call_count = 0
        llm_calls: list[LLMCallLog] = []
        for task_run in task_runs:
            agent_run_count += len(
                self._repository.list_agent_runs_by_request_id(task_run.request_id)
            )
            tool_call_count += len(
                self._repository.list_tool_calls_by_request_id(task_run.request_id)
            )
            llm_calls.extend(
                self._repository.list_llm_call_logs_by_request_id(task_run.request_id)
            )
        return {
            "session": session,
            "task_runs": task_runs,
            "aggregates": _build_runtime_aggregates(
                task_run_count=len(task_runs),
                agent_run_count=agent_run_count,
                tool_call_count=tool_call_count,
                llm_call_count=len(llm_calls),
                total_llm_tokens=sum(
                    item.total_tokens for item in llm_calls if item.status == "success"
                ),
                total_cost_usd=sum(item.cost_usd for item in llm_calls),
            ),
        }
