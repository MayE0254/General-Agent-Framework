import pytest

sqlalchemy = pytest.importorskip("sqlalchemy")

from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.agents.base.schemas import AgentExecutionStatus, AgentResult
from app.core import NotFoundError
from app.models import Base
from app.orchestrator import AgentRunRecord, OrchestratorState, WorkflowStatus
from app.repositories import RuntimeAuditRepository
from app.services import RuntimeQueryService, RuntimeRecordService
from app.tools import (
    ToolCallRequest,
    ToolContext,
    ToolErrorCategory,
    ToolExecutionStatus,
    ToolResult,
)


def _seed_runtime_data(session: Session, *, request_id: str, status: WorkflowStatus) -> None:
    repository = RuntimeAuditRepository(session)
    record_service = RuntimeRecordService(repository)
    state = OrchestratorState(
        request_id=request_id,
        session_id=f"session-{request_id}",
        workflow_id=f"workflow-{request_id}",
        trace_id=f"trace-{request_id}",
        initial_agent_id="planner_agent",
        status=status,
        execution_path=["planner_agent", "reviewer_agent"],
        run_history=[
            AgentRunRecord(
                step_index=1,
                agent_id="planner_agent",
                status=AgentExecutionStatus.SUCCESS,
                summary="planned",
                next_agent_id="reviewer_agent",
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
            ),
            AgentRunRecord(
                step_index=2,
                agent_id="reviewer_agent",
                status=AgentExecutionStatus.SUCCESS,
                summary="reviewed",
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
            ),
        ],
        final_result=AgentResult(
            agent_id="reviewer_agent",
            status=AgentExecutionStatus.SUCCESS,
            summary=f"completed-{request_id}",
        ),
    )
    record_service.save_runtime_records(state)
    record_service.save_tool_call_result(
        tool_context=ToolContext(
            request_id=request_id,
            agent_id="planner_agent",
            session_id=f"session-{request_id}",
            trace_id=f"trace-{request_id}",
        ),
        tool_request=ToolCallRequest(
            tool_name="knowledge_search" if request_id.endswith("1") else "document_parser",
            arguments={"query": f"runtime detail {request_id}"},
        ),
        tool_result=ToolResult(
            tool_name="knowledge_search" if request_id.endswith("1") else "document_parser",
            status=ToolExecutionStatus.SUCCESS if request_id.endswith("1") else ToolExecutionStatus.DENIED,
            message="ok",
            content={"hits": 1 if request_id.endswith("1") else 0},
            errors=[] if request_id.endswith("1") else ["tool_permission_denied"],
            error_category=(
                None
                if request_id.endswith("1")
                else ToolErrorCategory.POLICY_DENIED
            ),
            retryable=False,
            error_detail=(
                None
                if request_id.endswith("1")
                else "agent permission denied by tool allowlist"
            ),
        ),
        workflow_id=f"workflow-{request_id}",
    )


def test_runtime_query_service_lists_and_reads_runtime_detail() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)

    with session_factory() as session:
        _seed_runtime_data(session, request_id="req-query-1", status=WorkflowStatus.COMPLETED)
        _seed_runtime_data(session, request_id="req-query-2", status=WorkflowStatus.FAILED)
        query_service = RuntimeQueryService(RuntimeAuditRepository(session))

        task_runs = query_service.list_task_runs(limit=10, offset=0)
        detail = query_service.get_task_run_detail("req-query-1")
        agent_runs = query_service.list_agent_runs(
            request_id="req-query-1",
            limit=10,
            offset=0,
        )
        tool_calls = query_service.list_tool_calls(
            request_id="req-query-1",
            limit=10,
            offset=0,
        )

    assert task_runs["total"] == 2
    assert len(task_runs["items"]) == 2
    assert task_runs["items"][0].request_id in {"req-query-1", "req-query-2"}
    assert detail["task_run"].request_id == "req-query-1"
    assert len(detail["agent_runs"]) == 2
    assert len(detail["tool_calls"]) == 1
    assert detail["aggregates"]["task_run_count"] == 1
    assert detail["aggregates"]["agent_run_count"] == 2
    assert detail["aggregates"]["tool_call_count"] == 1
    assert detail["aggregates"]["llm_call_count"] == 0
    assert detail["aggregates"]["total_cost_usd"] == 0.0
    assert detail["tool_calls"][0].metrics["retryable"] is False
    assert agent_runs["total"] == 2
    assert len(agent_runs["items"]) == 2
    assert tool_calls["total"] == 1
    assert len(tool_calls["items"]) == 1


def test_runtime_query_service_applies_filters_and_pagination() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)

    with session_factory() as session:
        _seed_runtime_data(session, request_id="req-query-1", status=WorkflowStatus.COMPLETED)
        _seed_runtime_data(session, request_id="req-query-2", status=WorkflowStatus.FAILED)
        query_service = RuntimeQueryService(RuntimeAuditRepository(session))

        completed_tasks = query_service.list_task_runs(
            status=WorkflowStatus.COMPLETED.value,
            limit=10,
            offset=0,
        )
        paged_tasks = query_service.list_task_runs(limit=1, offset=0)
        failed_agent_runs = query_service.list_agent_runs(
            request_id="req-query-2",
            status=AgentExecutionStatus.SUCCESS.value,
            limit=10,
            offset=0,
        )
        denied_tool_calls = query_service.list_tool_calls(
            tool_name="document_parser",
            status=ToolExecutionStatus.DENIED.value,
            error_category=ToolErrorCategory.POLICY_DENIED.value,
            retryable=False,
            limit=10,
            offset=0,
        )

    assert completed_tasks["total"] == 1
    assert completed_tasks["items"][0].request_id == "req-query-1"
    assert paged_tasks["total"] == 2
    assert len(paged_tasks["items"]) == 1
    assert failed_agent_runs["total"] == 2
    assert all(item.request_id == "req-query-2" for item in failed_agent_runs["items"])
    assert denied_tool_calls["total"] == 1
    assert denied_tool_calls["items"][0].tool_name == "document_parser"
    assert denied_tool_calls["filters"]["error_category"] == ToolErrorCategory.POLICY_DENIED.value
    assert denied_tool_calls["filters"]["retryable"] is False


def test_runtime_query_service_raises_not_found_for_unknown_request() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)

    with session_factory() as session:
        query_service = RuntimeQueryService(RuntimeAuditRepository(session))

        with pytest.raises(NotFoundError):
            query_service.get_task_run_detail("missing-request-id")
