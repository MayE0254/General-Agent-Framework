import pytest

sqlalchemy = pytest.importorskip("sqlalchemy")

from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.agents.base.schemas import AgentExecutionStatus, AgentResult
from app.models import Base
from app.orchestrator import (
    AgentRunRecord,
    OrchestratorState,
    WorkflowStatus,
    WorkflowTerminalReason,
)
from app.repositories import RuntimeAuditRepository
from app.services import RuntimeRecordService
from app.tools import (
    ToolCallRequest,
    ToolContext,
    ToolErrorCategory,
    ToolExecutionStatus,
    ToolResult,
)


def test_runtime_record_service_persists_session_task_and_agent_logs() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)

    with session_factory() as session:
        repository = RuntimeAuditRepository(session)
        service = RuntimeRecordService(repository)
        state = OrchestratorState(
            request_id="req-audit-1",
            session_id="session-001",
            user_id="user-001",
            workflow_id="workflow-001",
            trace_id="trace-001",
            initial_agent_id="planner_agent",
            status=WorkflowStatus.COMPLETED,
            terminal_reason=WorkflowTerminalReason.NATURAL_COMPLETION,
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
                summary="completed",
            ),
        )

        saved = service.save_runtime_records(state)
        session.commit()

        stored_task = repository.get_task_run_by_request_id("req-audit-1")

    assert saved["session"] is not None
    assert stored_task is not None
    assert stored_task.session_id == "session-001"
    assert stored_task.final_agent_id == "reviewer_agent"
    assert (
        stored_task.metadata_payload["terminal_reason"]
        == WorkflowTerminalReason.NATURAL_COMPLETION.value
    )
    assert len(saved["agent_runs"]) == 2


def test_runtime_record_service_marks_continuation_source_review() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)

    with session_factory() as session:
        repository = RuntimeAuditRepository(session)
        service = RuntimeRecordService(repository)
        state = OrchestratorState(
            request_id="req-cont-1",
            session_id="session-cont-1",
            workflow_id="workflow-cont-1",
            trace_id="trace-cont-1",
            initial_agent_id="reviewer_agent",
            status=WorkflowStatus.COMPLETED,
            terminal_reason=WorkflowTerminalReason.NATURAL_COMPLETION,
            memory={"human_review": {"source_review_id": "review-source-1"}},
            final_result=AgentResult(
                agent_id="reviewer_agent",
                status=AgentExecutionStatus.SUCCESS,
                summary="completed after approval",
            ),
        )

        service.save_runtime_records(state)
        session.commit()

        stored_task = repository.get_task_run_by_request_id("req-cont-1")

    assert stored_task is not None
    assert stored_task.metadata_payload["source_review_id"] == "review-source-1"
    # A continuation run is not itself a pending review, so review_id stays None.
    assert stored_task.metadata_payload["review_id"] is None


def test_runtime_record_service_builds_and_persists_tool_call_log() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)

    with session_factory() as session:
        repository = RuntimeAuditRepository(session)
        service = RuntimeRecordService(repository)
        tool_log = service.build_tool_call_log(
            tool_context=ToolContext(
                request_id="req-tool-1",
                agent_id="knowledge_agent",
                session_id="session-tool-1",
                trace_id="trace-tool-1",
            ),
            tool_request=ToolCallRequest(
                tool_name="knowledge_search",
                arguments={"query": "enterprise agent framework"},
            ),
            tool_result=ToolResult(
                tool_name="knowledge_search",
                status=ToolExecutionStatus.TIMEOUT,
                message="timed out",
                content={"hits": 3},
                errors=["http_timeout"],
                error_category=ToolErrorCategory.TIMEOUT,
                retryable=True,
                error_detail="request timed out",
                metrics={"latency_ms": 1200.5},
            ),
            workflow_id="workflow-tool-1",
        )

        saved_tool_log = service.save_tool_call_log(tool_log)
        session.commit()

    assert saved_tool_log.request_id == "req-tool-1"
    assert saved_tool_log.tool_name == "knowledge_search"
    assert saved_tool_log.content["hits"] == 3
    assert saved_tool_log.status == ToolExecutionStatus.TIMEOUT.value
    assert saved_tool_log.metrics["error_category"] == ToolErrorCategory.TIMEOUT.value
    assert saved_tool_log.metrics["retryable"] is True
    assert saved_tool_log.metrics["error_detail"] == "request timed out"
