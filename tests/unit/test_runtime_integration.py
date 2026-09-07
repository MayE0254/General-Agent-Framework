import asyncio

import pytest

sqlalchemy = pytest.importorskip("sqlalchemy")

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.agents.registry import create_default_registry
from app.agents.system import PlannerAgent
from app.models import Base
from app.orchestrator import OrchestratorState, SimpleOrchestrator, WorkflowStatus
from app.repositories import RuntimeAuditRepository, WorkflowRunRepository
from app.services import RuntimeQueryService, RuntimeRecordService, WorkflowRunService
from app.tools import (
    BaseTool,
    ToolCallRequest,
    ToolContext,
    ToolExecutionStatus,
    ToolExecutor,
    ToolMetadata,
    ToolRegistry,
    ToolResult,
)


class FakeKnowledgeSearchTool(BaseTool):
    @classmethod
    def build_metadata(cls) -> ToolMetadata:
        return ToolMetadata(
            tool_name="knowledge_search",
            description="Returns a fake search payload.",
            tags=["test"],
        )

    async def _run(self, context: ToolContext, arguments: dict) -> ToolResult:
        return ToolResult(
            tool_name=self.metadata.tool_name,
            status=ToolExecutionStatus.SUCCESS,
            content={"query": arguments.get("query"), "hits": 2},
            message="fake tool executed",
        )


def test_orchestrator_auto_persists_runtime_records() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)

    with session_factory() as session:
        repository = RuntimeAuditRepository(session)
        workflow_repository = WorkflowRunRepository(session)
        runtime_record_service = RuntimeRecordService(repository)
        workflow_run_service = WorkflowRunService(workflow_repository)
        orchestrator = SimpleOrchestrator(
            create_default_registry(),
            runtime_record_service=runtime_record_service,
            workflow_run_service=workflow_run_service,
        )
        state = OrchestratorState(
            request_id="req-auto-1",
            session_id="session-auto-1",
            workflow_id="workflow-auto-1",
            trace_id="trace-auto-1",
            initial_agent_id="planner_agent",
            input_text="prepare a reusable platform",
            structured_input={"requested_agent": "executor_agent"},
        )

        result_state = asyncio.run(orchestrator.run(state))

        stored_task = repository.get_task_run_by_request_id("req-auto-1")
        stored_agent_runs = repository.list_agent_runs_by_request_id("req-auto-1")
        stored_workflow = workflow_repository.get_by_request_id("req-auto-1")

    assert result_state.status == WorkflowStatus.COMPLETED
    assert stored_task is not None
    assert stored_task.final_agent_id == "reviewer_agent"
    assert len(stored_agent_runs) == 4
    assert stored_agent_runs[0].agent_id == "planner_agent"
    # WorkflowRunService mounted on the orchestrator persists workflow runs.
    assert stored_workflow is not None
    assert stored_workflow.final_agent_id == "reviewer_agent"
    assert stored_workflow.execution_path == [
        "planner_agent",
        "router_agent",
        "executor_agent",
        "reviewer_agent",
    ]


def test_tool_executor_auto_persists_tool_call_logs() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)

    with session_factory() as session:
        repository = RuntimeAuditRepository(session)
        runtime_record_service = RuntimeRecordService(repository)
        registry = ToolRegistry()
        registry.register(FakeKnowledgeSearchTool)
        executor = ToolExecutor(
            registry,
            runtime_record_service=runtime_record_service,
        )
        agent = PlannerAgent()

        result = asyncio.run(
            executor.execute(
                agent=agent,
                request=ToolCallRequest(
                    tool_name="knowledge_search",
                    arguments={"query": "auto audit"},
                ),
                context=ToolContext(
                    request_id="req-tool-auto-1",
                    agent_id=agent.metadata.agent_id,
                    session_id="session-tool-auto-1",
                    trace_id="trace-tool-auto-1",
                ),
            )
        )

        stored_tool_calls = repository.list_tool_calls_by_request_id("req-tool-auto-1")

    assert result.status == ToolExecutionStatus.SUCCESS
    assert len(stored_tool_calls) == 1
    assert stored_tool_calls[0].tool_name == "knowledge_search"
    assert stored_tool_calls[0].content["hits"] == 2


def test_workflow_run_query_service_lists_and_details() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)

    with session_factory() as session:
        repository = RuntimeAuditRepository(session)
        workflow_repository = WorkflowRunRepository(session)
        workflow_run_service = WorkflowRunService(workflow_repository)
        orchestrator = SimpleOrchestrator(
            create_default_registry(),
            runtime_record_service=RuntimeRecordService(repository),
            workflow_run_service=workflow_run_service,
        )
        state = OrchestratorState(
            request_id="req-query-1",
            session_id="session-query-1",
            workflow_id="workflow-query-1",
            trace_id="trace-query-1",
            initial_agent_id="planner_agent",
            input_text="query the workflow runs",
            structured_input={"requested_agent": "executor_agent"},
        )

        asyncio.run(orchestrator.run(state))

        query_service = RuntimeQueryService(
            repository,
            workflow_run_repository=workflow_repository,
        )
        listing = query_service.list_workflow_runs()
        detail = query_service.get_workflow_run_detail("req-query-1")

    assert listing["total"] == 1
    assert listing["items"][0].request_id == "req-query-1"
    assert listing["items"][0].final_agent_id == "reviewer_agent"
    assert detail["workflow_run"].status == WorkflowStatus.COMPLETED.value
    assert [item.agent_id for item in detail["agent_runs"]] == [
        "planner_agent",
        "router_agent",
        "executor_agent",
        "reviewer_agent",
    ]
    assert detail["aggregates"]["task_run_count"] == 1
    assert detail["aggregates"]["agent_run_count"] == 4
    assert detail["aggregates"]["tool_call_count"] == 0
    assert detail["aggregates"]["llm_call_count"] == 0
    assert detail["aggregates"]["total_cost_usd"] == 0.0


def test_workflow_run_query_service_filters_by_status() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)

    with session_factory() as session:
        repository = RuntimeAuditRepository(session)
        workflow_repository = WorkflowRunRepository(session)
        workflow_run_service = WorkflowRunService(workflow_repository)
        orchestrator = SimpleOrchestrator(
            create_default_registry(),
            runtime_record_service=RuntimeRecordService(repository),
            workflow_run_service=workflow_run_service,
        )
        for i in range(3):
            state = OrchestratorState(
                request_id=f"req-filter-{i}",
                session_id=f"session-filter-{i}",
                workflow_id="workflow-filter",
                trace_id=f"trace-filter-{i}",
                initial_agent_id="planner_agent",
                input_text=f"run {i}",
                structured_input={"requested_agent": "executor_agent"},
            )
            asyncio.run(orchestrator.run(state))

        query_service = RuntimeQueryService(
            repository,
            workflow_run_repository=workflow_repository,
        )
        completed = query_service.list_workflow_runs(status="completed")
        missing = query_service.list_workflow_runs(
            status="does-not-exist",
            limit=5,
            offset=0,
        )
        limited = query_service.list_workflow_runs(limit=2, offset=0)

    assert completed["total"] == 3
    assert missing["total"] == 0
    assert missing["items"] == []
    assert limited["total"] == 3
    assert len(limited["items"]) == 2


def test_session_query_service_lists_and_details() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)

    with session_factory() as session:
        repository = RuntimeAuditRepository(session)
        runtime_record_service = RuntimeRecordService(repository)
        orchestrator = SimpleOrchestrator(
            create_default_registry(),
            runtime_record_service=runtime_record_service,
        )
        for i in range(2):
            state = OrchestratorState(
                request_id=f"req-session-{i}",
                session_id="session-query-detail",
                workflow_id="workflow-session",
                trace_id=f"trace-session-{i}",
                initial_agent_id="planner_agent",
                input_text=f"run {i}",
                structured_input={"requested_agent": "executor_agent"},
            )
            asyncio.run(orchestrator.run(state))

        query_service = RuntimeQueryService(repository)
        listing = query_service.list_sessions(status="active")
        detail = query_service.get_session_detail("session-query-detail")

    assert listing["total"] == 1
    assert listing["items"][0].session_id == "session-query-detail"
    assert listing["items"][0].last_request_id == "req-session-1"
    assert detail["session"].status == "active"
    assert {item.request_id for item in detail["task_runs"]} == {
        "req-session-0",
        "req-session-1",
    }
    assert detail["aggregates"]["task_run_count"] == 2
    assert detail["aggregates"]["agent_run_count"] == 8
    assert detail["aggregates"]["tool_call_count"] == 0
    assert detail["aggregates"]["llm_call_count"] == 0
    assert detail["aggregates"]["total_cost_usd"] == 0.0


def test_session_query_service_returns_404_for_missing_session() -> None:
    import pytest as _pytest

    from app.core import NotFoundError

    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)

    with session_factory() as session:
        query_service = RuntimeQueryService(RuntimeAuditRepository(session))

        with _pytest.raises(NotFoundError):
            query_service.get_session_detail("missing-session")
