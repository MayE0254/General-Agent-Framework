import pytest

sqlalchemy = pytest.importorskip("sqlalchemy")

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.agents.base.schemas import AgentExecutionStatus, AgentResult
from app.models import Base
from app.orchestrator import (
    OrchestratorState,
    WorkflowStatus,
    WorkflowTerminalReason,
)
from app.repositories import WorkflowRunRepository
from app.services import WorkflowRunService


def test_workflow_run_service_persists_example_record() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)

    with session_factory() as session:
        repository = WorkflowRunRepository(session)
        service = WorkflowRunService(repository)
        state = OrchestratorState(
            request_id="req-persist-1",
            initial_agent_id="planner_agent",
            workflow_id="workflow-001",
            trace_id="trace-001",
            knowledge_refs=["kb-1"],
            status=WorkflowStatus.COMPLETED,
            terminal_reason=WorkflowTerminalReason.NATURAL_COMPLETION,
            execution_path=["planner_agent", "reviewer_agent"],
            errors=[],
            final_result=AgentResult(
                agent_id="reviewer_agent",
                status=AgentExecutionStatus.SUCCESS,
                summary="Workflow completed successfully.",
            ),
        )

        record = service.save_workflow_run(state)
        session.commit()

        stored = repository.get_by_request_id("req-persist-1")

    assert stored is not None
    assert stored.id == record.id
    assert stored.final_agent_id == "reviewer_agent"
    assert stored.status == WorkflowStatus.COMPLETED.value
    assert stored.metadata_payload["trace_id"] == "trace-001"
    assert (
        stored.metadata_payload["terminal_reason"]
        == WorkflowTerminalReason.NATURAL_COMPLETION.value
    )
