import asyncio
import uuid

import pytest

sqlalchemy = pytest.importorskip("sqlalchemy")

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.agents.base.schemas import AgentExecutionStatus, AgentResult
from app.agents.registry import create_default_registry
from app.models import Base
from app.orchestrator import (
    OrchestratorState,
    SimpleOrchestrator,
    WorkflowStatus,
    WorkflowTerminalReason,
)
from app.repositories import (
    HumanReviewRepository,
    RuntimeAuditRepository,
    WorkflowRunRepository,
)
from app.services import (
    HumanReviewService,
    RuntimeRecordService,
    WorkflowRunService,
    event_to_sse_frame,
)
from app.schemas import ReviewContinuationRequest


def _make_state(request_id: str, **overrides) -> OrchestratorState:
    state = OrchestratorState(
        request_id=request_id,
        session_id="session-review-1",
        user_id="user-review-1",
        workflow_id="workflow-review-1",
        trace_id=f"trace-{request_id}",
        initial_agent_id="planner_agent",
        input_text="draft an approval request",
        structured_input={"requires_human": True},
        shared_state={"agent_outputs": {"planner_agent": {"ok": True}}},
        status=WorkflowStatus.NEEDS_HUMAN_REVIEW,
        terminal_reason=WorkflowTerminalReason.NEEDS_HUMAN_REVIEW,
        execution_path=["planner_agent", "router_agent"],
        final_result=AgentResult(
            agent_id="reviewer_agent",
            status=AgentExecutionStatus.NEEDS_HUMAN_REVIEW,
            summary="Review required before proceeding.",
            output={"review_passed": False},
            requires_human=True,
        ),
    )
    return state.model_copy(update=overrides)


def _make_service(engine) -> tuple[HumanReviewService, Session]:
    session_factory = sessionmaker(
        bind=engine, class_=Session, expire_on_commit=False
    )
    session = session_factory()
    service = HumanReviewService(HumanReviewRepository(session))
    return service, session


def test_review_service_full_lifecycle() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    service, session = _make_service(engine)

    try:
        created = asyncio.run(
            service.create_from_state(_make_state("req-review-1"))
        )
        assert created.status == "pending"
        assert created.request_id == "req-review-1"
        assert created.agent_id == "reviewer_agent"
        assert created.payload["workflow_id"] == "workflow-review-1"
        assert created.payload["trace_id"] == "trace-req-review-1"
        assert (
            created.payload["terminal_reason"]
            == WorkflowTerminalReason.NEEDS_HUMAN_REVIEW.value
        )
        assert created.payload["execution_path"] == [
            "planner_agent",
            "router_agent",
        ]

        listing = service.list_reviews(status="pending")
        assert listing["total"] == 1
        assert listing["items"][0].id == created.id

        approved = service.approve(
            created.id,
            decided_by="human-approver",
            note="looks good",
        )
        assert approved.status == "approved"
        assert approved.decided_by == "human-approver"
        assert approved.decided_at is not None

        with pytest.raises(ValueError):
            service.approve(created.id, decided_by="again", note="no-op")

        # Already decided -> rejecting must also raise, not silently succeed.
        with pytest.raises(ValueError):
            service.reject(created.id, decided_by="other", note="no-op")
    finally:
        service.close()
        session.close()


def test_review_service_missing_task_raises_key_error() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    service, session = _make_service(engine)

    try:
        with pytest.raises(KeyError):
            service.approve("missing-review-id", decided_by="x")
        with pytest.raises(KeyError):
            service.reject("missing-review-id", decided_by="x")
        assert service.get_review("missing-review-id") is None
    finally:
        service.close()
        session.close()


def test_review_service_event_frames() -> None:
    frame = event_to_sse_frame({"type": "review.created", "review_id": "r-1"})
    assert frame.startswith("data: ")
    assert frame.endswith("\n\n")
    assert '"review_id": "r-1"' in frame


def test_review_service_builds_continuation_state_from_snapshot() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    service, session = _make_service(engine)

    try:
        created = asyncio.run(
            service.create_from_state(
                _make_state(
                    "req-review-cont-1",
                    knowledge_refs=["kb-1"],
                    shared_state={
                        "agent_outputs": {"planner_agent": {"ok": True}},
                        "custom": {"v": 1},
                    },
                    memory={"human_review": {"review_id": "old-review-id"}},
                )
            )
        )

        approved = service.approve(
            created.id,
            decided_by="human-approver",
            note="resume execution",
        )
        state = service.build_continuation_state(
            approved,
            continuation=ReviewContinuationRequest(
                request_id="req-review-cont-2",
                agent_id="reviewer_agent",
                input_text="continue after approval",
                structured_input={"requires_human": False, "approved": True},
            ),
        )

        assert state.request_id == "req-review-cont-2"
        assert state.trace_id == "trace-req-review-cont-1"
        assert state.workflow_id == "workflow-review-1"
        assert state.knowledge_refs == ["kb-1"]
        assert state.structured_input["requires_human"] is False
        assert state.structured_input["approved"] is True
        assert state.shared_state["review_context"]["source_review_id"] == created.id
        assert state.shared_state["review_context"]["approved_by"] == "human-approver"
        assert state.memory["human_review"]["source_review_id"] == created.id
        assert "review_id" not in state.memory["human_review"]
        assert state.memory["human_review"]["previous_review_id"] == "old-review-id"
    finally:
        service.close()
        session.close()


def test_orchestrator_creates_review_task_and_persists_status() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(
        bind=engine, class_=Session, expire_on_commit=False
    )

    with session_factory() as session:
        review_service = HumanReviewService(HumanReviewRepository(session))
        runtime_record_service = RuntimeRecordService(
            RuntimeAuditRepository(session)
        )
        workflow_run_service = WorkflowRunService(
            WorkflowRunRepository(session)
        )
        orchestrator = SimpleOrchestrator(
            create_default_registry(),
            runtime_record_service=runtime_record_service,
            workflow_run_service=workflow_run_service,
            human_review_service=review_service,
        )
        request_id = f"req-hr-{uuid.uuid4().hex[:8]}"
        state = OrchestratorState(
            request_id=request_id,
            session_id="session-hr-1",
            user_id="user-hr-1",
            workflow_id="workflow-hr-1",
            initial_agent_id="planner_agent",
            input_text="approval needed for this request",
            structured_input={"requires_human": True},
        )

        result_state = asyncio.run(orchestrator.run(state))

        listing = review_service.list_reviews(status="pending")

    with session_factory() as query_session:
        stored_workflow = WorkflowRunRepository(
            query_session
        ).get_by_request_id(request_id)

    assert result_state.status == WorkflowStatus.NEEDS_HUMAN_REVIEW
    assert listing["total"] == 1
    review = listing["items"][0]
    assert review.request_id == request_id
    assert review.payload["workflow_id"] == "workflow-hr-1"
    assert review.status == "pending"
    assert result_state.memory["human_review"]["review_id"] == review.id
    # The workflow_run row must mirror the review-pending status.
    assert stored_workflow is not None
    assert stored_workflow.status == WorkflowStatus.NEEDS_HUMAN_REVIEW.value
    assert stored_workflow.final_agent_id == "reviewer_agent"


def _run_review_pending_workflow(api_client, request_id: str) -> None:
    response = api_client.post(
        "/api/v1/workflows/run",
        json={
            "request_id": request_id,
            "initial_agent_id": "planner_agent",
            "session_id": "session-api-review",
            "user_id": "user-api-review",
            "input_text": "this task needs human approval",
            "structured_input": {"requires_human": True},
        },
    )
    assert response.status_code == 200
    assert response.json()["status"] == "needs_human_review"


def test_review_service_records_continuation_linkage() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    service, session = _make_service(engine)

    try:
        created = asyncio.run(
            service.create_from_state(_make_state("req-review-cont-link"))
        )
        approved = service.approve(
            created.id,
            decided_by="human-approver",
            note="resume execution",
        )
        continuation_state = service.build_continuation_state(
            approved,
            continuation=ReviewContinuationRequest(
                request_id="req-review-cont-link-2",
                agent_id="reviewer_agent",
                input_text="continue after approval",
            ),
        )
        result_state = continuation_state.model_copy(
            update={
                "status": WorkflowStatus.COMPLETED,
                "terminal_reason": WorkflowTerminalReason.NATURAL_COMPLETION,
            }
        )

        updated = service.record_continuation(approved, result_state)

        assert updated.continuation_request_id == "req-review-cont-link-2"
        assert updated.continuation_status == WorkflowStatus.COMPLETED.value
        assert (
            updated.continuation_terminal_reason
            == WorkflowTerminalReason.NATURAL_COMPLETION.value
        )
        assert updated.continuation_run_at is not None

        # The linkage must survive a fresh fetch from the repository.
        refetched = service.get_review(created.id)
        assert refetched is not None
        assert refetched.continuation_request_id == "req-review-cont-link-2"
        assert refetched.continuation_status == WorkflowStatus.COMPLETED.value
    finally:
        service.close()
        session.close()


def test_review_service_creates_chained_review_with_source() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    service, session = _make_service(engine)

    try:
        source_review_id = "review-source-000"
        chained = asyncio.run(
            service.create_from_state(
                _make_state(
                    "req-review-chained",
                    memory={
                        "human_review": {"source_review_id": source_review_id}
                    },
                )
            )
        )
        assert chained.source_review_id == source_review_id

        # A normal (non-continuation) review has no source linkage.
        standalone = asyncio.run(
            service.create_from_state(_make_state("req-review-standalone"))
        )
        assert standalone.source_review_id is None
    finally:
        service.close()
        session.close()


def test_review_api_endpoints(api_client) -> None:
    request_id = f"api-review-{uuid.uuid4().hex[:8]}"
    _run_review_pending_workflow(api_client, request_id)

    listing = api_client.get(
        "/api/v1/runtime/reviews",
        params={"status": "pending"},
    )
    assert listing.status_code == 200
    body = listing.json()
    assert body["pagination"]["filters"] == {"status": "pending"}
    review = next(
        item for item in body["items"] if item["request_id"] == request_id
    )
    review_id = review["id"]
    assert review["status"] == "pending"
    assert review["payload"]["workflow_id"] is None

    detail = api_client.get(f"/api/v1/runtime/reviews/{review_id}")
    assert detail.status_code == 200
    assert detail.json()["id"] == review_id

    approved = api_client.post(
        f"/api/v1/runtime/reviews/{review_id}/approve",
        json={"decided_by": "tester", "note": "approved in test"},
    )
    assert approved.status_code == 200
    assert approved.json()["review"]["status"] == "approved"
    assert approved.json()["review"]["decided_by"] == "tester"
    assert approved.json()["continuation"] is None
    assert approved.json()["continuation_run"] is None

    # Double decision must conflict.
    again = api_client.post(
        f"/api/v1/runtime/reviews/{review_id}/approve",
        json={"note": "again"},
    )
    assert again.status_code == 409

    missing = api_client.get("/api/v1/runtime/reviews/not-a-real-id")
    assert missing.status_code == 404


def test_review_api_reject_and_sse(api_client) -> None:
    request_id = f"api-review-{uuid.uuid4().hex[:8]}"
    _run_review_pending_workflow(api_client, request_id)

    listing = api_client.get("/api/v1/runtime/reviews")
    assert listing.status_code == 200
    review = next(
        item for item in listing.json()["items"] if item["request_id"] == request_id
    )

    rejected = api_client.post(
        f"/api/v1/runtime/reviews/{review['id']}/reject",
        json={"note": "rejected in test"},
    )
    assert rejected.status_code == 200
    assert rejected.json()["status"] == "rejected"


def test_review_api_approve_with_continuation(api_client) -> None:
    request_id = f"api-review-{uuid.uuid4().hex[:8]}"
    _run_review_pending_workflow(api_client, request_id)

    listing = api_client.get(
        "/api/v1/runtime/reviews",
        params={"status": "pending"},
    )
    assert listing.status_code == 200
    review = next(
        item for item in listing.json()["items"] if item["request_id"] == request_id
    )

    approved = api_client.post(
        f"/api/v1/runtime/reviews/{review['id']}/approve",
        json={
            "decided_by": "tester",
            "note": "resume after review",
            "continuation": {
                "request_id": f"{request_id}-continuation",
                "agent_id": "reviewer_agent",
                "input_text": "continue after approval",
                "structured_input": {"requires_human": False},
            },
        },
    )
    assert approved.status_code == 200
    body = approved.json()
    assert body["review"]["status"] == "approved"
    assert body["continuation"]["source_review_id"] == review["id"]
    assert body["continuation"]["source_request_id"] == request_id
    assert body["continuation"]["source_workflow_status"] == "needs_human_review"
    assert body["continuation"]["source_terminal_reason"] == "needs_human_review"
    assert body["continuation"]["resumed_agent_id"] == "reviewer_agent"
    assert body["continuation_run"]["request_id"] == f"{request_id}-continuation"
    assert body["continuation_run"]["status"] == "completed"
    assert body["continuation_run"]["review_id"] is None
    assert body["continuation_run"]["source_review_id"] == review["id"]

    # The approve -> continuation linkage must be persisted on the review
    # row and surfaced by later review queries.
    detail = api_client.get(f"/api/v1/runtime/reviews/{review['id']}")
    assert detail.status_code == 200
    detail_body = detail.json()
    assert (
        detail_body["continuation_request_id"] == f"{request_id}-continuation"
    )
    assert detail_body["continuation_status"] == "completed"
    assert detail_body["continuation_terminal_reason"] == "natural_completion"
    assert detail_body["continuation_run_at"] is not None
    assert detail_body["source_review_id"] is None

    # The continuation request's runtime records must point back to the
    # source review for cross-linking from runtime queries.
    continuation_task = api_client.get(
        f"/api/v1/runtime/tasks/{request_id}-continuation"
    )
    assert continuation_task.status_code == 200
    metadata = continuation_task.json()["task_run"]["metadata_payload"]
    assert metadata["source_review_id"] == review["id"]


def test_review_sse_endpoint_returns_event_stream(
    api_client,
) -> None:
    # The SSE endpoint must answer with an event-stream response. Inspect the
    # response contract directly; a live client stream would never terminate
    # because the server generator blocks until an event is published.
    from fastapi.responses import StreamingResponse

    from app.api.routes.reviews import stream_review_events

    service = api_client.app.state.human_review_service
    assert service is not None
    response = asyncio.run(stream_review_events(service))
    assert isinstance(response, StreamingResponse)
    assert response.media_type == "text/event-stream"
    assert response.headers["x-accel-buffering"] == "no"


def test_review_broker_publishes_created_and_decided_events() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    service, session = _make_service(engine)

    async def _run() -> None:
        queue = service.subscribe()
        try:
            created = await service.create_from_state(_make_state("req-evt-1"))
            created_event = await asyncio.wait_for(queue.get(), timeout=2)
            assert created_event["type"] == "review.created"
            assert created_event["review_id"] == created.id

            service.approve(created.id, decided_by="x", note="ok")
            decided_event = await asyncio.wait_for(queue.get(), timeout=2)
            assert decided_event["type"] == "review.decided"
            assert decided_event["status"] == "approved"
        finally:
            service.unsubscribe(queue)

    try:
        asyncio.run(_run())
    finally:
        service.close()
        session.close()
