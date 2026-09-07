import asyncio

import pytest

sqlalchemy = pytest.importorskip("sqlalchemy")

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.agents.registry import create_default_registry
from app.core.settings import LLMSettings
from app.llm import LLMCallAudit, LLMError, LLMMessage, LLMRequest, LLMRole, LLMService
from app.models import Base
from app.orchestrator import OrchestratorState, SimpleOrchestrator
from app.repositories import RuntimeAuditRepository
from app.services import RuntimeQueryService, RuntimeRecordService


def _make_raw(model: str, content: str) -> object:
    class _Message:
        content: str = ""

    class _Choice:
        message: object = None

    class _Usage:
        prompt_tokens = 3
        completion_tokens = 2
        total_tokens = 5

    class _Raw:
        model: str = ""
        choices: list = []
        usage: object = None

    raw = _Raw()
    raw.model = model
    message = _Message()
    message.content = content
    raw.choices = [_Choice()]
    raw.choices[0].message = message
    raw.usage = _Usage()
    return raw


def _request(**overrides) -> LLMRequest:
    base = {
        "model": "gpt-test",
        "messages": [LLMMessage(role=LLMRole.USER, content="hello")],
        "request_id": "req-llm-1",
        "trace_id": "trace-llm-1",
        "agent_id": "planner_agent",
    }
    base.update(overrides)
    return LLMRequest(**base)


def test_llm_service_emits_success_audit() -> None:
    audits: list[LLMCallAudit] = []

    async def _fake_completion(kwargs: dict) -> object:
        return _make_raw(kwargs["model"], "ok")

    service = LLMService(
        LLMSettings(api_key="sk-real", model="gpt-test"),
        completion_fn=_fake_completion,
        audit_recorder=audits.append,
    )

    asyncio.run(service.complete(_request()))

    assert len(audits) == 1
    audit = audits[0]
    assert audit.status == "success"
    assert audit.model == "gpt-test"
    assert audit.request_id == "req-llm-1"
    assert audit.trace_id == "trace-llm-1"
    assert audit.agent_id == "planner_agent"
    assert audit.prompt_tokens == 3
    assert audit.completion_tokens == 2
    assert audit.total_tokens == 5
    assert audit.latency_ms >= 0
    assert audit.error is None
    assert len(audit.messages) == 1


def test_llm_service_emits_failed_audit_on_call_error() -> None:
    audits: list[LLMCallAudit] = []

    async def _fake_completion(kwargs: dict) -> object:
        raise ValueError("upstream exploded")

    service = LLMService(
        LLMSettings(api_key="sk-real", model="gpt-test"),
        completion_fn=_fake_completion,
        audit_recorder=audits.append,
    )

    with pytest.raises(LLMError):
        asyncio.run(service.complete(_request()))

    assert len(audits) == 1
    audit = audits[0]
    assert audit.status == "failed"
    assert "LLM call failed" in (audit.error or "")
    assert audit.total_tokens == 0


def test_llm_service_emits_failed_audit_when_not_configured() -> None:
    audits: list[LLMCallAudit] = []
    service = LLMService(
        LLMSettings(api_key="replace_me", model="gpt-test"),
        audit_recorder=audits.append,
    )

    with pytest.raises(LLMError):
        asyncio.run(service.complete(_request()))

    assert len(audits) == 1
    assert audits[0].status == "failed"
    assert "not configured" in (audits[0].error or "")


def _build_workflow_fake_completion():
    async def _fake_completion(kwargs: dict) -> object:
        system = kwargs["messages"][0]["content"]
        if "workflow router" in system:
            content = '{"next_agent": "executor_agent"}'
        elif "task executor" in system:
            content = '{"result": "fake execution output"}'
        elif "task planner" in system:
            content = '{"plan_steps": ["step 1"], "tool_query": ""}'
        else:
            content = '{"requires_human": false, "feedback": "ok"}'
        return _make_raw(kwargs["model"], content)

    return _fake_completion


def test_workflow_persists_llm_call_logs_to_audit_store() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)

    with session_factory() as session:
        repository = RuntimeAuditRepository(session)
        record_service = RuntimeRecordService(repository)
        llm_service = LLMService(
            LLMSettings(
                api_key="sk-real",
                model="gpt-test",
                providers={
                    "deepseek": {
                        "api_base": "https://api.deepseek.com",
                        "api_key": "sk-deepseek",
                        "model": "deepseek/deepseek-chat",
                    }
                },
            ),
            completion_fn=_build_workflow_fake_completion(),
            audit_recorder=record_service.save_llm_call_log,
        )
        orchestrator = SimpleOrchestrator(
            create_default_registry(),
            runtime_record_service=record_service,
            llm_service=llm_service,
        )
        state = OrchestratorState(
            request_id="req-llm-audit-1",
            session_id="session-llm-1",
            workflow_id="workflow-llm-1",
            trace_id="trace-llm-1",
            initial_agent_id="planner_agent",
            input_text="analyze the quarterly numbers",
            structured_input={"requested_agent": "executor_agent"},
        )

        result = asyncio.run(orchestrator.run(state))

        assert result.status.value == "completed"
        llm_calls = repository.list_llm_call_logs()[0]
        assert len(llm_calls) == 4
        assert {item.agent_id for item in llm_calls} == {
            "planner_agent",
            "router_agent",
            "executor_agent",
            "reviewer_agent",
        }
        for item in llm_calls:
            assert item.request_id == "req-llm-audit-1"
            assert item.trace_id == "trace-llm-1"
            assert item.workflow_id == "workflow-llm-1"
            assert item.session_id == "session-llm-1"
            assert item.status == "success"
            assert item.total_tokens == 5
            assert item.model == "gpt-test"
            assert item.messages

        # Query-side: filtered listing and per-request detail aggregates.
        query_service = RuntimeQueryService(repository)
        filtered = query_service.list_llm_calls(agent_id="router_agent")
        assert filtered["total"] == 1
        assert filtered["items"][0].agent_id == "router_agent"

        detail = query_service.get_task_run_detail("req-llm-audit-1")
        assert detail["aggregates"]["llm_call_count"] == 4
        assert detail["aggregates"]["total_llm_tokens"] == 20
        assert len(detail["llm_calls"]) == 4
        query_service.close()
