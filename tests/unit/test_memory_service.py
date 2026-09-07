"""Unit tests for the session-context and long-term memory system."""

import asyncio
import json

import pytest

sqlalchemy = pytest.importorskip("sqlalchemy")

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.agents.registry import create_default_registry
from app.models import Base
from app.orchestrator import OrchestratorState, SimpleOrchestrator, WorkflowStatus
from app.repositories import MemoryRepository
from app.services import MemoryService


def _session_factory():
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, class_=Session, expire_on_commit=False)


class FakeLLMService:
    """Minimal LLMService stand-in: configured, returns a fixed payload."""

    is_configured = True

    def __init__(self, payload: str) -> None:
        self._payload = payload
        self.last_request = None

    async def complete(self, request) -> object:
        self.last_request = request

        class _Response:
            content = ""
            model = "deepseek/deepseek-chat"

        response = _Response()
        response.content = self._payload
        return response


def _make_service(
    factory,
    *,
    llm_service=None,
) -> tuple[MemoryService, MemoryRepository]:
    session = factory()
    repository = MemoryRepository(session)
    return MemoryService(repository, llm_service=llm_service), repository


def test_memory_crud_roundtrip() -> None:
    factory = _session_factory()
    service, _ = _make_service(factory)

    record = service.add_memory(
        memory_type="preference",
        content="user prefers concise answers",
        scope="user",
        scope_key="user-1",
        source="manual",
        importance=4,
        tags=["style"],
    )
    listing = service.list_memories(scope="user", scope_key="user-1")
    assert listing["total"] == 1
    assert listing["items"][0].content == "user prefers concise answers"

    updated = service.update_memory(
        record.id,
        content="user prefers very concise answers",
        importance=5,
    )
    assert updated is not None
    assert updated.content == "user prefers very concise answers"
    assert updated.importance == 5

    assert service.delete_memory(record.id) is True
    assert service.delete_memory(record.id) is False
    assert service.list_memories()["total"] == 0


def test_append_and_list_session_messages_latest_first() -> None:
    factory = _session_factory()
    service, _ = _make_service(factory)

    service.append_session_messages(
        session_id="sess-1",
        request_id="req-1",
        turns=[
            {"role": "user", "content": "first question", "step_index": 1},
            {"role": "assistant", "content": "first answer", "step_index": 2},
        ],
    )
    service.append_session_messages(
        session_id="sess-1",
        request_id="req-2",
        turns=[
            {"role": "user", "content": "second question", "step_index": 1},
            {"role": "assistant", "content": "second answer", "step_index": 2},
        ],
    )

    messages = service.list_session_messages("sess-1", limit=2)
    assert service.count_session_messages("sess-1") == 4
    # The two most recent messages, oldest-first among them.
    assert [item.content for item in messages] == [
        "second question",
        "second answer",
    ]
    assert messages[0].request_id == "req-2"
    assert messages[0].step_index == 1


def test_load_session_context_recalls_user_then_project_memories() -> None:
    factory = _session_factory()
    service, _ = _make_service(factory)
    service.add_memory(
        memory_type="preference",
        content="project level fact",
        scope="project",
        scope_key="default",
        source="manual",
        importance=2,
    )
    service.add_memory(
        memory_type="preference",
        content="user level fact",
        scope="user",
        scope_key="user-1",
        source="manual",
        importance=5,
    )
    service.append_session_messages(
        session_id="sess-1",
        request_id="req-1",
        turns=[{"role": "user", "content": "hi", "step_index": 1}],
    )

    context = service.load_session_context(
        session_id="sess-1",
        user_id="user-1",
        project_key="default",
    )
    assert context["session_id"] == "sess-1"
    assert context["session_history"] == [{"role": "user", "content": "hi"}]
    recalled = [item["content"] for item in context["recalled_memories"]]
    # User-scope memories are recalled before project-scope ones, both
    # ordered by importance descending.
    assert recalled == ["user level fact", "project level fact"]


def test_extract_and_store_memories_persists_preferences_and_facts() -> None:
    factory = _session_factory()
    payload = json.dumps(
        {
            "preferences": [
                {"content": "likes short replies", "importance": 4}
            ],
            "facts": [{"content": "project targets 2026 Q3", "importance": 3}],
        }
    )
    fake_llm = FakeLLMService(payload)
    service, _ = _make_service(factory, llm_service=fake_llm)

    stored = asyncio.run(
        service.extract_and_store_memories(
            session_id="sess-1",
            user_id="user-1",
            request_id="req-1",
            user_input="hello",
            assistant_output="hi there",
            project_key="default",
        )
    )
    listing = service.list_memories(scope="user", scope_key="user-1")
    assert stored == 2
    assert {item.memory_type for item in listing["items"]} == {
        "preference",
        "fact",
    }
    assert listing["items"][0].source == "llm"
    # The extraction request carried the right session/agent metadata.
    assert fake_llm.last_request is not None
    assert fake_llm.last_request.agent_id == "memory_curator"


def test_extract_and_store_memories_best_effort_on_bad_payload() -> None:
    factory = _session_factory()
    # Non-JSON content (e.g. a fallback response) must not raise or block.
    service, _ = _make_service(factory, llm_service=FakeLLMService("not-json"))
    stored = asyncio.run(
        service.extract_and_store_memories(
            session_id="sess-1",
            user_id=None,
            request_id="req-1",
            user_input="hello",
            assistant_output="hi",
            project_key="default",
        )
    )
    assert stored == 0
    assert service.list_memories()["total"] == 0


def test_extract_and_store_memories_skips_when_unconfigured() -> None:
    factory = _session_factory()
    service, _ = _make_service(factory, llm_service=None)
    stored = asyncio.run(
        service.extract_and_store_memories(
            session_id="sess-1",
            user_id=None,
            request_id="req-1",
            user_input="hello",
            assistant_output="hi",
            project_key="default",
        )
    )
    assert stored == 0


def test_orchestrator_persists_history_and_recalls_across_runs() -> None:
    factory = _session_factory()
    session = factory()
    repository = MemoryRepository(session)
    # Extraction yields durable memories; agents keep deterministic rules.
    memory_llm = FakeLLMService(
        json.dumps(
            {
                "preferences": [
                    {"content": "user likes bullet lists", "importance": 4}
                ],
                "facts": [],
            }
        )
    )
    memory_service = MemoryService(
        repository,
        llm_service=memory_llm,
    )
    orchestrator = SimpleOrchestrator(
        create_default_registry(),
        memory_service=memory_service,
    )

    for i in range(2):
        state = OrchestratorState(
            request_id=f"req-memory-{i}",
            session_id="session-memory-1",
            workflow_id="workflow-memory",
            trace_id=f"trace-memory-{i}",
            user_id="user-memory-1",
            initial_agent_id="planner_agent",
            input_text=f"follow-up question {i}",
            structured_input={"requested_agent": "executor_agent"},
        )
        asyncio.run(orchestrator.run(state))

    messages = memory_service.list_session_messages("session-memory-1", limit=50)
    assert memory_service.count_session_messages("session-memory-1") == 4
    assert [item.role for item in messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert [item.content for item in messages][::2] == [
        "follow-up question 0",
        "follow-up question 1",
    ]

    memories = memory_service.list_memories(
        scope="user",
        scope_key="user-memory-1",
    )
    # Both runs extracted the same preference, so duplicate-merge keeps one.
    assert memories["total"] == 1
    assert memories["items"][0].content == "user likes bullet lists"
    assert memories["items"][0].metadata_payload["request_ids"] == [
        "req-memory-0",
        "req-memory-1",
    ]
    assert memories["items"][0].metadata_payload["merge_count"] == 1

    # Second run loaded the first run's history into the planner context.
    context = memory_service.load_session_context(
        session_id="session-memory-1",
        user_id="user-memory-1",
        project_key="default",
    )
    assert len(context["session_history"]) == 4


def test_add_memory_merges_duplicates_and_keeps_best_fields() -> None:
    factory = _session_factory()
    service, _ = _make_service(factory)

    first = service.add_memory(
        memory_type="preference",
        content="user prefers 简洁的要点式回复!",
        scope="user",
        scope_key="user-merge-1",
        source="manual",
        importance=3,
        tags=["style"],
    )
    # Near-identical statement (punctuation/space/case differences only).
    merged = service.add_memory(
        memory_type="preference",
        content="User  prefers 简洁的要点式回复.",
        scope="user",
        scope_key="user-merge-1",
        source="llm",
        importance=5,
        tags=["style", "tone"],
        metadata_payload={"request_id": "req-merge-1", "session_id": "sess-1"},
    )

    assert merged.id == first.id
    listing = service.list_memories(scope="user", scope_key="user-merge-1")
    assert listing["total"] == 1
    entry = listing["items"][0]
    # Longer content wins, importance takes the max, tags union, provenance
    # recorded, and the merge is observable.
    assert entry.content == "User  prefers 简洁的要点式回复."
    assert entry.importance == 5
    assert entry.tags == ["style", "tone"]
    assert entry.metadata_payload["merge_count"] == 1
    assert entry.metadata_payload["request_ids"] == ["req-merge-1"]


def test_add_memory_does_not_merge_distinct_statements() -> None:
    factory = _session_factory()
    service, _ = _make_service(factory)

    service.add_memory(
        memory_type="preference",
        content="likes concise replies",
        scope="user",
        scope_key="user-merge-2",
        source="manual",
        importance=3,
    )
    service.add_memory(
        memory_type="preference",
        content="prefers bullet lists",
        scope="user",
        scope_key="user-merge-2",
        source="manual",
        importance=4,
    )

    listing = service.list_memories(scope="user", scope_key="user-merge-2")
    assert listing["total"] == 2


def test_add_memory_merges_near_duplicate_wording() -> None:
    factory = _session_factory()
    service, _ = _make_service(factory)

    first = service.add_memory(
        memory_type="preference",
        content="用户偏好把项目事实记录成英文术语",
        scope="user",
        scope_key="user-merge-3",
        source="llm",
        importance=4,
        metadata_payload={"request_id": "req-a", "session_id": "sess-a"},
    )
    # Same statement, tiny wording drift ("将/把", "为/成").
    merged = service.add_memory(
        memory_type="preference",
        content="用户偏好将项目事实记录为英文术语",
        scope="user",
        scope_key="user-merge-3",
        source="llm",
        importance=5,
        metadata_payload={"request_id": "req-b", "session_id": "sess-b"},
    )

    assert merged.id == first.id
    listing = service.list_memories(scope="user", scope_key="user-merge-3")
    assert listing["total"] == 1
    entry = listing["items"][0]
    assert entry.importance == 5
    assert entry.metadata_payload["merge_count"] == 1
    assert entry.metadata_payload["request_ids"] == ["req-a", "req-b"]
    assert entry.metadata_payload["last_merge_similarity"] > 0.85


def test_add_memory_merge_disabled_appends_duplicates() -> None:
    from app.core.settings import MemorySettings

    factory = _session_factory()
    session = factory()
    repository = MemoryRepository(session)
    service = MemoryService(
        repository,
        settings=MemorySettings(merge_duplicates=False),
    )

    for i in range(2):
        service.add_memory(
            memory_type="preference",
            content="likes concise replies",
            scope="user",
            scope_key="user-merge-4",
            source="manual",
            importance=3,
        )

    listing = service.list_memories(scope="user", scope_key="user-merge-4")
    assert listing["total"] == 2
