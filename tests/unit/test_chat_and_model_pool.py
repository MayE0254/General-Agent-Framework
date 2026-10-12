"""Tests for the console chat entry point, the model pool and usage reporting.

Covers the three promises made by the chat feature:

* the model pool is admin-curated, member-selectable and never leaks keys,
* a chat turn is either streamed by the selected model or handed to a
  specialist agent through the async queue, decided by the model itself,
* every LLM call is billed back to a user and a model, which is what the
  console usage report aggregates.
"""

from __future__ import annotations

import asyncio
import json
import uuid

import pytest

sqlalchemy = pytest.importorskip("sqlalchemy")

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core import NotFoundError
from app.core.settings import LLMSettings, ModelPricingSettings
from app.llm import (
    LLMCallAudit,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMRole,
    LLMService,
)
from app.models import Base, ModelPoolEntry
from app.repositories import (
    MemoryRepository,
    ModelPoolRepository,
    RuntimeAuditRepository,
)
from app.schemas import WorkflowRunRequest
from app.services import (
    ChatService,
    DispatchedTask,
    MemoryService,
    ModelPoolService,
    RuntimeQueryService,
    RuntimeRecordService,
)


def _session_factory():
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, class_=Session, expire_on_commit=False)


# ---------- model pool ----------


def _pool_service(factory) -> ModelPoolService:
    return ModelPoolService(ModelPoolRepository(factory()))


def test_model_pool_crud_roundtrip() -> None:
    service = _pool_service(_session_factory())

    entry = service.create_entry(
        name="DeepSeek Chat",
        provider_key="deepseek",
        model="deepseek/deepseek-v4-flash",
        is_default=True,
    )
    assert service.resolve_entry().id == entry.id
    assert service.resolve_entry(name="DeepSeek Chat").id == entry.id
    assert service.resolve_entry(entry_id=entry.id).id == entry.id

    other = service.create_entry(
        name="停用的模型", provider_key="deepseek", model="deepseek/other", enabled=False
    )
    listing = service.list_entries()
    assert listing["total"] == 2
    enabled = service.list_entries(enabled_only=True)
    assert [item.id for item in enabled["items"]] == [entry.id]

    service.delete_entry(other.id)
    assert service.list_entries()["total"] == 1
    with pytest.raises(NotFoundError):
        service.get_entry(other.id)


def test_model_pool_default_is_exclusive_and_disabled_entry_unselectable() -> None:
    service = _pool_service(_session_factory())
    first = service.create_entry(
        name="A", provider_key="deepseek", model="m-a", is_default=True
    )
    second = service.create_entry(name="B", provider_key="deepseek", model="m-b")

    service.update_entry(second.id, is_default=True)
    assert service.resolve_entry().id == second.id
    assert service.get_entry(first.id).is_default is False

    # A disabled entry must fail loudly instead of quietly answering with a
    # different model than the one the caller asked for.
    service.update_entry(second.id, enabled=False)
    with pytest.raises(NotFoundError):
        service.resolve_entry(entry_id=second.id)

    with pytest.raises(NotFoundError):
        service.resolve_entry(name="does-not-exist")


def test_model_pool_update_honours_false_but_ignores_illegal_nulls() -> None:
    service = _pool_service(_session_factory())
    entry = service.create_entry(
        name="A",
        provider_key="deepseek",
        model="m-a",
        temperature=0.2,
        max_tokens=1024,
        is_default=True,
        sort_order=5,
    )

    updated = service.update_entry(
        entry.id,
        is_default=False,
        sort_order=0,
        temperature=None,
        max_tokens=None,
    )
    # Explicit False/0 are real values, not "leave unchanged".
    assert updated.is_default is False
    assert updated.sort_order == 0
    # A non-nullable column cannot be blanked by a stray null...
    assert updated.temperature == 0.2
    # ...while the token cap is explicitly nullable.
    assert updated.max_tokens is None


def test_pricing_resolver_prefers_pool_prices_and_falls_through() -> None:
    from app.runtime_stack import build_model_pricing_resolver

    factory = _session_factory()
    service = _pool_service(factory)
    service.create_entry(
        name="Priced",
        provider_key="deepseek",
        model="priced-model",
        input_per_million=1.0,
        output_per_million=2.0,
    )
    service.create_entry(
        name="Unpriced", provider_key="deepseek", model="unpriced-model"
    )

    class _Resources:
        database_session_factory = factory

    resolver = build_model_pricing_resolver(_Resources())
    assert resolver is not None
    assert resolver("priced-model") == ModelPricingSettings(
        input_per_million=1.0, output_per_million=2.0
    )
    # An entry with no prices must not shadow the static [llm].pricing table.
    assert resolver("unpriced-model") is None
    assert resolver("unknown-model") is None

    class _NoDatabase:
        database_session_factory = None

    assert build_model_pricing_resolver(_NoDatabase()) is None


# ---------- LLM streaming + audit ----------


class _Usage:
    def __init__(self, prompt: int, completion: int) -> None:
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = prompt + completion


def _chunk(
    content: str | None,
    *,
    usage: _Usage | None = None,
    finish_reason: str | None = None,
):
    class _Delta:
        def __init__(self, text: str | None) -> None:
            self.content = text

    class _Choice:
        def __init__(self, text: str | None) -> None:
            self.delta = _Delta(text)
            self.finish_reason = finish_reason

    class _Chunk:
        pass

    chunk = _Chunk()
    chunk.choices = [] if content is None else [_Choice(content)]
    chunk.usage = usage
    return chunk


def _request(**overrides) -> LLMRequest:
    base = {
        "model": "deepseek/deepseek-chat",
        "messages": [LLMMessage(role=LLMRole.USER, content="你好")],
        "request_id": "req-chat-1",
        "session_id": "session-chat-1",
        "agent_id": "chat_agent",
        "user_id": "member-chat-1",
    }
    base.update(overrides)
    return LLMRequest(**base)


def test_stream_complete_streams_deltas_and_audits_the_caller() -> None:
    audits: list[LLMCallAudit] = []

    async def _fake_stream(kwargs: dict) -> object:
        async def _generate():
            yield _chunk("你")
            yield _chunk("好")
            # Usage-only trailer, as providers send when include_usage is set.
            yield _chunk(None, usage=_Usage(11, 7))

        return _generate()

    service = LLMService(
        LLMSettings(api_key="sk-real", model="gpt-test"),
        stream_fn=_fake_stream,
        audit_recorder=audits.append,
    )

    deltas: list[str] = []

    async def _consume() -> None:
        async for delta in service.stream_complete(_request()):
            deltas.append(delta)

    asyncio.run(_consume())

    assert "".join(deltas) == "你好"
    assert len(audits) == 1
    audit = audits[0]
    assert audit.status == "success"
    assert audit.model == "deepseek/deepseek-chat"
    assert audit.user_id == "member-chat-1"
    assert audit.session_id == "session-chat-1"
    assert audit.agent_id == "chat_agent"
    assert (audit.prompt_tokens, audit.completion_tokens, audit.total_tokens) == (
        11,
        7,
        18,
    )


def test_stream_complete_reports_the_finish_reason_of_the_final_chunk() -> None:
    # The final chunk is the only one that says *why* the model stopped; without
    # propagating it a streamed answer cut off at max_tokens looks complete.
    async def _fake_stream(kwargs: dict) -> object:
        async def _generate():
            yield _chunk("你")
            yield _chunk("好", finish_reason="length")
            yield _chunk(None, usage=_Usage(11, 7))

        return _generate()

    service = LLMService(
        LLMSettings(api_key="sk-real", model="gpt-test"),
        stream_fn=_fake_stream,
    )

    captured: list[LLMResponse] = []

    async def _consume() -> None:
        async for _ in service.stream_complete(
            _request(), on_complete=captured.append
        ):
            pass

    asyncio.run(_consume())

    assert captured[0].finish_reason == "length"


def test_complete_reports_the_finish_reason_of_the_completion() -> None:
    class _Message:
        content = ""

    class _Choice:
        message = _Message()
        finish_reason = "length"

    class _Raw:
        model = "deepseek/deepseek-v4-flash"
        choices = [_Choice()]
        usage = _Usage(11, 3000)

    async def _fake_completion(kwargs: dict) -> object:
        return _Raw()

    service = LLMService(
        LLMSettings(api_key="sk-real", model="deepseek/deepseek-v4-flash"),
        completion_fn=_fake_completion,
    )

    response = asyncio.run(service.complete(_request()))

    assert response.content == ""
    assert response.finish_reason == "length"


# ---------- ChatService ----------


def _entry(**overrides) -> ModelPoolEntry:
    base = {
        "id": "entry-1",
        "name": "DeepSeek Chat",
        "provider_key": "deepseek",
        "model": "deepseek/deepseek-chat",
        "description": "",
        "temperature": 0.2,
        "max_tokens": None,
        "input_per_million": 0.0,
        "output_per_million": 0.0,
        "enabled": True,
        "is_default": True,
        "sort_order": 0,
    }
    base.update(overrides)
    return ModelPoolEntry(**base)


class _FakeChatLLM:
    """Stands in for LLMService: canned routing reply + canned streamed text."""

    is_configured = True

    def __init__(self, *, routing: str = "not-json", reply: str = "这是直接回答") -> None:
        self.routing = routing
        self.reply = reply
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> object:
        self.requests.append(request)

        class _Response:
            content = ""
            model = "deepseek/deepseek-chat"
            cost_usd = 0.0

        response = _Response()
        response.content = self.routing
        return response

    async def stream_complete(self, request: LLMRequest, *, on_complete=None):
        self.requests.append(request)
        yield self.reply

        class _Usage:
            prompt_tokens = 11
            completion_tokens = 7
            total_tokens = 18

        class _Response:
            model = "deepseek/deepseek-chat"
            cost_usd = 0.000025
            usage = _Usage()

        if on_complete is not None:
            on_complete(_Response())


def _chat_service(factory, *, llm=None, dispatcher=None, agents=None):
    session = factory()
    service = ChatService(
        RuntimeAuditRepository(session),
        MemoryService(MemoryRepository(session)),
        llm_service=llm,
        task_dispatcher=dispatcher,
        agent_route_provider=(lambda: agents) if agents is not None else (lambda: {}),
    )
    return service, MemoryService(MemoryRepository(session))


async def _collect(iterator) -> list[dict]:
    return [event async for event in iterator]


def test_chat_session_is_owned_by_its_creator() -> None:
    factory = _session_factory()
    service, _ = _chat_service(factory)
    session = service.create_session(user_id="member-1")

    assert service.ensure_session(session_id=session.session_id, user_id="member-1")
    with pytest.raises(NotFoundError):
        service.ensure_session(session_id=session.session_id, user_id="member-2")
    with pytest.raises(NotFoundError):
        service.ensure_session(session_id="no-such-session", user_id="member-1")


def test_decide_without_specialists_always_answers_directly() -> None:
    factory = _session_factory()
    service, _ = _chat_service(factory, llm=None, agents={})
    session = service.create_session(user_id="member-1")

    decision = asyncio.run(
        service.decide(
            content="帮我写一篇公众号文章",
            request_id="req-1",
            session_id=session.session_id,
            user_id="member-1",
            model_entry=_entry(),
        )
    )
    assert decision.mode == "chat"
    assert decision.agent_id is None


@pytest.mark.parametrize(
    ("routing", "expected_mode", "expected_agent"),
    [
        # Unparsable replies and hallucinated agent ids must never dispatch.
        ("not-json", "chat", None),
        ('{"mode": "task", "agent_id": "ghost_agent"}', "chat", None),
        ('{"mode": "chat"}', "chat", None),
        (
            '{"mode": "task", "agent_id": "article_agent", "reason": "needs a '
            'published article"}',
            "task",
            "article_agent",
        ),
    ],
)
def test_decide_maps_model_reply_onto_a_known_agent(
    routing: str, expected_mode: str, expected_agent: str | None
) -> None:
    factory = _session_factory()
    llm = _FakeChatLLM(routing=routing)
    service, _ = _chat_service(
        factory, llm=llm, agents={"article_agent": "生成并发布公众号文章"}
    )
    session = service.create_session(user_id="member-1")

    decision = asyncio.run(
        service.decide(
            content="帮我写一篇公众号文章",
            request_id="req-1",
            session_id=session.session_id,
            user_id="member-1",
            model_entry=_entry(),
        )
    )
    assert decision.mode == expected_mode
    assert decision.agent_id == expected_agent
    # The classifier's candidate list is the routable-agent directory, so a
    # newly registered vertical shows up in the prompt without code changes.
    assert "article_agent: 生成并发布公众号文章" in llm.requests[0].messages[0].content


def test_handle_turn_streams_plain_conversation_and_records_transcript() -> None:
    factory = _session_factory()
    llm = _FakeChatLLM(routing="not-json", reply="你好，我可以帮你写文章。")
    service, memory = _chat_service(factory, llm=llm, agents={"article_agent": "写文章"})
    session = service.create_session(user_id="member-1")

    events = asyncio.run(
        _collect(
            service.handle_turn(
                session=session,
                content="你好",
                user_id="member-1",
                model_entry=_entry(),
            )
        )
    )

    assert [event["type"] for event in events] == ["status", "delta", "done"]
    assert events[1]["content"] == "你好，我可以帮你写文章。"
    assert events[-1]["mode"] == "chat"

    turns = memory.list_session_messages(session.session_id)
    assert [turn.role for turn in turns] == ["user", "assistant"]
    assert turns[0].content == "你好"
    assert turns[1].content == "你好，我可以帮你写文章。"
    # The session is named after its first user turn.
    assert service.get_session(session.session_id).metadata_payload["title"] == "你好"


def test_handle_turn_hands_a_vertical_task_to_the_async_queue() -> None:
    factory = _session_factory()
    llm = _FakeChatLLM(
        routing=json.dumps(
            {
                "mode": "task",
                "agent_id": "article_agent",
                "reason": "需要发布一篇公众号文章",
            }
        )
    )
    dispatched: dict = {}

    def _dispatcher(task_name: str, payload: dict) -> DispatchedTask:
        dispatched["task_name"] = task_name
        dispatched["payload"] = payload
        return DispatchedTask(task_id="task-1", status="pending")

    service, memory = _chat_service(
        factory,
        llm=llm,
        dispatcher=_dispatcher,
        agents={"article_agent": "生成并发布公众号文章"},
    )
    session = service.create_session(user_id="member-1")

    events = asyncio.run(
        _collect(
            service.handle_turn(
                session=session,
                content="帮我写一篇公众号文章",
                user_id="member-1",
                model_entry=_entry(),
            )
        )
    )

    assert [event["type"] for event in events] == ["status", "task", "done"]
    task_event = events[1]
    assert task_event["agent_id"] == "article_agent"
    assert task_event["task_id"] == "task-1"
    assert events[-1]["mode"] == "task"

    assert dispatched["task_name"] == "workflow.run"
    payload = dispatched["payload"]
    # The hand-off payload must be a valid workflow request, because the real
    # dispatcher validates it before it reaches Celery.
    workflow_request = WorkflowRunRequest.model_validate(payload)
    assert workflow_request.session_id == session.session_id
    assert workflow_request.request_id == task_event["request_id"]
    assert workflow_request.user_id == "member-1"
    assert workflow_request.structured_input["requested_agent"] == "article_agent"
    assert workflow_request.structured_input["source"] == "chat"
    # The model chosen in chat drives the whole run.
    assert workflow_request.llm_profile is not None
    assert workflow_request.llm_profile.provider == "deepseek"
    assert workflow_request.llm_profile.model == "deepseek/deepseek-chat"

    turns = memory.list_session_messages(session.session_id)
    assert [turn.role for turn in turns] == ["user", "assistant"]
    assert turns[1].metadata_payload["task_id"] == "task-1"
    assert turns[1].metadata_payload["agent_id"] == "article_agent"
    # The orchestrator skips re-appending the user turn this request already has.
    assert memory.has_request_turn(
        session_id=session.session_id, request_id=turns[0].request_id
    )


def test_handle_turn_requires_an_async_transport() -> None:
    from app.core import ResourceNotReadyError

    factory = _session_factory()
    llm = _FakeChatLLM(
        routing='{"mode": "task", "agent_id": "article_agent", "reason": "x"}'
    )
    service, _ = _chat_service(
        factory, llm=llm, dispatcher=None, agents={"article_agent": "写文章"}
    )
    session = service.create_session(user_id="member-1")

    with pytest.raises(ResourceNotReadyError):
        asyncio.run(
            _collect(
                service.handle_turn(
                    session=session,
                    content="写一篇公众号文章",
                    user_id="member-1",
                    model_entry=_entry(),
                )
            )
        )


# ---------- usage report ----------


def _seed_llm_call(
    record_service: RuntimeRecordService,
    *,
    user_id: str,
    model: str,
    prompt: int,
    completion: int,
    status: str = "success",
) -> None:
    record_service.save_llm_call_log(
        LLMCallAudit(
            request_id=f"req-{uuid.uuid4().hex[:8]}",
            user_id=user_id,
            model=model,
            status=status,
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
        )
    )


def test_llm_usage_report_rolls_up_by_user_and_model() -> None:
    factory = _session_factory()
    session = factory()
    repository = RuntimeAuditRepository(session)
    record_service = RuntimeRecordService(repository)

    _seed_llm_call(
        record_service, user_id="member-1", model="deepseek/deepseek-chat",
        prompt=10, completion=5,
    )
    _seed_llm_call(
        record_service, user_id="member-1", model="zhipu/glm-4",
        prompt=4, completion=0, status="failed",
    )
    _seed_llm_call(
        record_service, user_id="member-2", model="deepseek/deepseek-chat",
        prompt=1, completion=1,
    )

    query_service = RuntimeQueryService(repository)
    report = query_service.get_llm_usage_report()

    assert report["totals"]["call_count"] == 3
    assert report["totals"]["failed_call_count"] == 1
    assert report["totals"]["total_tokens"] == 21
    assert report["totals"]["prompt_tokens"] == 15
    assert report["totals"]["completion_tokens"] == 6

    assert {item["key"] for item in report["by_model"]} == {
        "deepseek/deepseek-chat",
        "zhipu/glm-4",
    }
    assert {item["key"] for item in report["by_user"]} == {"member-1", "member-2"}
    cross = {
        (item["user_id"], item["model"]): item["total_tokens"]
        for item in report["by_user_model"]
    }
    assert cross[("member-1", "deepseek/deepseek-chat")] == 15
    assert cross[("member-1", "zhipu/glm-4")] == 4
    assert cross[("member-2", "deepseek/deepseek-chat")] == 2

    # Members are scoped to their own rows by the route, which passes user_id.
    scoped = query_service.get_llm_usage_report(user_id="member-1")
    assert scoped["totals"]["total_tokens"] == 19
    assert {item["key"] for item in scoped["by_user"]} == {"member-1"}
    assert scoped["filters"]["user_id"] == "member-1"

    by_model = query_service.get_llm_usage_report(model="zhipu/glm-4")
    assert by_model["totals"]["total_tokens"] == 4
    assert by_model["totals"]["failed_call_count"] == 1
    assert by_model["filters"]["model"] == "zhipu/glm-4"


# ---------- HTTP surface ----------


def _chat_stream_chunk(content: str | None, *, usage: _Usage | None = None):
    return _chunk(content, usage=usage)


def test_model_pool_endpoints_scope_members_to_enabled_entries(
    api_client: TestClient, as_user
) -> None:
    name = f"测试模型-{uuid.uuid4().hex[:6]}"
    created = api_client.post(
        "/api/v1/models",
        json={
            "name": name,
            "provider_key": "deepseek",
            "model": "deepseek/deepseek-v4-flash",
            "enabled": False,
            "is_default": False,
        },
    )
    assert created.status_code == 201
    payload = created.json()
    entry_id = payload["id"]
    # Credentials never travel with the pool: provider_key names a config block.
    assert "api_key" not in payload
    assert payload["provider_key"] == "deepseek"

    try:
        as_user(user_id=f"pool-member-{uuid.uuid4().hex[:8]}", username="member")
        hidden = api_client.get("/api/v1/models").json()["items"]
        assert entry_id not in {item["id"] for item in hidden}

        as_user(user_id="test-admin-id", username="test-admin", role="admin")
        enabled = api_client.patch(
            f"/api/v1/models/{entry_id}",
            json={"enabled": True, "sort_order": 3, "temperature": 0.7},
        )
        assert enabled.status_code == 200
        assert enabled.json()["enabled"] is True
        assert enabled.json()["sort_order"] == 3
        assert enabled.json()["temperature"] == 0.7

        as_user(user_id=f"pool-member-{uuid.uuid4().hex[:8]}", username="member")
        visible = api_client.get("/api/v1/models").json()["items"]
        assert entry_id in {item["id"] for item in visible}
        # Members may read but never curate.
        assert (
            api_client.patch(
                f"/api/v1/models/{entry_id}", json={"temperature": 0.1}
            ).status_code
            == 403
        )
        assert api_client.delete(f"/api/v1/models/{entry_id}").status_code == 403
    finally:
        as_user(user_id="test-admin-id", username="test-admin", role="admin")
        deleted = api_client.delete(f"/api/v1/models/{entry_id}")
        assert deleted.status_code == 200
        assert deleted.json()["deleted"] is True
        remaining = api_client.get("/api/v1/models").json()["items"]
        assert entry_id not in {item["id"] for item in remaining}


def test_chat_sessions_are_scoped_to_their_owner(
    api_client: TestClient, as_user
) -> None:
    owner = as_user(user_id=f"chat-owner-{uuid.uuid4().hex[:8]}", username="owner")
    created = api_client.post("/api/v1/chat/sessions", json={"title": "我的会话"})
    assert created.status_code == 201
    session_id = created.json()["session_id"]
    assert created.json()["user_id"] == owner.id

    mine = api_client.get("/api/v1/chat/sessions").json()["items"]
    assert [item["session_id"] for item in mine] == [session_id]

    as_user(user_id=f"chat-other-{uuid.uuid4().hex[:8]}", username="other")
    assert api_client.get("/api/v1/chat/sessions").json()["items"] == []
    assert (
        api_client.get(f"/api/v1/chat/sessions/{session_id}/messages").status_code
        == 404
    )
    assert api_client.delete(f"/api/v1/chat/sessions/{session_id}").status_code == 404

    as_user(user_id=owner.id, username="owner")
    deleted = api_client.delete(f"/api/v1/chat/sessions/{session_id}")
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True
    assert api_client.get("/api/v1/chat/sessions").json()["items"] == []


def test_console_chat_turn_streams_and_bills_the_asking_member(
    api_client: TestClient, as_user, monkeypatch
) -> None:
    created = api_client.post(
        "/api/v1/models",
        json={
            "name": f"对话模型-{uuid.uuid4().hex[:6]}",
            "provider_key": "deepseek",
            "model": "deepseek/deepseek-v4-flash",
            "input_per_million": 1.0,
            "output_per_million": 2.0,
            "is_default": True,
        },
    )
    assert created.status_code == 201
    entry_id = created.json()["id"]

    async def _fake_call_stream(self, kwargs: dict) -> object:
        async def _generate():
            yield _chat_stream_chunk("你好")
            yield _chat_stream_chunk("，这是回答。")
            yield _chat_stream_chunk(None, usage=_Usage(11, 7))

        return _generate()

    monkeypatch.setattr(LLMService, "_call_stream", _fake_call_stream)

    member = as_user(
        user_id=f"chat-member-{uuid.uuid4().hex[:8]}", username="chat-member"
    )
    session_id = ""
    try:
        response = api_client.post(
            "/api/v1/chat/turns",
            json={"content": "你好", "model_entry_id": entry_id},
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = response.text
        assert '"type": "delta"' in body
        assert "，这是回答。" in body
        assert '"type": "done"' in body
        assert '"mode": "chat"' in body

        transcript = api_client.get("/api/v1/chat/sessions").json()["items"]
        assert len(transcript) == 1
        session_id = transcript[0]["session_id"]
        messages = api_client.get(
            f"/api/v1/chat/sessions/{session_id}/messages"
        ).json()["items"]
        assert [item["role"] for item in messages] == ["user", "assistant"]
        assert messages[1]["content"] == "你好，这是回答。"

        # The streamed call is billed to the member who asked, on the model
        # the pool priced -- which is what the usage report shows.
        usage = api_client.get("/api/v1/runtime/usage/llm").json()
        assert usage["filters"]["user_id"] == member.id
        # Two audited calls per turn: the routing classification and the answer.
        assert usage["totals"]["call_count"] == 2
        assert usage["totals"]["total_tokens"] >= 18
        assert usage["totals"]["cost_usd"] > 0
        cross = {item["model"]: item for item in usage["by_user_model"]}
        assert cross["deepseek/deepseek-v4-flash"]["total_tokens"] >= 18
        assert cross["deepseek/deepseek-v4-flash"]["user_id"] == member.id
    finally:
        if session_id:
            api_client.delete(f"/api/v1/chat/sessions/{session_id}")
        as_user(user_id="test-admin-id", username="test-admin", role="admin")
        api_client.delete(f"/api/v1/models/{entry_id}")
