"""Single-hop console integration, with no model or Milvus network calls."""
import asyncio
import json
import threading
import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core import ResourceNotReadyError
from app.core.settings import KnowledgeSettings
from app.knowledge.base import KnowledgeHit, QUOTA_OVERFLOW_KEY
from app.knowledge.factory import RuntimeKnowledgeRetriever, build_single_hop_retriever
from app.knowledge.rag import KnowledgeAnswerer, INSUFFICIENT_EVIDENCE_MARKER
from app.llm import LLMResponse, LLMUsage
from app.models import Base, ModelPoolEntry
from app.repositories import MemoryRepository, RuntimeAuditRepository
from app.services import ChatService, MemoryService
from app.tools.bootstrap import create_default_tool_registry


class Retriever:
    name = "hybrid"

    def __init__(self, hits=None, error=None):
        self.hits = hits if hits is not None else [
            KnowledgeHit(
                id="clause-1", title="测试法规", content="第一条 测试依据。",
                source="test.docx", metadata_payload={"section": "第一条"},
            )
        ]
        self.queries = []
        self.error = error

    async def retrieve(self, query, *, top_k=5):
        self.queries.append((query, top_k))
        if self.error:
            raise self.error
        return self.hits


class LLM:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        return LLMResponse(
            model=request.model, content=next(self.replies),
            usage=LLMUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            cost_usd=0.01, latency_ms=10,
        )

    async def stream_complete(self, request, **kwargs):
        raise AssertionError("RAG must not fall back to ordinary chat")
        yield ""


def make_service(retriever, llm):
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)()
    memory = MemoryService(MemoryRepository(session))
    settings = KnowledgeSettings(
        retriever="hybrid",
        chat={"enabled": True, "provider_extra_body": {
            "deepseek": {"thinking": {"type": "disabled"}}
        }},
    )
    service = ChatService(
        RuntimeAuditRepository(session), memory, llm_service=llm,
        knowledge_retriever=retriever, knowledge_settings=settings,
        agent_route_provider=lambda: {},
    )
    return service, memory


def entry(provider="deepseek"):
    return ModelPoolEntry(
        id="model-1", name="Selected", provider_key=provider,
        model=f"{provider}/selected", temperature=0.5, max_tokens=None,
    )


def run_turn(service, content="税率是多少？", session=None):
    session = session or service.create_session(user_id="member")

    async def collect():
        return [event async for event in service.handle_turn(
            session=session, content=content, user_id="member", model_entry=entry()
        )]

    return session, asyncio.run(collect())


def test_rag_turn_uses_selected_model_and_persists_evidence():
    retriever = Retriever()
    llm = LLM(['{"mode":"rag"}', "结论依据第一条。[1]"])
    service, memory = make_service(retriever, llm)
    session, events = run_turn(service)
    assert [e["type"] for e in events] == [
        "status", "status", "delta", "evidence", "done"
    ]
    evidence = events[-2]
    assert evidence["backend"] == "hybrid"
    assert evidence["graph_enabled"] is False
    assert evidence["contexts"] == ["第一条 测试依据。"]
    assert evidence["citations"][0]["chunk_id"] == "clause-1"
    assert retriever.queries == [("税率是多少？", 5)]
    turns = memory.list_session_messages(session.session_id)
    assert len(turns) == 2
    assert turns[-1].metadata_payload["citations"] == evidence["citations"]
    request = llm.requests[-1]
    assert request.model == "deepseek/selected"
    assert request.provider == "deepseek"
    assert request.user_id == "member"
    assert request.session_id == session.session_id
    assert request.request_id == events[-1]["request_id"]
    assert request.extra_body == {"thinking": {"type": "disabled"}}


def test_supplementary_retry_only_publishes_final_answer_and_sums_usage():
    retriever = Retriever([
        KnowledgeHit(id="p", content="主证据。"),
        KnowledgeHit(id="s", content="补充依据。",
                     metadata_payload={QUOTA_OVERFLOW_KEY: True}),
    ])
    llm = LLM(['{"mode":"rag"}', INSUFFICIENT_EVIDENCE_MARKER, "最终结论。[2]"])
    service, _ = make_service(retriever, llm)
    _, events = run_turn(service)
    assert [e["content"] for e in events if e["type"] == "delta"] == ["最终结论。[2]"]
    assert events[-1]["total_tokens"] == 30
    assert events[-1]["cost_usd"] == 0.02
    assert events[-2]["citations"][1]["supplementary"] is True
    assert len(llm.requests) == 3
    assert llm.requests[1].request_id == llm.requests[2].request_id


def test_followup_retrieves_resolved_question_without_reusing_old_answer_as_evidence():
    llm = LLM([
        '{"mode":"rag","retrieval_query":"企业所得税法的法定税率是多少？"}',
        "根据当前检索依据回答。[1]",
    ])
    retriever = Retriever()
    service, memory = make_service(retriever, llm)
    session = service.create_session(user_id="member")
    memory.append_session_messages(
        session_id=session.session_id, request_id="previous",
        turns=[{"role": "user", "content": "企业所得税法",
                "step_index": 1, "metadata_payload": {}},
               {"role": "assistant", "content": "旧回答不是法规证据",
                "step_index": 2, "metadata_payload": {}}],
    )
    run_turn(service, "它的法定税率呢？", session)
    assert retriever.queries[0][0] == "企业所得税法的法定税率是多少？"
    assert "旧回答不是法规证据" not in llm.requests[-1].messages[-1].content
    assert "它的法定税率呢？" in llm.requests[-1].messages[-1].content


@pytest.mark.parametrize("routing", ["not-json", "[]", '{"mode":"task","agent_id":"ghost"}'])
def test_bad_routing_defaults_to_retrieval_not_ungrounded_chat(routing):
    retriever = Retriever()
    service, _ = make_service(retriever, LLM([routing, "有据回答。[1]"]))
    _, events = run_turn(service)
    assert events[-1]["mode"] == "rag"
    assert retriever.queries


def test_empty_retrieval_refuses_without_generation():
    llm = LLM(['{"mode":"rag"}'])
    service, _ = make_service(Retriever([]), llm)
    _, events = run_turn(service)
    assert events[-2]["insufficient_evidence"] is True
    assert events[-2]["citations"] == []
    assert len(llm.requests) == 1


def test_retrieval_failure_does_not_generate_or_persist_an_answer():
    llm = LLM(['{"mode":"rag"}'])
    service, memory = make_service(
        Retriever(error=ResourceNotReadyError("Milvus unavailable")), llm
    )
    session = service.create_session(user_id="member")
    with pytest.raises(ResourceNotReadyError):
        run_turn(service, session=session)
    assert len(llm.requests) == 1
    assert [r.role for r in memory.list_session_messages(session.session_id)] == ["user"]


def test_answer_contexts_match_truncated_prompt_and_provider_options_are_scoped():
    llm = LLM(["回答。[1]"])
    answerer = KnowledgeAnswerer(
        Retriever([KnowledgeHit(id="long", content="正文" * 1500)]),
        llm, model="zhipu/selected", provider="zhipu",
        user_id="member", session_id="session", request_id="request",
    )
    answer = asyncio.run(answerer.answer("问题"))
    assert len(answer.contexts[0]) == 2000
    assert answer.contexts[0] in llm.requests[0].messages[-1].content
    assert llm.requests[0].extra_body is None


def test_factory_ignores_graph_flags_and_preserves_baseline_parameters(monkeypatch):
    import app.knowledge.factory as module

    class Store:
        def __init__(self, settings):
            self.verified = False
        def verify_schema(self):
            self.verified = True
        def close(self):
            pass

    monkeypatch.setattr(module, "MilvusStore", Store)
    settings = KnowledgeSettings(
        hybrid_candidates=50, quota_overflow_top_n=3,
        graph={"expand": {"enabled": True, "use_cites": True, "use_mentions": True}},
    )
    retriever = build_single_hop_retriever(settings, chunks=Retriever().hits)
    assert retriever.graph_expander is None
    assert retriever.allow_vector_fallback is False
    assert retriever.store.verified
    assert retriever.candidates == 50
    assert retriever.quota_overflow_top_n == 3
    assert retriever.max_chunks_per_document == 3


def test_lazy_runtime_reuses_models_off_the_event_loop_and_closes():
    created = []
    thread_ids = []
    closed = []
    retriever = Retriever()
    retriever.store = SimpleNamespace(close=lambda: closed.append(True))

    def build(settings):
        created.append(True)
        thread_ids.append(threading.get_ident())
        return retriever

    runtime = RuntimeKnowledgeRetriever(KnowledgeSettings(), builder=build)
    assert created == []
    asyncio.run(runtime.retrieve("一"))
    asyncio.run(runtime.retrieve("二"))
    assert created == [True]
    assert thread_ids[0] != threading.get_ident()
    runtime.close()
    runtime.close()
    assert closed == [True]
    with pytest.raises(ResourceNotReadyError):
        asyncio.run(runtime.retrieve("三"))


def test_tools_share_the_runtime_retriever():
    retriever = Retriever()
    registry = create_default_tool_registry(
        knowledge_settings=KnowledgeSettings(retriever="hybrid"),
        knowledge_retriever=retriever,
    )
    result = asyncio.run(registry.create("knowledge_search")._run(
        None, {"query": "税率"}
    ))
    assert result.content["backend"] == "hybrid"
    assert result.content["results"][0]["id"] == "clause-1"
    assert retriever.queries


def test_console_rag_http_persists_evidence_and_bills_user(api_client, as_user, monkeypatch):
    from app.main import app

    created = api_client.post("/api/v1/models", json={
        "name": f"RAG-{uuid.uuid4().hex[:8]}", "provider_key": "deepseek",
        "model": "deepseek/deepseek-v4-flash", "is_default": True,
    })
    assert created.status_code == 201
    model_id = created.json()["id"]
    monkeypatch.setattr(app.state.settings.knowledge.chat, "enabled", True)
    monkeypatch.setattr(app.state, "knowledge_retriever", Retriever())
    replies = iter(['{"mode":"rag"}', "依据条款回答。[1]"])

    async def complete(kwargs):
        return SimpleNamespace(
            model=kwargs["model"],
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=next(replies)), finish_reason="stop"
            )],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

    monkeypatch.setattr(app.state.llm_service, "_completion_fn", complete)
    user = as_user(user_id=f"rag-{uuid.uuid4().hex[:8]}")
    response = api_client.post("/api/v1/chat/turns", json={
        "content": "税率是多少？", "model_entry_id": model_id,
    })
    assert response.status_code == 200
    events = [json.loads(line[6:]) for line in response.text.splitlines()
              if line.startswith("data: ")]
    assert events[-1]["mode"] == "rag"
    sid = events[-1]["session_id"]
    messages = api_client.get(f"/api/v1/chat/sessions/{sid}/messages").json()["items"]
    assert messages[-1]["metadata_payload"]["graph_enabled"] is False
    assert messages[-1]["metadata_payload"]["citations"][0]["chunk_id"] == "clause-1"
    usage = api_client.get("/api/v1/runtime/usage/llm").json()
    assert usage["filters"]["user_id"] == user.id
    assert usage["totals"]["call_count"] == 2
    assert usage["totals"]["total_tokens"] == 30
