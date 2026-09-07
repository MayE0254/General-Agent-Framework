"""Unit tests for the knowledge retrieval abstraction and tool wiring."""

import asyncio

import pytest

from app.knowledge import (
    BaseKnowledgeRetriever,
    InMemoryKnowledgeRetriever,
    KnowledgeHit,
)
from app.tools.bootstrap import create_default_tool_registry
from app.tools.system import KnowledgeSearchTool

DOCS = [
    {
        "id": "kb-001",
        "title": "Platform overview",
        "content": "multi-agent platform with orchestrator and tools",
        "tags": ["platform"],
    },
    {
        "id": "kb-002",
        "title": "Memory guide",
        "content": "session memory and durable memory scopes",
        "tags": ["memory"],
    },
    {
        "id": "kb-003",
        "title": "Deployment",
        "content": "docker compose deployment for the platform",
        "tags": ["deploy"],
    },
]


def test_retriever_ranks_by_token_overlap() -> None:
    retriever = InMemoryKnowledgeRetriever(DOCS)

    hits = asyncio.run(retriever.retrieve("multi-agent platform", top_k=3))

    assert hits, "expected at least one hit for an overlapping query"
    assert hits[0].id == "kb-001"
    assert hits[0].score >= hits[-1].score
    assert all(0.0 <= hit.score <= 1.0 for hit in hits)


def test_retriever_respects_top_k_and_min_score() -> None:
    # Partial overlap (1/3 query tokens) survives the default threshold...
    partial_hits = asyncio.run(
        InMemoryKnowledgeRetriever(DOCS).retrieve(
            "platform blockchain quantum", top_k=5
        )
    )
    assert partial_hits
    assert all(hit.id in {"kb-001", "kb-003"} for hit in partial_hits)

    # ...but is filtered out by a strict threshold. Exact matches (score 1.0)
    # still pass, so the query must be a partial-overlap one.
    strict = InMemoryKnowledgeRetriever(DOCS, min_score=0.9)
    assert (
        asyncio.run(strict.retrieve("platform blockchain quantum", top_k=5)) == []
    )

    limited = InMemoryKnowledgeRetriever(DOCS)
    hits = asyncio.run(limited.retrieve("platform deployment docker", top_k=1))
    assert len(hits) == 1
    assert hits[0].id == "kb-003"


def test_retriever_handles_empty_inputs() -> None:
    retriever = InMemoryKnowledgeRetriever([])
    assert asyncio.run(retriever.retrieve("anything")) == []

    empty_query = InMemoryKnowledgeRetriever(DOCS)
    assert asyncio.run(empty_query.retrieve("", top_k=3)) == []


def test_retriever_rerank_hook_keeps_order_by_default() -> None:
    retriever = InMemoryKnowledgeRetriever(DOCS)
    hits = asyncio.run(retriever.retrieve("platform", top_k=2))
    reranked = retriever.rerank("platform", hits)
    assert [hit.id for hit in reranked] == [hit.id for hit in hits]


def test_knowledge_hit_citation_ref() -> None:
    hit = KnowledgeHit(id="kb-9", content="text")
    assert hit.citation_ref("inmemory") == "inmemory:kb-9"


def test_knowledge_search_tool_uses_configured_documents() -> None:
    tool = KnowledgeSearchTool(
        config={
            "documents": DOCS,
            "top_k": 2,
            "min_score": 0.05,
        }
    )
    result = asyncio.run(
        tool._run(context=None, arguments={"query": "multi-agent platform"})
    )

    assert result.status.value == "success"
    assert result.content["backend"] == "inmemory"
    assert 1 <= result.content["hits"] <= 2
    assert result.content["results"]
    assert result.content["results"][0]["id"] == "kb-001"


def test_knowledge_search_tool_without_documents_returns_zero_hits() -> None:
    tool = KnowledgeSearchTool()
    result = asyncio.run(tool._run(context=None, arguments={"query": "anything"}))

    assert result.status.value == "success"
    assert result.content["hits"] == 0
    assert result.content["results"] == []


def test_tool_registry_binds_knowledge_settings() -> None:
    from app.core.settings import KnowledgeDocument, KnowledgeSettings

    settings = KnowledgeSettings(
        documents=[
            KnowledgeDocument(id="kb-1", content="enterprise agents platform")
        ]
    )
    registry = create_default_tool_registry(knowledge_settings=settings)
    tool = registry.create("knowledge_search")

    result = asyncio.run(
        tool._run(context=None, arguments={"query": "enterprise agents"})
    )

    assert result.content["hits"] == 1
    assert result.content["results"][0]["id"] == "kb-1"


def test_retriever_is_swap_for_base_protocol() -> None:
    assert issubclass(InMemoryKnowledgeRetriever, BaseKnowledgeRetriever)
