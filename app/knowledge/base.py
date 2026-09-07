"""Unified knowledge retrieval abstraction.

The base protocol keeps the orchestration trunk independent from any
specific backend (in-memory keyword store today, pgvector/Milvus later).
Retrieval results carry explicit provenance so agent outputs can cite
where every piece of knowledge came from.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class KnowledgeHit(BaseModel):
    """One retrieved knowledge chunk with citation-ready provenance."""

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str = ""
    content: str
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    source: str = "builtin"
    metadata_payload: dict[str, Any] = Field(default_factory=dict)

    def citation_ref(self, retriever_name: str) -> str:
        """Materialize this hit as a knowledge ref for shared state."""
        return f"{retriever_name}:{self.id}"


class BaseKnowledgeRetriever(ABC):
    """Contract every knowledge backend must fulfill.

    Implementations provide ``retrieve``; ``rerank`` is an optional hook
    reserved for future cross-encoder / business-rule reranking so callers
    can already route hits through it today.
    """

    name: str = "base"

    @abstractmethod
    async def retrieve(self, query: str, *, top_k: int = 5) -> list[KnowledgeHit]:
        """Return the top-k hits for a query, best match first."""

    def rerank(
        self,
        query: str,
        hits: list[KnowledgeHit],
    ) -> list[KnowledgeHit]:
        """Optional reranking hook; default keeps retrieval order."""
        return hits
