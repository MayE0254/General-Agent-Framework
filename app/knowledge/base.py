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
    # Relevance signal varies by backend (vector=cosine in [0,1], BM25 is
    # unbounded, cross-encoder rerank are logits).  Keep it non-negative but
    # do NOT clamp to [0,1]; callers wanting a probability should normalize.
    score: float = Field(default=0.0, ge=0.0)
    source: str = "builtin"
    # Parent/child indexing.  ``id`` stays the addressable child unit (that is
    # what exists in the vector store / BM25 snapshot), while ``parent_id``
    # names the enclosing clause ("第X条") and ``parent_text`` carries its full
    # text when the child is only a fragment of it.  Callers wanting whole-
    # clause context use ``expand_to_parent`` (app.knowledge.diversity) rather
    # than reading these fields directly, so the "is there a parent?" rule
    # lives in exactly one place.
    parent_id: str = ""
    parent_text: str = ""
    metadata_payload: dict[str, Any] = Field(default_factory=dict)

    def citation_ref(self, retriever_name: str) -> str:
        """Materialize this hit as a knowledge ref for shared state."""
        return f"{retriever_name}:{self.id}"


# ``metadata_payload`` key marking a hit the per-document quota rejected and the
# retriever is handing back as *supplementary* evidence (see
# ``diversity.cap_per_document``).  Such a hit passed both recall legs -- the
# retriever did judge it relevant -- but never reached the cross-encoder, so it
# is deliberately ranked below the primary results and must not be presented as
# equivalent evidence.
QUOTA_OVERFLOW_KEY = "quota_overflow"


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
