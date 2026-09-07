"""Minimal in-memory knowledge retriever.

Keyword-overlap scoring over documents provided through configuration.
This is the P1 baseline backend: no external vector store required, yet
the full retrieve -> rerank -> cite chain is real and swappable later.
"""
from __future__ import annotations

import re

from app.knowledge.base import BaseKnowledgeRetriever, KnowledgeHit

_TOKEN_PATTERN = re.compile(r"[\w]+", re.UNICODE)


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_PATTERN.findall(text or "")}


class InMemoryKnowledgeRetriever(BaseKnowledgeRetriever):
    """Score documents by weighted token overlap with the query."""

    name = "inmemory"

    def __init__(
        self,
        documents: list[dict] | None = None,
        *,
        min_score: float = 0.05,
    ) -> None:
        self._documents: list[dict] = list(documents or [])
        self._min_score = min_score

    async def retrieve(self, query: str, *, top_k: int = 5) -> list[KnowledgeHit]:
        query_tokens = _tokenize(query)
        if not query_tokens or not self._documents:
            return []
        scored: list[tuple[float, dict]] = []
        for document in self._documents:
            content = str(document.get("content", ""))
            document_tokens = _tokenize(
                f"{document.get('title', '')} {content} "
                f"{' '.join(document.get('tags', []) or [])}"
            )
            if not document_tokens:
                continue
            overlap = query_tokens & document_tokens
            if not overlap:
                continue
            score = len(overlap) / len(query_tokens)
            if score >= self._min_score:
                scored.append((score, document))
        scored.sort(key=lambda item: (-item[0], str(item[1].get("id", ""))))

        hits: list[KnowledgeHit] = []
        for score, document in scored[: max(top_k, 0)]:
            hits.append(
                KnowledgeHit(
                    id=str(document.get("id", "")),
                    title=str(document.get("title", "")),
                    content=str(document.get("content", "")),
                    score=round(min(score, 1.0), 4),
                    source=str(document.get("source", "builtin")),
                    metadata_payload=dict(document.get("metadata_payload", {}) or {}),
                )
            )
        return self.rerank(query, hits)
