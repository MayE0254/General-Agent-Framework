"""Cross-encoder reranker (BGE-reranker) on top of BGE RAG output.

Reranks a limited candidate list (top-k from the hybrid retrieval) by a
query-passage cross-encoder, so the final ordering reflects fine-grained
relevance rather than raw vector cosine or BM25.  Kept lazy — the model is
loaded only on first use, which matters for low-resource machines.
"""
from __future__ import annotations

from app.knowledge.base import KnowledgeHit


class CrossEncoderReranker:
    """Rerank KnowledgeHit lists with a local BGE cross-encoder."""

    def __init__(self, model_name: str, *, max_length: int = 512) -> None:
        self.model_name = model_name
        self._model = None
        self._max_length = max_length

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, max_length=self._max_length)
        return self._model

    def rerank(self, query: str, hits: list[KnowledgeHit]) -> list[KnowledgeHit]:
        """Return hits reordered by cross-encoder relevance (descending);

        assigns a ``rerank_score`` into each hit's metadata_payload.

        The re-scored hit is *derived* with ``model_copy`` rather than rebuilt
        field by field: this stage runs just before ``expand_to_parent``, so a
        hand-written field list that forgets ``parent_id`` / ``parent_text``
        silently disables clause expansion for every returned hit.
        """
        if not hits:
            return []
        # BGE cross-encoder expects a raw (query, passage) pair; the
        # "为这个句子生成表示…" instruction belongs to the embedding model only.
        pairs = [
            (query, (hit.title or "") + "\n" + hit.content)
            for hit in hits
        ]
        model = self._get_model()
        scores = model.predict(pairs)
        indexed = list(zip(hits, scores))
        indexed.sort(key=lambda item: -float(item[1]))
        result: list[KnowledgeHit] = []
        for hit, score in indexed:
            result.append(
                hit.model_copy(
                    update={
                        "score": float(score),
                        "metadata_payload": {
                            **hit.metadata_payload,
                            "rerank_score": float(score),
                        },
                    }
                )
            )
        return result