"""Local embedding engine backed by sentence-transformers (BGE-zh).

Loads ``BAAI/bge-base-zh-v1.5`` (768-d).  For Chinese retrieval we follow the
publisher's guidance and add BGE-zh retrieval instruction ORP to the query
side for better recall — instruction text is only applied to ST queries, not
to the corpus documents.
"""
from __future__ import annotations

from typing import Sequence

_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


class LocalEmbeddingModel:
    """Thin lazy wrapper around a sentence-transformers embedding model."""

    _BGE_INSTRUCTION = _INSTRUCTION

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        return self._get_model().get_sentence_embedding_dimension()

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed corpus chunks (no instruction prefix)."""
        model = self._get_model()
        return [v.tolist() for v in model.encode(list(texts), normalize_embeddings=True)]

    def embed_query(self, query: str) -> list[float]:
        """Embed a search query with BGE's retrieval instruction."""
        model = self._get_model()
        return (
            model.encode([f"{self._BGE_INSTRUCTION}{query}"], normalize_embeddings=True)[0]
            .tolist()
        )