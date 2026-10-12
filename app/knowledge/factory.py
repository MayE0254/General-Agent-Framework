"""Shared single-hop construction for the console, tools and evaluation."""
from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from app.core import ResourceNotReadyError
from app.core.settings import KnowledgeSettings
from app.knowledge.base import BaseKnowledgeRetriever, KnowledgeHit
from app.knowledge.embeddings import LocalEmbeddingModel
from app.knowledge.hybrid import HybridRetriever
from app.knowledge.ingest.pipeline import load_chunk_snapshot
from app.knowledge.rerank import CrossEncoderReranker
from app.knowledge.vectorstore.milvus import MilvusStore

ROOT = Path(__file__).resolve().parents[2]


def resolve_model_ref(model_ref: str) -> str:
    candidate = Path(model_ref)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    if candidate.is_dir():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        return str(candidate)
    return model_ref


def build_single_hop_retriever(
    settings: KnowledgeSettings,
    *,
    embedding: LocalEmbeddingModel | None = None,
    chunks: list[KnowledgeHit] | None = None,
    strict_vector: bool = True,
) -> HybridRetriever:
    """Never consumes graph configuration, even when experiment flags are on."""
    if not settings.milvus.enabled:
        raise ValueError("Single-hop RAG requires Milvus to be enabled.")
    if chunks is None:
        path = ROOT / settings.chunk_snapshot
        chunks = load_chunk_snapshot(str(path))
    if not chunks:
        raise ValueError("The knowledge chunk snapshot is empty.")
    store = MilvusStore(settings.milvus)
    try:
        store.verify_schema()
        return HybridRetriever(
            store,
            embedding or LocalEmbeddingModel(resolve_model_ref(settings.embedding_model)),
            CrossEncoderReranker(resolve_model_ref(settings.reranker_model)),
            chunk_snapshot=chunks,
            candidates=settings.hybrid_candidates,
            top_k=settings.top_k,
            rrf_k=settings.rrf_k,
            prefusion_top_n=settings.prefusion_top_n,
            max_chunks_per_document=settings.max_chunks_per_document,
            quota_overflow_top_n=settings.quota_overflow_top_n,
            allow_vector_fallback=not strict_vector,
            graph_expander=None,
        )
    except Exception:
        store.close()
        raise


class RuntimeKnowledgeRetriever(BaseKnowledgeRetriever):
    """One lazy model/index pair per process, off the API event loop.

    A single worker serializes local model loading and inference. Only
    retrieval runs here; user-scoped DB sessions and LLM calls stay on the
    calling thread. The worker is also safe across Celery's per-task loops.
    """

    name = "hybrid"

    def __init__(
        self,
        settings: KnowledgeSettings,
        *,
        builder: Callable[..., HybridRetriever] = build_single_hop_retriever,
    ) -> None:
        self._settings = settings.model_copy(deep=True)
        self._builder = builder
        self._retriever: HybridRetriever | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag")
        self._closed = False

    def _retrieve(self, query: str, top_k: int) -> list[KnowledgeHit]:
        if self._retriever is None:
            self._retriever = self._builder(self._settings)
        return asyncio.run(self._retriever.retrieve(query, top_k=top_k))

    async def retrieve(self, query: str, *, top_k: int = 5) -> list[KnowledgeHit]:
        if self._closed:
            raise ResourceNotReadyError("知识库检索服务已关闭。")
        try:
            return await asyncio.get_running_loop().run_in_executor(
                self._executor, self._retrieve, query, top_k
            )
        except Exception as exc:
            raise ResourceNotReadyError(
                "知识库检索暂不可用，请检查 Milvus、本地模型和语料快照；本次未生成无依据回答。"
            ) from exc

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=True)
        if self._retriever is not None:
            self._retriever.store.close()
            self._retriever = None
