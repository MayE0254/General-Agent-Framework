"""Hybrid knowledge retriever: Milvus vectors + BM25 keyword + cross-encoder rerank.

Pipeline::

    query
      -> embed (query)                      LocalEmbeddingModel
      -> Milvus cosine top-N                 MilvusStore.search
      -> jieba tokens -> BM25 top-N          BM25Index
      -> graph expansion top-M (optional)    app.knowledge.graph.expand
      -> RRF pre-screen (rank fusion)        reciprocal_rank_fusion
      -> collapse sibling children           diversity.collapse_by_parent
      -> per-document quota                  diversity.cap_per_document
      -> cross-encoder rerank top-k          CrossEncoderReranker
      -> expand child -> whole clause        diversity.expand_to_parent
      -> final KnowledgeHit list (primary + flagged quota overflow)

Two-stage design: RRF merges the legs by *rank* (so the incomparable cosine /
BM25 score scales never need normalising) and caps the pool; the cross-encoder
then decides the final order on the whole pre-screened pool.  With N per leg the
pool is at most 2N, so the RRF cap is a no-op for small N and only bites once
the recall fan-out grows.

Two diversity controls sit between fusion and reranking because a pure top-k
pool has no notion of coverage: on a corpus mixing long statutes with short
notices a single statute can fill every slot, leaving the cross-encoder to
choose among fragments of one document.  See ``app.knowledge.diversity``.

The quota has one blind spot that no pool size fixes: a question answered across
several clauses of the *same* statute, where the article holding the answer is
the 4th clause of a document that may only contribute 3.  Such clauses come back
after the top-k, flagged (``QUOTA_OVERFLOW_KEY``) rather than merged, so the
quota still shapes the ranked results while the answerer can offer them as
explicitly secondary evidence.

The BM25 leg needs the full chunk list; it is built once from a chunk snapshot
file (the ingestion step writes ``processed/chunks.jsonl``) so keyword recall
does not depend on re-embedding the whole corpus on every start.
"""
from __future__ import annotations

from collections.abc import Iterable

from app.knowledge.base import (
    QUOTA_OVERFLOW_KEY,
    BaseKnowledgeRetriever,
    KnowledgeHit,
)
from app.knowledge.bm25 import BM25Index
from app.knowledge.diversity import (
    cap_per_document,
    collapse_by_parent,
    expand_to_parent,
)
from app.knowledge.embeddings import LocalEmbeddingModel
from app.knowledge.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from app.knowledge.graph.expand import GraphExpander, seed_clause_ids
from app.knowledge.rerank import CrossEncoderReranker
from app.knowledge.vectorstore.milvus import MilvusStore


class HybridRetriever(BaseKnowledgeRetriever):
    """Vector + keyword fusion with local cross-encoder reranking."""

    name = "hybrid"

    def __init__(
        self,
        store: MilvusStore,
        embedding: LocalEmbeddingModel,
        reranker: CrossEncoderReranker,
        chunk_snapshot: Iterable[KnowledgeHit] | None = None,
        *,
        candidates: int = 8,
        top_k: int = 5,
        rrf_k: int = DEFAULT_RRF_K,
        prefusion_top_n: int = 20,
        max_chunks_per_document: int = 3,
        quota_overflow_top_n: int = 0,
        graph_expander: GraphExpander | None = None,
        allow_vector_fallback: bool = True,
    ) -> None:
        self.store = store
        self.embedding = embedding
        self.reranker = reranker
        self.candidates = candidates
        self.top_k = top_k
        self.rrf_k = rrf_k
        self.prefusion_top_n = prefusion_top_n
        self.max_chunks_per_document = max_chunks_per_document
        self.quota_overflow_top_n = quota_overflow_top_n
        self.allow_vector_fallback = allow_vector_fallback
        # Absent by default: the graph leg only runs when a caller wires an
        # expander in, and even then ``[knowledge.graph.expand].enabled`` gates
        # it (see ``GraphExpander.expand``).  Two switches because "the chain
        # cannot expand" and "this run chose not to" must stay distinguishable.
        self.graph_expander = graph_expander
        self.bm25: BM25Index | None = None
        if chunk_snapshot is not None:
            self.bm25 = BM25Index(list(chunk_snapshot))

    def build_bm25(self, chunks: Iterable[KnowledgeHit]) -> None:
        self.bm25 = BM25Index(list(chunks))

    async def retrieve(self, query: str, *, top_k: int = 5) -> list[KnowledgeHit]:
        # Recall: each leg ranks independently; keep their lists separate so
        # RRF can use the ranks.
        vec_hits: list[KnowledgeHit] = []
        try:
            qv = self.embedding.embed_query(query)
            vec_hits = self.store.search(qv, top_k=self.candidates)
        except Exception:
            if not self.allow_vector_fallback:
                raise
            # Milvus down — the keyword leg can still answer.
            pass

        bm_hits: list[KnowledgeHit] = []
        if self.bm25 is not None:
            bm_hits = self.bm25.to_hits(self.bm25.search(query, top_k=self.candidates))

        # Third recall leg: follow the graph's own cross-references and shared
        # subjects out of the clauses the lexical legs already ranked (see
        # ``app.knowledge.graph.expand``).  Seeded from *both* lexical legs so
        # the expansion starts where either of them found something; the leg's
        # output joins the same RRF contest as the other two, so the
        # cross-encoder -- not the graph -- still decides the final order.
        graph_hits: list[KnowledgeHit] = []
        if self.graph_expander is not None:
            seeds = seed_clause_ids(
                [vec_hits, bm_hits],
                limit=self.graph_expander.settings.seed_top_n,
            )
            # The pool budget is shared with the lexical legs, so the graph leg
            # is capped at the pool it feeds rather than at its own max alone.
            graph_hits = self.graph_expander.expand(seeds, limit=self.prefusion_top_n)

        # Pre-screen: rank-fuse the legs and cap the pool reaching the reranker.
        fused = reciprocal_rank_fusion([vec_hits, bm_hits, graph_hits], k=self.rrf_k)
        if not fused:
            return []

        # Sibling child windows of one clause would occupy several slots with
        # identical text once expanded, so collapse them first (this also
        # removes the overlap duplication between adjacent windows).
        collapsed = collapse_by_parent(fused)

        # Diversity: quota each source document.  `limit` is the pool size the
        # reranker should see, so it is max(..., top_k) for the same reason the
        # plain slice below guards against a misconfigured cap starving top_k.
        prefiltered, overflow = cap_per_document(
            collapsed,
            max_per_document=self.max_chunks_per_document,
            limit=max(self.prefusion_top_n, top_k),
            overflow_limit=self.quota_overflow_top_n,
        )

        # Precision: the cross-encoder decides the final ordering on child text.
        reranked = self.reranker.rerank(query, prefiltered)
        # Return whole clauses ("retrieve small, return big").
        primary = expand_to_parent(reranked)[: max(top_k, 1)]

        # Supplementary recall for what the quota rejected (see
        # ``cap_per_document``).  Deliberately NOT reranked: the quota bounds the
        # cross-encoder's input because scoring is the expensive step, so paying
        # that cost again for hints would negate the reason the pool is capped.
        # Their RRF rank order is the only ordering they have, and the flag is
        # what keeps them from being read as equal evidence downstream.
        supplementary = [
            hit.model_copy(
                update={
                    "metadata_payload": {
                        **hit.metadata_payload,
                        QUOTA_OVERFLOW_KEY: True,
                    }
                }
            )
            for hit in expand_to_parent(overflow)
        ]
        return primary + supplementary
