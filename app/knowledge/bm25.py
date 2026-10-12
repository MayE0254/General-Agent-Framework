"""BM25 keyword retriever with Chinese tokenization.

The full-text (keyword) leg of hybrid retrieval.  Unlike the embedding leg it
does not need any model download and works offline over the in-memory or
indexed chunk list; it complements vector recall for exact-term queries
(e.g. a taxpayer searching "增值税专用发票" verbatim).

BM25 is implemented locally (no extra dependency beyond ``jieba``) so the
retrieval pipeline holds up even before the embedding model finishes loading.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

from app.knowledge.base import KnowledgeHit


def tokenize_zh(text: str) -> list[str]:
    """Chinese tokenization via jieba; falls back to raw ascii tokens."""
    try:
        import jieba

        jieba.initialize()
        tokens = [t.strip().lower() for t in jieba.cut(text or "") if t.strip()]
    except Exception:  # pragma: no cover - jieba absent
        tokens = [t.lower() for t in (text or "").split() if t.isalnum()]
    return tokens


class BM25Index:
    """In-memory BM25 index over a fixed chunk list."""

    def __init__(self, chunks: list[KnowledgeHit], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_len: list[int] = []
        self.avg_dl = 0.0
        self._build(chunks)

    def _build(self, chunks: list[KnowledgeHit]) -> None:
        docs: list[list[str]] = []
        df: Counter[str] = Counter()
        for hit in chunks:
            toks = tokenize_zh(f"{hit.title} {hit.content}")
            docs.append(toks)
            df.update(set(toks))
        self.doc_len = [len(d) for d in docs]
        total = sum(self.doc_len)
        self.avg_dl = total / len(docs) if docs else 0.0
        self._docs = docs
        self._df = df
        # term -> postings: doc index -> term frequency
        self._tf: list[dict[str, int]] = []
        for d in docs:
            self._tf.append(Counter(d))
        self.N = len(docs)
        # idf per term
        self._idf: dict[str, float] = {
            term: math.log1p((self.N - freq + 0.5) / (freq + 0.5))
            for term, freq in self._df.items()
        }

    def search(self, query: str, *, top_k: int = 5) -> list[tuple[float, int]]:
        q_tokens = tokenize_zh(query)
        if not q_tokens or not self._docs:
            return []
        scores: list[float] = [0.0] * self.N
        for term in set(q_tokens):
            if term not in self._idf:
                continue
            idf = self._idf[term]
            for doc_idx, tf in self._tf_freqs(term):
                dl = self.doc_len[doc_idx]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avg_dl) if dl else 1.0
                scores[doc_idx] += idf * (tf * (self.k1 + 1)) / denom
        ranked = sorted(
            ((scores[i], i) for i in range(self.N) if scores[i] > 0),
            key=lambda x: -x[0],
        )
        return ranked[:top_k]

    def _tf_freqs(self, term: str) -> list[tuple[int, int]]:
        return [(i, self._tf[i].get(term, 0)) for i in range(self.N) if self._tf[i].get(term, 0) > 0]

    def to_hits(self, ranked: list[tuple[float, int]]) -> list[KnowledgeHit]:
        result: list[KnowledgeHit] = []
        for score, idx in ranked:
            base = self.chunks[idx]
            result.append(
                KnowledgeHit(
                    id=base.id,
                    title=base.title,
                    content=base.content,
                    score=round(score, 4),
                    source=base.source,
                    parent_id=base.parent_id,
                    parent_text=base.parent_text,
                    metadata_payload={
                        **base.metadata_payload,
                        "bm25_score": round(score, 4),
                    },
                )
            )
        return result