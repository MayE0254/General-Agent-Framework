"""Reciprocal Rank Fusion (RRF) for multi-leg recall pre-screening.

The hybrid retriever fans out to several recall legs (Milvus vector, BM25
keyword, and in the future graph/field-filtered legs).  Each leg returns its
own ranked list with an *incomparable* score scale (cosine in [0, 1] vs BM25
which is unbounded), so raw scores cannot be merged directly.

RRF sidesteps that by using only **ranks**::

    score(d) = Σ_legs  1 / (k + rank_leg(d))        rank starts at 1

A document that both legs rank highly accumulates a large score; one found by
a single leg still gets a contribution.  This is used purely as a **cheap
pre-screen**: the fused list is truncated before the expensive cross-encoder
rerank, so the final ordering still comes from the cross-encoder, not from RRF.
"""
from __future__ import annotations

from collections.abc import Sequence

from app.knowledge.base import KnowledgeHit

# Standard RRF smoothing constant (Cormack et al., 2009).  Larger k flattens
# the rank weighting (diminishes the advantage of being #1).
DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[KnowledgeHit]],
    *,
    k: int = DEFAULT_RRF_K,
) -> list[KnowledgeHit]:
    """Fuse ranked hit lists into one RRF-ordered list.

    - dedupes by ``KnowledgeHit.id`` across legs;
    - merges each leg's ``metadata_payload`` so the fused hit carries e.g.
      both ``vector_score`` and ``bm25_score`` for observability;
    - writes ``rrf_score`` and ``rrf_rank`` into the payload;
    - ties break by best single-leg rank, then id, for deterministic output.

    The fused hit is *derived* from its representative with ``model_copy``
    rather than rebuilt field by field.  A hand-written field list silently
    drops whatever it does not mention -- ``parent_id`` / ``parent_text`` were
    lost here, which made the whole parent/child index inert downstream
    (``collapse_by_parent`` fell back to deduping by id, and
    ``expand_to_parent`` passed every hit through), so the retriever returned
    142-char child fragments instead of whole clauses.
    """
    if k < 1:
        raise ValueError("RRF k must be >= 1")

    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    representatives: dict[str, KnowledgeHit] = {}
    payloads: dict[str, dict] = {}

    for ranked in ranked_lists:
        for rank, hit in enumerate(ranked, start=1):
            cid = hit.id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            best_rank[cid] = min(best_rank.get(cid, rank), rank)
            representatives.setdefault(cid, hit)
            payloads.setdefault(cid, {}).update(hit.metadata_payload)

    ordered = sorted(
        scores,
        key=lambda cid: (-scores[cid], best_rank[cid], cid),
    )

    fused: list[KnowledgeHit] = []
    for new_rank, cid in enumerate(ordered, start=1):
        payload = dict(payloads[cid])
        payload["rrf_score"] = scores[cid]
        payload["rrf_rank"] = new_rank
        fused.append(
            representatives[cid].model_copy(
                update={"score": scores[cid], "metadata_payload": payload}
            )
        )
    return fused
