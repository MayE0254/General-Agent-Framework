"""Post-retrieval diversity controls for the hybrid retriever.

Three small, order-preserving transforms that sit between fusion/rerank and the
caller.  They exist because raw top-k similarity has no notion of *coverage*:
a corpus that mixes long statutes (民法典 alone is ~1.2e3 chunks) with one-page
notices lets the statute fill every candidate slot, so the cross-encoder ends up
choosing among fragments of a single document and the answer loses the other
relevant sources entirely.

- :func:`collapse_by_parent` -- keep one child per clause ("第X条").  Children
  of the same parent carry identical text once expanded, so without this they
  consume several slots each.
- :func:`cap_per_document` -- greedy per-source quota (the "N chunks per
  document" rule), which is what actually bounds a long statute's share.  The
  quota-rejected remainder comes back as a separate list so a caller can offer
  it as supplementary evidence without letting it back into the pool.
- :func:`expand_to_parent` -- swap a child window for its whole clause, i.e.
  "retrieve small, return big".

All three preserve relative order, so they compose with a rank-ordered list
(RRF output, reranker output) without needing to re-sort.
"""
from __future__ import annotations

from app.knowledge.base import KnowledgeHit


def collapse_by_parent(hits: list[KnowledgeHit]) -> list[KnowledgeHit]:
    """Keep only the top-ranked child of each parent clause.

    Keyed on ``parent_id`` when present, else on ``id`` (unparented hits are
    their own parent).  The dropped-sibling count is recorded on the surviving
    hit so the overlap duplication between adjacent child windows stays
    observable instead of being silently absorbed.
    """
    survivors: dict[str, KnowledgeHit] = {}
    dropped: dict[str, int] = {}
    order: list[str] = []
    for hit in hits:
        key = hit.parent_id or hit.id
        if key in survivors:
            dropped[key] = dropped.get(key, 0) + 1
            continue
        survivors[key] = hit
        order.append(key)

    collapsed: list[KnowledgeHit] = []
    for key in order:
        hit = survivors[key]
        if dropped.get(key):
            hit = hit.model_copy(
                update={
                    "metadata_payload": {
                        **hit.metadata_payload,
                        "collapsed_siblings": dropped[key],
                    }
                }
            )
        collapsed.append(hit)
    return collapsed


def cap_per_document(
    hits: list[KnowledgeHit],
    *,
    max_per_document: int,
    limit: int,
    overflow_limit: int = 0,
) -> tuple[list[KnowledgeHit], list[KnowledgeHit]]:
    """Greedy per-source quota, in rank order, capped at ``limit`` results.

    Walks ``hits`` best-first and takes a hit only while its source document has
    contributed fewer than ``max_per_document`` so far.  Rank order is the input
    order, so this is "highest-scoring N chunks per document" -- relevance
    inside a document is never sacrificed for a weaker document's top hit.

    ``limit`` is the pool size the caller needs (``prefusion_top_n``).  If the
    quota under-fills it, the remaining slots are backfilled from the skipped
    candidates in the same rank order: the quota exists to *guarantee* diversity
    when it is available, not to return fewer results when the corpus really is
    dominated by one document.  Backfilling therefore cannot shrink the pool
    relative to no capping at all.

    A hit with an empty ``source`` falls back to its own ``id`` as the grouping
    key, so uncited hits are never lumped into one bucket.

    Returns ``(kept, overflow)``.  ``overflow`` holds the candidates the *quota*
    rejected and that backfilling did not absorb -- i.e. clauses the retriever
    judged relevant and then dropped purely for being the 4th, 5th... clause of
    an already-represented document, in rank order, at most ``overflow_limit`` of
    them.  They answer a query shape the quota structurally cannot: a question
    whose answer spans several clauses of one statute (only 3 of them may enter
    the pool, so the article that actually holds the answer can lose to its
    neighbours).  The caller decides how to present them; they are returned
    separately, never appended to ``kept``, because they lose the rank contest
    and never see the cross-encoder -- mixing them in would silently undo the
    diversity this function exists to provide.

    Candidates past the rank-``limit`` cutoff are *not* overflow: they lost the
    ranking competition rather than the quota, so "was judged relevant" does not
    apply to them.
    """
    if max_per_document < 1:
        raise ValueError(f"max_per_document must be >= 1, got {max_per_document}")
    if limit < 1:
        return [], []
    if overflow_limit < 0:
        raise ValueError(f"overflow_limit must be >= 0, got {overflow_limit}")

    kept: list[KnowledgeHit] = []
    skipped: list[KnowledgeHit] = []
    per_source: dict[str, int] = {}
    for hit in hits:
        key = hit.source or hit.id
        if per_source.get(key, 0) < max_per_document:
            per_source[key] = per_source.get(key, 0) + 1
            kept.append(hit)
            if len(kept) >= limit:
                return kept, skipped[:overflow_limit]
        else:
            skipped.append(hit)

    backfill = skipped[: max(0, limit - len(kept))]
    return kept + backfill, skipped[len(backfill) :][:overflow_limit]


def expand_to_parent(hits: list[KnowledgeHit]) -> list[KnowledgeHit]:
    """Replace a child window's text with its parent clause's full text.

    ``parent_text`` is empty when the child *is* the whole clause, in which case
    the hit passes through untouched -- so "is there a parent to expand?" is
    answered in exactly one place.  ``id`` deliberately still points at the
    matched child: that is the unit that exists in Milvus and the BM25 snapshot,
    while ``parent_id`` identifies the clause for citation.
    """
    expanded: list[KnowledgeHit] = []
    for hit in hits:
        if not hit.parent_text:
            expanded.append(hit)
            continue
        expanded.append(
            hit.model_copy(
                update={
                    "content": hit.parent_text,
                    "metadata_payload": {
                        **hit.metadata_payload,
                        "matched_child_id": hit.id,
                        "matched_child_char_len": len(hit.content),
                        "expanded_to_parent": True,
                    },
                }
            )
        )
    return expanded
