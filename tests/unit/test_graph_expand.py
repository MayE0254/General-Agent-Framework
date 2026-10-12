"""Unit tests for the graph expansion recall leg (no Neo4j, no Milvus).

``GraphExpander`` takes its Cypher runner by injection, so the traversal can be
exercised with a fake ``query_all``: what is under test here is the ranking,
the seeding, the rehydration and the caps -- i.e. everything that decides
*which* clauses the leg hands to RRF.  The Cypher text itself is asserted on
only for direction-agnosticism, since that is a correctness property a fake
cannot catch.
"""

from app.core.settings import GraphExpandSettings
from app.knowledge.base import KnowledgeHit
from app.knowledge.graph.expand import (
    GRAPH_EXPANDED_KEY,
    GRAPH_KIND_KEY,
    GRAPH_SEEDS_KEY,
    GRAPH_WEIGHT_KEY,
    KIND_CITES,
    KIND_MENTIONS,
    ClauseTextIndex,
    GraphExpander,
    clause_key,
    rank_neighbors,
    seed_clause_ids,
)


def _hit(
    chunk_id: str,
    *,
    parent_id: str = "",
    title: str = "中华人民共和国企业所得税法",
    section: str = "第三条",
    content: str = "第三条……",
    parent_text: str = "",
) -> KnowledgeHit:
    return KnowledgeHit(
        id=chunk_id,
        title=title,
        content=content,
        parent_id=parent_id,
        parent_text=parent_text,
        metadata_payload={"section": section},
    )


# --------------------------------------------------------------------------
# clause_key / ClauseTextIndex
# --------------------------------------------------------------------------
def test_clause_key_mirrors_the_write_side_fallback() -> None:
    # extract.py keys an unparented chunk by its own id; the read side must
    # agree or those clauses are in the graph but unreachable from the corpus.
    assert clause_key(_hit("c1", parent_id="p1")) == "p1"
    assert clause_key(_hit("c1")) == "c1"


def test_clause_index_keeps_the_hit_unchanged_and_first_wins() -> None:
    first = _hit("c1", parent_id="p1", content="child one", parent_text="full clause")
    second = _hit("c2", parent_id="p1", content="child two", parent_text="full clause")
    index = ClauseTextIndex([first, second])

    assert len(index) == 1
    kept = index.get("p1")
    # The child window is what the cross-encoder scores, so the leg must hand
    # over the same unit the lexical legs do -- not the parent text.
    assert kept is not None
    assert kept.id == "c1"
    assert kept.content == "child one"
    assert kept.parent_text == "full clause"


def test_clause_index_indexes_unparented_chunks_under_their_own_id() -> None:
    index = ClauseTextIndex([_hit("c1")])
    assert index.get("c1") is not None
    assert index.get("") is None


# --------------------------------------------------------------------------
# seed_clause_ids
# --------------------------------------------------------------------------
def test_seeds_walk_legs_round_robin_not_concatenated() -> None:
    vector = [_hit("v1", parent_id="pv1"), _hit("v2", parent_id="pv2")]
    bm25 = [_hit("b1", parent_id="pb1"), _hit("b2", parent_id="pb2")]

    # Interleaved: a single leg long enough to fill `limit` must not spend
    # every seed slot and starve the other leg's top clause.
    assert seed_clause_ids([vector, bm25], limit=3) == ["pv1", "pb1", "pv2"]
    assert seed_clause_ids([vector, bm25], limit=10) == [
        "pv1", "pb1", "pv2", "pb2"
    ]


def test_seeds_dedupe_across_legs_and_honour_the_limit() -> None:
    vector = [_hit("v1", parent_id="shared"), _hit("v2", parent_id="pv2")]
    bm25 = [_hit("b1", parent_id="shared"), _hit("b2", parent_id="pb2")]

    assert seed_clause_ids([vector, bm25], limit=5) == ["shared", "pv2", "pb2"]
    assert seed_clause_ids([vector, bm25], limit=1) == ["shared"]
    assert seed_clause_ids([], limit=5) == []


# --------------------------------------------------------------------------
# rank_neighbors
# --------------------------------------------------------------------------
def test_rank_neighbors_prefers_weight_then_seed_consensus() -> None:
    rows = [
        {"seed": "s1", "clause_id": "a", "weight": 2},
        {"seed": "s2", "clause_id": "a", "weight": 2},
        {"seed": "s1", "clause_id": "b", "weight": 3},
    ]
    ranked = rank_neighbors(rows, kind=KIND_CITES, max_results=10)

    # a: weight 4 from two seeds, b: weight 3 from one.
    assert [n.clause_id for n in ranked] == ["a", "b"]
    assert ranked[0].seed_count == 2
    assert ranked[1].seed_count == 1


def test_rank_neighbors_ties_break_on_clause_id() -> None:
    rows = [
        {"seed": "s1", "clause_id": "z", "weight": 1},
        {"seed": "s1", "clause_id": "y", "weight": 1},
    ]
    cites = rank_neighbors(rows, kind=KIND_CITES, max_results=10)
    mentions = rank_neighbors(rows, kind=KIND_MENTIONS, max_results=10)

    # Deterministic output: identical weights resolve by id, not by dict order.
    assert [n.clause_id for n in cites] == ["y", "z"]
    assert [n.clause_id for n in mentions] == ["y", "z"]
    assert cites[0].kinds == frozenset({KIND_CITES})
    assert mentions[0].kinds == frozenset({KIND_MENTIONS})


def test_rank_neighbors_ignores_rows_without_a_clause_and_caps_output() -> None:
    rows = [
        {"seed": "s1", "clause_id": "", "weight": 9},
        {"seed": "s1", "clause_id": None, "weight": 9},
        {"seed": "s1", "clause_id": "a", "weight": 5},
        {"seed": "s1", "clause_id": "b", "weight": 4},
    ]
    ranked = rank_neighbors(rows, kind=KIND_CITES, max_results=1)
    assert [n.clause_id for n in ranked] == ["a"]


def test_rank_neighbors_tolerates_a_missing_weight() -> None:
    ranked = rank_neighbors(
        [{"seed": "s1", "clause_id": "a"}], kind=KIND_CITES, max_results=5
    )
    assert ranked[0].weight == 0


# --------------------------------------------------------------------------
# GraphExpander
# --------------------------------------------------------------------------
class _FakeGraph:
    """Records the queries it is asked and answers from a fixed table."""

    def __init__(self, rows: dict[str, list[dict]]) -> None:
        self.rows = rows
        self.queries: list[str] = []

    def __call__(self, query: str, params: dict) -> list[dict]:
        self.queries.append(query)
        return list(self.rows.get("result", []))


def _expander(rows: list[dict], chunks: list[KnowledgeHit], **overrides):
    settings = GraphExpandSettings(enabled=True, **overrides)
    fake = _FakeGraph({"result": rows})
    return GraphExpander(settings, fake, ClauseTextIndex(chunks)), fake


def test_expand_is_inert_until_explicitly_enabled() -> None:
    chunks = [_hit("c1", parent_id="p1")]
    fake = _FakeGraph({"result": [{"seed": "s", "clause_id": "p1", "weight": 1}]})
    expander = GraphExpander(
        GraphExpandSettings(enabled=False), fake, ClauseTextIndex(chunks)
    )

    assert expander.expand(["s"]) == []
    assert fake.queries == []


def test_expand_returns_rehydrated_hits_with_provenance_payload() -> None:
    chunks = [_hit("c1", parent_id="p1", content="child", parent_text="whole clause")]
    rows = [{"seed": "s", "clause_id": "p1", "weight": 2}]
    expander, _ = _expander(rows, chunks, use_mentions=False)

    hits = expander.expand(["s"])
    assert len(hits) == 1
    payload = hits[0].metadata_payload
    assert payload[GRAPH_EXPANDED_KEY] is True
    assert payload[GRAPH_KIND_KEY] == KIND_CITES
    assert payload[GRAPH_WEIGHT_KEY] == 2
    assert payload[GRAPH_SEEDS_KEY] == 1
    # Same unit as every other leg: child window, parent text still available.
    assert hits[0].content == "child"
    assert hits[0].parent_text == "whole clause"


def test_expand_drops_clauses_the_snapshot_cannot_rehydrate() -> None:
    chunks = [_hit("c1", parent_id="p1")]
    rows = [
        {"seed": "s", "clause_id": "p1", "weight": 1},
        {"seed": "s", "clause_id": "stale", "weight": 5},
    ]
    expander, _ = _expander(rows, chunks, use_mentions=False)

    hits = expander.expand(["s"])
    # No empty-content hit may enter the prompt just because the graph has the
    # clause id.
    assert [h.parent_id for h in hits] == ["p1"]


def test_expand_runs_one_query_per_enabled_leg() -> None:
    chunks = [_hit("c1", parent_id="p1")]
    rows = [{"seed": "s", "clause_id": "p1", "weight": 1}]

    cites_only, fake_c = _expander(rows, chunks, use_mentions=False)
    cites_only.expand(["s"])
    assert len(fake_c.queries) == 1

    both, fake_b = _expander(rows, chunks)
    both.expand(["s"])
    assert len(fake_b.queries) == 2

    neither, fake_n = _expander(rows, chunks, use_cites=False, use_mentions=False)
    assert neither.expand(["s"]) == []
    assert fake_n.queries == []


def test_expand_caps_at_max_results_and_only_tightens_with_limit() -> None:
    chunks = [_hit(f"c{i}", parent_id=f"p{i}") for i in range(5)]
    rows = [
        {"seed": "s", "clause_id": f"p{i}", "weight": 5 - i} for i in range(5)
    ]

    capped, _ = _expander(rows, chunks, use_mentions=False, max_results=3)
    assert len(capped.expand(["s"])) == 3
    # `limit` is the shared RRF pool budget: it can shrink the leg, never grow it.
    assert len(capped.expand(["s"], limit=2)) == 2
    assert len(capped.expand(["s"], limit=99)) == 3


def test_expand_no_seeds_means_no_query() -> None:
    chunks = [_hit("c1", parent_id="p1")]
    rows = [{"seed": "s", "clause_id": "p1", "weight": 1}]
    expander, fake = _expander(rows, chunks, use_mentions=False)
    assert expander.expand([]) == []
    assert fake.queries == []


def test_expand_merges_both_legs_for_the_same_clause() -> None:
    chunks = [_hit("c1", parent_id="p1")]
    settings = GraphExpandSettings(enabled=True)
    # Same clause reached by both legs: weights add, kinds union.
    fake = _FakeGraph({"result": [{"seed": "s", "clause_id": "p1", "weight": 2}]})
    expander = GraphExpander(settings, fake, ClauseTextIndex(chunks))

    hits = expander.expand(["s"])
    assert len(hits) == 1
    payload = hits[0].metadata_payload
    assert payload[GRAPH_WEIGHT_KEY] == 4
    assert payload[GRAPH_KIND_KEY] == f"{KIND_CITES}+{KIND_MENTIONS}"


def test_queries_are_direction_agnostic() -> None:
    from app.knowledge.graph.expand import _CITES_QUERY, _MENTIONS_QUERY

    # A citation points backwards in the corpus, so a direction-sensitive
    # pattern would make the hop one-way and hide the cited clause.
    assert "(c)-[r:CITES]-(n:Clause)" in _CITES_QUERY
    assert "(c)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(n:Clause)" in _MENTIONS_QUERY
    # Caps live in Cypher, not only afterwards: the per-seed LIMIT is what keeps
    # a generic entity from spending the whole fan-out budget.
    assert "LIMIT $per_seed" in _CITES_QUERY
    assert "LIMIT $per_seed" in _MENTIONS_QUERY
    # Variable-scope subquery: the bare ``CALL { WITH c`` form makes Neo4j log a
    # deprecation warning on every call, i.e. on every retrieval.
    assert "CALL (c) {" in _CITES_QUERY
    assert "CALL (c) {" in _MENTIONS_QUERY


def test_verify_reports_the_clause_count() -> None:
    chunks = [_hit("c1", parent_id="p1")]
    fake = _FakeGraph({"result": [{"c": 23111}]})
    expander = GraphExpander(
        GraphExpandSettings(enabled=True), fake, ClauseTextIndex(chunks)
    )
    assert expander.verify() == {"clauses": 23111}


# --------------------------------------------------------------------------
# HybridRetriever wiring
# --------------------------------------------------------------------------
def test_hybrid_wires_the_graph_leg_into_rrf() -> None:
    import asyncio

    from app.knowledge.hybrid import HybridRetriever

    chunks = [_hit("c1", parent_id="p1", parent_text="whole clause")]
    expander, _ = _expander(
        [{"seed": "s", "clause_id": "p1", "weight": 1}],
        chunks,
        use_mentions=False,
    )

    class _Store:
        def search(self, *a, **kw):  # vector leg returns the seed clause only
            return [_hit("seed", parent_id="seed")]

    class _Embed:
        def embed_query(self, q):
            return [0.0]

    class _Rerank:
        def rerank(self, query, hits):
            return hits

    retriever = HybridRetriever(
        _Store(), _Embed(), _Rerank(), graph_expander=expander
    )
    hits = asyncio.run(retriever.retrieve("q", top_k=5))

    # The expanded clause reached the final contexts, i.e. the leg really was
    # fused rather than merely constructed.
    assert any(h.parent_id == "p1" for h in hits)
