"""Graph expansion as a third recall leg for the hybrid retriever.

The vector and BM25 legs can only reach a clause whose *words* match the query.
Two clause-to-clause paths exist in the graph that no lexical leg can follow:

* ``Clause -CITES-> Clause`` -- the corpus's own cross-references, including the
  hand-off from a law to its implementing regulation (mining 企税法 第三条 to
  实施条例 第九十一条).
* ``Clause -MENTIONS-> Entity <-MENTIONS- Clause`` -- two clauses that discuss
  the same subject without naming each other (国发〔2022〕8号 and 国发〔2023〕13号
  both cover 3 岁以下婴幼儿照护扣除).  This is a *two-hop traversal*, not a
  materialised edge: a modelled "related clauses" relation would be O(n^2) junk
  (see ``app.knowledge.graph.neo4j``).

Both are read direction-agnostically on purpose.  A citation points backwards in
the corpus (第四条 cites 第三条), so following it forwards only would make the
hop available to the citing clause and not to the cited one -- and the query
usually matches the *definition*, not the *application*.

Three design points worth stating:

* **This is a recall leg, not a reranker.**  It returns candidates that are then
  rank-fused with the other legs (RRF) and reranked by the cross-encoder, so it
  never has to produce a calibrated score.  Order within the leg only breaks
  ties, and every hit carries ``graph_expanded`` / ``graph_kind`` so the share
  of the final context that came from the graph stays measurable -- a leg whose
  output is silently outranked would otherwise look identical to a leg that
  never ran.
* **The graph stores no clause text** (connectivity only, see ``neo4j.py``), so
  expanded clauses are rehydrated from the chunk snapshot through
  :class:`ClauseTextIndex`.  A clause with no snapshot row is dropped, never
  emitted with empty content.
* **Two hops, hard caps.**  Fan-out is real: an entity like 增值税 is mentioned
  by thousands of clauses.  Each leg is therefore capped per seed in Cypher and
  again in total, because an unbounded expansion would flood the RRF pool and
  displace the lexical legs entirely.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from app.core.settings import GraphExpandSettings
from app.knowledge.base import KnowledgeHit

# Read-only multi-row query runner (``Neo4jGraphStore.query_all``).  Injected
# rather than imported so the ranking logic is testable without a live Neo4j.
QueryAllFn = Callable[[str, dict], list[dict]]

KIND_CITES = "cites"
KIND_MENTIONS = "mentions"

# Payload keys written onto expanded hits.  Named constants because the
# comparison script and any future report read them back.
GRAPH_EXPANDED_KEY = "graph_expanded"
GRAPH_KIND_KEY = "graph_kind"
GRAPH_SEEDS_KEY = "graph_seeds"
GRAPH_WEIGHT_KEY = "graph_weight"

# Direction-agnostic by construction: no arrow on the ``CITES`` pattern, so a
# clause is reachable both from what it cites and from what cites it.
# ``CALL (c) {`` is the variable-scope form: the bare ``CALL { WITH c`` still
# works but Neo4j logs a deprecation warning per call, which on a per-query
# path turns into console noise during evaluation.
_CITES_QUERY = """
UNWIND $seeds AS sid
MATCH (c:Clause {clause_id: sid})
CALL (c) {
  MATCH (c)-[r:CITES]-(n:Clause)
  WHERE NOT n.clause_id IN $seeds
  RETURN n.clause_id AS clause_id, count(r) AS weight
  ORDER BY weight DESC, clause_id
  LIMIT $per_seed
}
RETURN sid AS seed, clause_id AS clause_id, weight AS weight
"""

# The two-hop bridge.  ``weight`` is how many entities the pair shares, which is
# the only strength signal available and is what ``min_shared`` filters on --
# filtering in Cypher rather than afterwards keeps the per-seed cap from being
# spent on pairs that would be discarded anyway.
_MENTIONS_QUERY = """
UNWIND $seeds AS sid
MATCH (c:Clause {clause_id: sid})
CALL (c) {
  MATCH (c)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(n:Clause)
  WHERE NOT n.clause_id IN $seeds
  WITH n, count(DISTINCT e) AS weight
  WHERE weight >= $min_shared
  RETURN n.clause_id AS clause_id, weight AS weight
  ORDER BY weight DESC, clause_id
  LIMIT $per_seed
}
RETURN sid AS seed, clause_id AS clause_id, weight AS weight
"""


class ClauseTextIndex:
    """``clause_id -> KnowledgeHit`` for clauses the graph reached.

    Built from the same chunk snapshot the BM25 leg uses.  The hit is kept
    *exactly* as the snapshot recorded it -- child text plus ``parent_text`` --
    so an expanded clause behaves like any other child hit downstream: the
    cross-encoder scores it on the same kind of window the lexical legs hand it,
    and ``expand_to_parent`` expands it after reranking.  Rehydrating straight
    into the parent text would instead hand the reranker a different unit of
    text than its competitors and quietly bias the comparison.

    The first chunk of a clause wins: any of its child windows yields the same
    ``parent_text``, and taking the first keeps ``id`` a real, stable Milvus /
    BM25 id rather than a synthetic one, which matters because
    ``collapse_by_parent`` dedupes on ``parent_id`` and RRF dedupes on ``id``.
    """

    def __init__(self, chunks: Iterable[KnowledgeHit]) -> None:
        self._by_clause: dict[str, KnowledgeHit] = {}
        for chunk in chunks:
            clause_id = clause_key(chunk)
            if clause_id in self._by_clause:
                continue
            self._by_clause[clause_id] = chunk

    def get(self, clause_id: str) -> KnowledgeHit | None:
        return self._by_clause.get(clause_id)

    def __len__(self) -> int:
        return len(self._by_clause)


def clause_key(hit: KnowledgeHit) -> str:
    """The graph's clause id for a hit: ``parent_id``, else the chunk's own id.

    Mirrors ``app.knowledge.graph.extract``'s ``parent_id or id``.  An
    unparented chunk *is* its own clause on the write side, so the read side has
    to key it identically -- otherwise those clauses would exist in the graph
    yet be unreachable from the corpus snapshot (or vice versa).
    """
    return (hit.parent_id or "").strip() or hit.id


@dataclass(frozen=True)
class Neighbor:
    """One expansion candidate, aggregated across the seeds that reached it."""

    clause_id: str
    weight: int
    kinds: frozenset[str]
    seed_count: int


def rank_neighbors(
    rows: Iterable[dict],
    *,
    kind: str,
    max_results: int,
) -> list[Neighbor]:
    """Aggregate raw query rows into a ranked, capped candidate list.

    Ranked by total bridge weight, then by how many seeds agree, then by kind
    (``cites`` before ``mentions``: a stated cross-reference is a stronger
    signal than a shared subject), then by id for determinism.  A clause reached
    from two seeds outranks one reached from one at the same weight, which is
    the cheap consensus signal available without a second graph pass.
    """
    weight: dict[str, int] = {}
    kinds: dict[str, set[str]] = {}
    seeds: dict[str, set[str]] = {}

    for row in rows:
        clause_id = row.get("clause_id")
        if not clause_id:
            continue
        weight[clause_id] = weight.get(clause_id, 0) + int(row.get("weight") or 0)
        kinds.setdefault(clause_id, set()).add(kind)
        seed = row.get("seed")
        if seed:
            seeds.setdefault(clause_id, set()).add(seed)

    neighbors = [
        Neighbor(
            clause_id=clause_id,
            weight=weight[clause_id],
            kinds=frozenset(kinds[clause_id]),
            seed_count=len(seeds.get(clause_id, ())),
        )
        for clause_id in weight
    ]
    neighbors.sort(
        key=lambda n: (
            -n.weight,
            -n.seed_count,
            0 if KIND_CITES in n.kinds else 1,
            n.clause_id,
        )
    )
    return neighbors[:max_results]


def seed_clause_ids(
    legs: Sequence[Sequence[KnowledgeHit]], *, limit: int
) -> list[str]:
    """The clause ids the expansion starts from, best-first and deduped.

    Takes the recall legs *separately* and walks them round-robin by rank
    (each leg's #1, then each leg's #2, ...) instead of concatenating: a single
    leg returning more than ``limit`` hits would otherwise spend every seed
    slot, and the other leg's top-ranked clause -- the one its whole leg exists
    to surface -- would never become a seed.

    Uses :func:`clause_key` -- the clause id the graph is keyed on -- never the
    child ``id``.  Order is the legs' own rank order, so the per-seed Cypher cap
    is spent on the best-ranked clauses.
    """
    seeds: list[str] = []
    seen: set[str] = set()
    depth = 0
    while len(seeds) < limit:
        progressed = False
        for leg in legs:
            if depth >= len(leg):
                continue
            progressed = True
            clause_id = clause_key(leg[depth])
            if not clause_id or clause_id in seen:
                continue
            seen.add(clause_id)
            seeds.append(clause_id)
            if len(seeds) >= limit:
                break
        if not progressed:
            break
        depth += 1
    return seeds


class GraphExpander:
    """Graph expansion leg, driven by ``[knowledge.graph.expand]``."""

    name = "graph"

    def __init__(
        self,
        settings: GraphExpandSettings,
        query_all: QueryAllFn,
        index: ClauseTextIndex,
    ) -> None:
        self.settings = settings
        self._query_all = query_all
        self._index = index

    def verify(self) -> dict[str, int]:
        """Fail loudly when the graph is unreachable.

        Mirrors ``MilvusStore.verify_schema``: a caller that intends to measure
        the graph leg must not silently end up measuring a chain where it never
        ran.  ``retrieve`` itself stays tolerant (production keeps answering),
        so the check has to be explicit.
        """
        rows = self._query_all(
            "MATCH (c:Clause) RETURN count(c) AS c", {}
        )
        return {"clauses": int(rows[0]["c"]) if rows else 0}

    def expand(self, seeds: Sequence[str], *, limit: int | None = None) -> list[KnowledgeHit]:
        """Return candidate clauses reachable from ``seeds`` in one or two hops.

        ``limit`` is the caller's total cap (the RRF pool budget); it can only
        tighten the configured ``max_results``, never raise it.
        """
        if not seeds or not self.settings.enabled:
            return []
        cap = self.settings.max_results
        if limit is not None:
            cap = min(cap, limit)
        if cap < 1:
            return []

        neighbors: list[Neighbor] = []
        params = {
            "seeds": list(seeds),
            "per_seed": self.settings.neighbors_per_seed,
            "min_shared": self.settings.min_shared_entities,
        }
        for kind, query in self._queries():
            rows = self._query_all(query, params)
            neighbors.extend(rank_neighbors(rows, kind=kind, max_results=cap))

        merged = _merge_kinds(neighbors)
        merged.sort(
            key=lambda n: (
                -n.weight,
                -n.seed_count,
                0 if KIND_CITES in n.kinds else 1,
                n.clause_id,
            )
        )

        hits: list[KnowledgeHit] = []
        for neighbor in merged[:cap]:
            hit = self._index.get(neighbor.clause_id)
            if hit is None:
                # The clause exists in the graph but has no snapshot row (e.g. a
                # stale clause node from an older chunking pass).  Dropped, not
                # emitted empty: an empty context block would be counted as
                # retrieved evidence and dilute the prompt.
                continue
            hits.append(
                hit.model_copy(
                    update={
                        "metadata_payload": {
                            **hit.metadata_payload,
                            GRAPH_EXPANDED_KEY: True,
                            GRAPH_KIND_KEY: "+".join(sorted(neighbor.kinds)),
                            GRAPH_WEIGHT_KEY: neighbor.weight,
                            GRAPH_SEEDS_KEY: neighbor.seed_count,
                        }
                    }
                )
            )
        return hits

    def _queries(self) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        if self.settings.use_cites:
            out.append((KIND_CITES, _CITES_QUERY))
        if self.settings.use_mentions:
            out.append((KIND_MENTIONS, _MENTIONS_QUERY))
        return out


def _merge_kinds(neighbors: Iterable[Neighbor]) -> list[Neighbor]:
    """Combine one clause's rows from both legs into a single candidate."""
    weight: dict[str, int] = {}
    kinds: dict[str, set[str]] = {}
    seeds: dict[str, int] = {}
    for neighbor in neighbors:
        weight[neighbor.clause_id] = weight.get(neighbor.clause_id, 0) + neighbor.weight
        kinds.setdefault(neighbor.clause_id, set()).update(neighbor.kinds)
        seeds[neighbor.clause_id] = max(
            seeds.get(neighbor.clause_id, 0), neighbor.seed_count
        )
    return [
        Neighbor(
            clause_id=clause_id,
            weight=weight[clause_id],
            kinds=frozenset(kinds[clause_id]),
            seed_count=seeds[clause_id],
        )
        for clause_id in weight
    ]
