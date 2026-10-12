"""Neo4j knowledge-graph scaffolding for the fiscal/tax corpus.

Three layers are written here:

1. **Skeleton** (Phase-P2) -- each source document becomes a ``Document`` node
   (with file_no / title / doc_type), every chunk becomes a chunk node linked
   by ``CONTAINS``.  A post-pass (``link_sequence``) chains documents in corpus
   order with ``NEXT``.  Written during ingestion, one chunk at a time.
2. **Structural layer** (``write_structure``) -- ``Clause`` nodes (one per
   chunker parent) plus the four relations the corpus states about itself:
   ``HAS_CLAUSE``, ``CITES``, ``IMPLEMENTS``, ``REFERENCES``.  These are
   derived by ``app.knowledge.graph.extract`` from the text alone, so they are
   exact, free and re-runnable; the pass reads the chunk snapshot rather than
   the raw files, which is why it lives in ``scripts/build_graph.py`` instead
   of the ingestion loop.
3. **Semantic entity layer** (``write_entities``) -- ``Entity`` nodes and the
   ``MENTIONS`` edges from clause to entity, read out of each document by an
   LLM against a closed type list (``app.knowledge.graph.entities``).  This is
   the only layer that costs money and the only one whose output a rule cannot
   reproduce, so it is written by ``scripts/build_entities.py``, separately
   from the deterministic pass.

What the entity layer is *for*: two clauses that share a subject without
naming each other (国发〔2022〕8号 and 国发〔2023〕13号 both cover 3 岁以下婴幼儿
照护扣除, and neither cites the other).  That is a two-hop
``Clause -> Entity -> Clause`` traversal, so no clause-to-clause edge is
materialised -- a modelled "related clauses" relation would be O(n^2) junk.

Node labels and the containment relationship name are read from
``[knowledge.graph]`` (``chunk_entity_label`` / ``relation_label``) so the
config and the written graph cannot drift apart.  The structural and entity
labels are *not* configurable: they are part of the graph contract, and a typo
in a config file would silently produce an unconnected graph.  Clauses are also
stored without clause text -- the graph is for connectivity, and the text is
already in Milvus, reachable through ``chunk_ids``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.core.settings import GraphSettings
    from app.knowledge.graph.entities import EntityExtraction
    from app.knowledge.graph.extract import CitationEdge, ClauseRecord, StructuralExtraction
    from app.knowledge.ingest.models import Chunk


class Neo4jGraphStore:
    """Write-only scaffold anchored on document/chunk/clause nodes."""

    _DOC_LABEL = "Document"
    _CHUNK_LABEL = "KnowledgeChunk"
    _CONTAINS = "CONTAINS"
    _NEXT = "NEXT"
    _CLAUSE_LABEL = "Clause"
    _HAS_CLAUSE = "HAS_CLAUSE"
    _CITES = "CITES"
    _IMPLEMENTS = "IMPLEMENTS"
    _REFERENCES = "REFERENCES"
    _ENTITY_LABEL = "Entity"
    _MENTIONS = "MENTIONS"

    def __init__(self, settings: GraphSettings) -> None:
        self.settings = settings
        self._driver = None

    @property
    def chunk_label(self) -> str:
        """Chunk node label, from ``[knowledge.graph].chunk_entity_label``."""
        return self.settings.chunk_entity_label or self._CHUNK_LABEL

    @property
    def contains_rel(self) -> str:
        """Document→chunk relationship, from ``[knowledge.graph].relation_label``."""
        return self.settings.relation_label or self._CONTAINS

    def _get_driver(self):
        if self._driver is not None:
            return self._driver
        from neo4j import GraphDatabase

        self._driver = GraphDatabase.driver(
            self.settings.uri,
            auth=(self.settings.user, self.settings.password),
            connection_timeout=10,
        )
        return self._driver

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def ensure_constraints(self) -> None:
        with self._get_driver().session(database=self.settings.database) as s:
            s.run(
                f"CREATE CONSTRAINT {self._DOC_LABEL}_doc_id IF NOT EXISTS "
                f"FOR (d:{self._DOC_LABEL}) REQUIRE d.doc_id IS UNIQUE"
            )
            s.run(
                f"CREATE CONSTRAINT {self.chunk_label}_chunk_id IF NOT EXISTS "
                f"FOR (c:{self.chunk_label}) REQUIRE c.chunk_id IS UNIQUE"
            )
            s.run(
                f"CREATE CONSTRAINT {self._CLAUSE_LABEL}_clause_id IF NOT EXISTS "
                f"FOR (c:{self._CLAUSE_LABEL}) REQUIRE c.clause_id IS UNIQUE"
            )
            s.run(
                f"CREATE CONSTRAINT {self._ENTITY_LABEL}_entity_id IF NOT EXISTS "
                f"FOR (e:{self._ENTITY_LABEL}) REQUIRE e.entity_id IS UNIQUE"
            )
            # Non-unique: the type is a property, not a label (five labels would
            # force every expansion query to union five MATCH clauses), so the
            # type filter needs its own index.
            s.run(
                f"CREATE INDEX {self._ENTITY_LABEL}_type IF NOT EXISTS "
                f"FOR (e:{self._ENTITY_LABEL}) ON (e.type)"
            )

    def counts(self) -> dict[str, int]:
        """Node count per label — used for ingestion/verification reporting."""
        out: dict[str, int] = {}
        labels = (
            self._DOC_LABEL, self.chunk_label, self._CLAUSE_LABEL,
            self._ENTITY_LABEL,
        )
        with self._get_driver().session(database=self.settings.database) as s:
            for label in labels:
                rec = s.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()
                out[label] = int(rec["c"]) if rec else 0
        return out

    def query_one(self, query: str, params: dict | None = None) -> dict | None:
        """Run one read-only Cypher query and return its single record.

        Exists so verification scripts can assert on the graph in Cypher without
        reaching into the driver; write paths still go through the typed methods
        above.
        """
        with self._get_driver().session(database=self.settings.database) as s:
            rec = s.run(query, params or {}).single()
        return dict(rec) if rec else None

    def query_all(self, query: str, params: dict | None = None) -> list[dict]:
        """Run one read-only Cypher query and return every record.

        The multi-row sibling of ``query_one``, used by the graph expansion leg
        (``app.knowledge.graph.expand``) to pull a batch of neighbours in a
        single round trip: one query per leg per retrieval, rather than one per
        seed clause.
        """
        with self._get_driver().session(database=self.settings.database) as s:
            return [dict(rec) for rec in s.run(query, params or {})]

    def relation_counts(self) -> dict[str, int]:
        """Relationship count per type — the structural layer's audit trail.

        Kept separate from ``counts`` so a report cannot confuse "nodes written"
        with "edges written"; an empty ``CITES`` is the whole difference between
        a graph and a pile of documents.
        """
        rels = (
            self.contains_rel, self._NEXT, self._HAS_CLAUSE,
            self._CITES, self._IMPLEMENTS, self._REFERENCES, self._MENTIONS,
        )
        out: dict[str, int] = {}
        with self._get_driver().session(database=self.settings.database) as s:
            for rel in rels:
                rec = s.run(f"MATCH ()-[r:{rel}]->() RETURN count(r) AS c").single()
                out[rel] = int(rec["c"]) if rec else 0
        return out

    def reset(self) -> int:
        """Delete this scaffold's nodes (documents + chunks + clauses) for a clean rebuild.

        Scoped to the labels this store owns, so unrelated data in the same
        database is untouched.  Destructive by design: the corpus under
        ``data/knowledge/raw`` is the source of truth.  Used by the ingest
        CLI's ``--reset``; the ingestion path itself never calls it.
        """
        with self._get_driver().session(database=self.settings.database) as s:
            rec = s.run(
                "MATCH (n) WHERE n:" + self._DOC_LABEL
                + " OR n:" + self.chunk_label
                + " OR n:" + self._CLAUSE_LABEL
                + " OR n:" + self._ENTITY_LABEL + " "
                "DETACH DELETE n RETURN count(*) AS c"
            ).single()
        return int(rec["c"]) if rec else 0

    # ------------------------------------------------------------------
    # Structural layer
    # ------------------------------------------------------------------
    def reset_structure(self) -> dict[str, int]:
        """Delete only the structural layer, keeping documents and chunks.

        Rebuilding the structural layer is meant to be cheap (no LLM, no
        embedding), so it needs a way to drop stale nodes without forcing a
        re-ingest: ``Clause`` nodes take ``HAS_CLAUSE`` / ``CITES`` with them,
        and the two document-to-document relations are deleted explicitly
        because they touch no clause node.

        Returns the number of clause nodes and document edges removed.
        """
        with self._get_driver().session(database=self.settings.database) as s:
            clauses = s.run(
                f"MATCH (c:{self._CLAUSE_LABEL}) DETACH DELETE c RETURN count(*) AS c"
            ).single()
            edges = s.run(
                f"MATCH ()-[r:{self._IMPLEMENTS}|{self._REFERENCES}]->() "
                "DELETE r RETURN count(*) AS c"
            ).single()
        return {
            "clauses": int(clauses["c"]) if clauses else 0,
            "doc_edges": int(edges["c"]) if edges else 0,
        }

    def write_structure(
        self, extraction: StructuralExtraction, *, batch_size: int = 500
    ) -> dict[str, int]:
        """Persist the deterministic structural layer and report what was written.

        Clause nodes go first: ``IMPLEMENTS`` / ``REFERENCES`` join two
        ``Document`` nodes, and the document upsert happens as a side effect of
        the clause pass, so writing edges first would drop them.  The returned
        counts are *distinct* graph objects written, not occurrences found, so
        they can be compared against ``relation_counts()`` after the run.
        """
        written = {
            "clauses": self._write_clauses(extraction.clauses, batch_size),
            "citations": self._write_citations(extraction.citations, batch_size),
            "implements": self._write_doc_edges(
                extraction.implements, self._IMPLEMENTS, batch_size
            ),
            "references": self._write_doc_edges(
                extraction.references, self._REFERENCES, batch_size
            ),
        }
        return written

    def _write_clauses(self, clauses: list[ClauseRecord], batch_size: int) -> int:
        rows = [
            {
                "clause_id": c.clause_id,
                "doc_id": c.doc_id,
                "title": c.title,
                "source_rel": c.source_rel,
                "section": c.section,
                "article": c.article,
                "chunk_ids": list(c.chunk_ids),
                "char_len": len(c.text),
            }
            for c in clauses
        ]
        query = (
            "UNWIND $rows AS row "
            "MERGE (d:" + self._DOC_LABEL + " {doc_id: row.doc_id}) "
            "ON CREATE SET d.title = row.title, d.source_rel = row.source_rel "
            "MERGE (c:" + self._CLAUSE_LABEL + " {clause_id: row.clause_id}) "
            "SET c.doc_id = row.doc_id, c.title = row.title, "
            "    c.source_rel = row.source_rel, c.section = row.section, "
            "    c.article = row.article, c.chunk_ids = row.chunk_ids, "
            "    c.char_len = row.char_len "
            "MERGE (d)-[:" + self._HAS_CLAUSE + "]->(c)"
        )
        self._run_batched(query, rows, batch_size)
        return len(rows)

    def _write_citations(self, citations: list[CitationEdge], batch_size: int) -> int:
        """Write ``CITES`` edges, one per clause pair.

        Several occurrences of the same pair collapse into a single edge (with
        ``occurrences`` counting them): traversal wants connectivity, not
        multiplicity, and a pair asserting 第(一)项 and 第(二)项 of the same
        target is still one hop.  The first occurrence supplies the descriptive
        properties.
        """
        merged: dict[tuple[str, str], dict] = {}
        for edge in citations:
            key = (edge.src_clause_id, edge.dst_clause_id)
            row = merged.get(key)
            if row is None:
                merged[key] = {
                    "src": edge.src_clause_id,
                    "dst": edge.dst_clause_id,
                    "raw": edge.props.get("raw", ""),
                    "law_token": edge.props.get("law_token", ""),
                    "article": edge.props.get("article"),
                    "paragraph": edge.props.get("paragraph"),
                    "item": edge.props.get("item"),
                    "cross_doc": bool(edge.props.get("cross_doc", False)),
                    "occurrences": 1,
                }
            else:
                row["occurrences"] += 1
        rows = list(merged.values())
        query = (
            "UNWIND $rows AS row "
            "MATCH (a:" + self._CLAUSE_LABEL + " {clause_id: row.src}) "
            "MATCH (b:" + self._CLAUSE_LABEL + " {clause_id: row.dst}) "
            "MERGE (a)-[r:" + self._CITES + "]->(b) "
            "SET r.raw = row.raw, r.law_token = row.law_token, "
            "    r.article = row.article, r.paragraph = row.paragraph, "
            "    r.item = row.item, r.cross_doc = row.cross_doc, "
            "    r.occurrences = row.occurrences"
        )
        self._run_batched(query, rows, batch_size)
        return len(rows)

    def _write_doc_edges(
        self, pairs: list[tuple[str, str]], rel: str, batch_size: int
    ) -> int:
        unique = list(dict.fromkeys(
            (a, b) for a, b in pairs if a and b and a != b
        ))
        rows = [{"a": a, "b": b} for a, b in unique]
        query = (
            "UNWIND $rows AS row "
            "MATCH (a:" + self._DOC_LABEL + " {doc_id: row.a}) "
            "MATCH (b:" + self._DOC_LABEL + " {doc_id: row.b}) "
            "MERGE (a)-[:" + rel + "]->(b)"
        )
        self._run_batched(query, rows, batch_size)
        return len(rows)

    # ------------------------------------------------------------------
    # Semantic entity layer
    # ------------------------------------------------------------------
    def reset_entities(self) -> dict[str, int]:
        """Delete the entity layer, keeping documents, chunks and clauses.

        ``MENTIONS`` hangs off the entity node, so ``DETACH DELETE`` removes
        both in one pass.  Scoped this way so re-running the LLM pass (the slow,
        paid one) never forces a rebuild of the deterministic layers.
        """
        with self._get_driver().session(database=self.settings.database) as s:
            entities = s.run(
                f"MATCH (e:{self._ENTITY_LABEL}) DETACH DELETE e RETURN count(*) AS c"
            ).single()
        return {"entities": int(entities["c"]) if entities else 0}

    def write_entities(
        self, extraction: EntityExtraction, *, batch_size: int = 500
    ) -> dict[str, int]:
        """Persist one document's entity layer; returns distinct objects written.

        ``MENTIONS`` is merged on the bare pattern with its properties set
        afterwards: putting ``extractor`` inside the MERGE pattern would create
        a *second* edge the moment a re-run used a different model (e.g. after
        a quota fallback), duplicating every hop.

        A mention whose clause is absent from the graph matches nothing and is
        skipped, so the caller must cross-check the summed return value against
        ``relation_counts()`` -- that comparison is the check, not the return.
        """
        entities: dict[str, dict] = {}
        edges: dict[tuple[str, str], dict] = {}
        for mention in extraction.mentions:
            entities.setdefault(
                mention.entity_id,
                {
                    "entity_id": mention.entity_id,
                    "name": mention.name,
                    "type": mention.type_key,
                },
            )
            edges.setdefault(
                (mention.clause_id, mention.entity_id),
                {
                    "clause_id": mention.clause_id,
                    "entity_id": mention.entity_id,
                    "extractor": mention.extractor,
                },
            )
        if not entities:
            return {"entities": 0, "mentions": 0}
        self._run_batched(
            "UNWIND $rows AS row "
            "MERGE (e:" + self._ENTITY_LABEL + " {entity_id: row.entity_id}) "
            "SET e.name = row.name, e.type = row.type",
            list(entities.values()),
            batch_size,
        )
        self._run_batched(
            "UNWIND $rows AS row "
            "MATCH (c:" + self._CLAUSE_LABEL + " {clause_id: row.clause_id}) "
            "MATCH (e:" + self._ENTITY_LABEL + " {entity_id: row.entity_id}) "
            "MERGE (c)-[r:" + self._MENTIONS + "]->(e) "
            "SET r.extractor = row.extractor",
            list(edges.values()),
            batch_size,
        )
        return {"entities": len(entities), "mentions": len(edges)}

    def _run_batched(self, query: str, rows: list[dict], batch_size: int) -> None:
        """Run one UNWIND query over ``rows`` in batches; a no-op for empty input."""
        if not rows or batch_size < 1:
            return
        with self._get_driver().session(database=self.settings.database) as s:
            for start in range(0, len(rows), batch_size):
                s.run(query, {"rows": rows[start:start + batch_size]})

    @staticmethod
    def doc_id_for(source_rel: str) -> str:
        """Stable ``Document`` node id — keyed by the corpus-relative path.

        The path is unique per file; ``file_no`` is not (two corpus files can
        share a sequence number, e.g. ``261_*.docx``), and keying on it would
        collapse distinct documents into one node and produce a ``NEXT``
        self-loop.  Separators are normalised so the id does not depend on the
        host OS.
        """
        return "doc:" + source_rel.replace("\\", "/")

    def ingest_document(self, doc_no: str, title: str, doc_type: str,
                        source_rel: str, chunks: list[Chunk]) -> None:
        """Upsert a document node and its chunk children under MERGE semantics.

        Nodes are keyed by stable ids (the relative path, ``chunk.id``) so a
        re-run updates rather than duplicates.
        """
        if not chunks:
            return
        doc_id = self.doc_id_for(source_rel)
        for chunk in chunks:
            self._write_doc_chunk(
                doc_id=doc_id,
                doc_no=doc_no,
                title=title,
                doc_type=doc_type,
                source_rel=source_rel,
                chunk=chunk,
            )

    def _write_doc_chunk(self, *, doc_id, doc_no, title, doc_type, source_rel, chunk) -> None:
        with self._get_driver().session(database=self.settings.database) as s:
            s.run(
                "MERGE (d:" + self._DOC_LABEL + " {doc_id: $doc_id}) "
                "ON CREATE SET d.title=$title, d.file_no=$doc_no, d.doc_type=$doc_type, "
                "              d.source_rel=$source_rel "
                "ON MATCH SET d.title=$title, d.file_no=$doc_no, d.doc_type=$doc_type, "
                "            d.source_rel=$source_rel "
                "MERGE (c:" + self._CHUNK_LABEL + " {chunk_id: $chunk_id}) "
                "ON CREATE SET c.text=$text, c.section=$section, c.chunk_index=$ci, "
                "              c.doc_id=$doc_id "
                "ON MATCH SET c.text=$text, c.section=$section, c.chunk_index=$ci, "
                "            c.doc_id=$doc_id "
                "MERGE (d)-[r:" + self.contains_rel + "]->(c)",
                {
                    "doc_id": doc_id,
                    "title": title,
                    "doc_no": doc_no,
                    "doc_type": doc_type,
                    "source_rel": source_rel,
                    "chunk_id": chunk.id,
                    "text": chunk.text,
                    "section": chunk.section,
                    "ci": chunk.chunk_index,
                },
            )

    def link_sequence(self, doc_ids: list[str]) -> int:
        """Chain documents in corpus order with ``NEXT`` relationships.

        This is the one piece of cross-document structure derivable without an
        LLM, and it is what makes the skeleton walkable (e.g. follow a law to
        the regulation that implements it).  A single UNWIND keeps the 4189-doc
        pass to one round trip; ``MERGE`` keeps re-runs idempotent.  Docs that
        are missing from the graph are skipped silently (their pair simply does
        not match) so a partial graph still links what it can.

        Returns the number of ``NEXT`` relationships written.
        """
        if len(doc_ids) < 2:
            return 0
        pairs = [
            {"a": doc_ids[i], "b": doc_ids[i + 1]}
            for i in range(len(doc_ids) - 1)
        ]
        with self._get_driver().session(database=self.settings.database) as s:
            rec = s.run(
                "UNWIND $pairs AS p "
                "MATCH (a:" + self._DOC_LABEL + " {doc_id: p.a}) "
                "MATCH (b:" + self._DOC_LABEL + " {doc_id: p.b}) "
                "MERGE (a)-[:" + self._NEXT + "]->(b) "
                "RETURN count(*) AS c",
                {"pairs": pairs},
            ).single()
        return int(rec["c"]) if rec else 0