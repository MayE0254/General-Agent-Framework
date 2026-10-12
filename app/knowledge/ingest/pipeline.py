"""End-to-end ingestion pipeline: scan -> parse -> chunk -> embed -> index.

Writes into two targets:

- **Milvus**  collection ``knowledge_chunks`` (vector, cosine).
- **Neo4j**   :doc: / :KnowledgeChunk: structural scaffold (MERGE semantics).

and exports a ``processed/chunks.jsonl`` snapshot so the BM25 keyword leg of
hybrid retrieval can be rebuilt without re-embedding the corpus.

Idempotence: chunk ids are stable sha1s of ``source_path + index + head``, and
the Milvus primary key is derived from that chunk id, so re-runs overwrite the
same rows instead of duplicating them; Neo4j uses MERGE on the same stable ids.
A progress marker (``processed/progress.json``) records already-ingested files
for resume / incremental sync.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.knowledge.base import KnowledgeHit
from app.knowledge.embeddings import LocalEmbeddingModel
from app.knowledge.graph.neo4j import Neo4jGraphStore
from app.knowledge.ingest.chunker import chunk_document
from app.knowledge.ingest.parsers import parse_document
from app.knowledge.vectorstore.milvus import MilvusStore

if TYPE_CHECKING:
    from app.core.settings import KnowledgeSettings

_ACCEPTED = {".docx", ".pdf", ".xlsx"}
# Office lock/temp artifacts (``~$foo.docx``) sit next to the real files with
# the hidden attribute set; they are not corpus content and must be skipped.
_IGNORED_PREFIXES = ("~$",)


def discover_files(corpus_dir: str, *, limit: int | None = None) -> list[Path]:
    """Accepted corpus files under ``corpus_dir`` in deterministic sorted order.

    Shared by the pipeline and the ingest CLI so the run's file universe (and
    therefore its progress denominator) is defined in exactly one place.
    """
    root = Path(corpus_dir)
    files = sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in _ACCEPTED
        and not p.name.startswith(_IGNORED_PREFIXES)
    )
    return files[:limit] if limit is not None else files


class IngestionPipeline:
    """Orchestrates parse -> chunk -> embed -> (milvus, neo4j, snapshot)."""

    def __init__(
        self,
        knowledge_settings: KnowledgeSettings,
        *,
        embedding: LocalEmbeddingModel,
        store: MilvusStore,
        graph: Neo4jGraphStore | None = None,
    ) -> None:
        self.settings = knowledge_settings
        self.embedding = embedding
        self.store = store
        self.graph = graph
        self.chunk_size = knowledge_settings.ingest.chunk_size_chars
        self.overlap = knowledge_settings.ingest.chunk_overlap_chars
        self.child_window = knowledge_settings.ingest.child_window_chars

    def run(
        self,
        corpus_dir: str,
        *,
        limit: int | None = None,
        output_dir: str = "data/knowledge/processed",
        batch_size: int = 64,
        persist_every: int = 25,
    ) -> dict[str, Any]:
        """Execute ingestion over one corpus directory; returns a report dict.

        Crash safety: progress is persisted every ``persist_every`` files and
        again in a ``finally`` block, so an interrupt (Ctrl-C, power loss)
        resumes instead of re-embedding the whole corpus.  Vector work and
        graph work carry **separate** done-markers: a Neo4j outage never forces
        a re-embed, and a graph gap can be repaired by re-running (the graph
        pass is cheap and MERGE-idempotent) without touching Milvus.
        """
        root = Path(corpus_dir)
        all_files = discover_files(corpus_dir)
        files = all_files[:limit] if limit is not None else all_files

        progress_path = Path(output_dir) / "progress.json"
        done: set[str] = set()
        graph_done: set[str] = set()
        if progress_path.exists():
            saved = json.loads(progress_path.read_text(encoding="utf-8"))
            done = set(saved.get("done", []))
            graph_done = set(saved.get("graph_done", []))

        stats = {
            "files": 0, "chunks": 0, "skipped": 0, "failed": 0,
            "graph_files": 0, "graph_failed": 0, "graph_links": 0,
            "snapshot_missing": 0, "errors": [],
        }
        since_persist = 0

        def persist() -> None:
            """Atomic write so a crash mid-write cannot corrupt the marker."""
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = progress_path.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps(
                    {
                        "done": sorted(done),
                        "graph_done": sorted(graph_done),
                        "total": stats["chunks"],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            tmp.replace(progress_path)

        try:
            for fp in files:
                rel = str(fp.relative_to(root)).replace("\\", "/")
                need_vec = rel not in done
                need_graph = self.graph is not None and rel not in graph_done
                if not need_vec and not need_graph:
                    stats["skipped"] += 1
                    continue

                try:
                    parsed = parse_document(str(fp), str(root))
                    chunks = chunk_document(
                        parsed,
                        chunk_size=self.chunk_size,
                        overlap=self.overlap,
                        child_window=self.child_window,
                    )
                except Exception as exc:  # keep going on a bad file
                    stats["failed"] += 1
                    stats["errors"].append(f"parse {rel}: {exc}")
                    continue

                # Counted here, before the vector/graph split: a graph-only
                # rebuild parses and chunks every file but embeds none, and the
                # marker's ``total`` must still report the corpus size instead
                # of collapsing to 0.
                stats["chunks"] += len(chunks)

                if need_vec:
                    # Embed in a scalable batch: chunk texts -> vectors.  Any
                    # failure here leaves the file un-marked so a re-run retries
                    # it instead of silently dropping its chunks.
                    try:
                        texts = [c.text for c in chunks]
                        vectors: list[list[float]] = []
                        for i in range(0, len(texts), batch_size):
                            vectors.extend(
                                self.embedding.embed_documents(texts[i : i + batch_size])
                            )
                        # Flush this file's chunks to Milvus (clear per file so a
                        # later file never re-sends earlier rows). Row identity
                        # is the content-derived chunk id, not a counter.
                        rows = [
                            {
                                "chunk_id": c.id,
                                "embedding": vec,
                                "text": c.text,
                                "title": c.title,
                                "source": c.source_rel,
                                "file_no": c.file_no,
                                "doc_type": c.doc_type,
                                "section": c.section,
                                "parent_id": c.parent_id,
                                "parent_text": c.parent_text,
                            }
                            for c, vec in zip(chunks, vectors)
                        ]
                        self.store.upsert_chunks(rows)
                    except Exception as exc:
                        stats["failed"] += 1
                        stats["errors"].append(f"index {rel}: {exc}")
                        continue
                    done.add(rel)
                    stats["files"] += 1

                if need_graph and chunks:
                    try:
                        self.graph.ingest_document(
                            doc_no=parsed.file_no,
                            title=parsed.title,
                            doc_type=parsed.doc_type,
                            source_rel=parsed.source_rel,
                            chunks=chunks,
                        )
                        graph_done.add(rel)
                        stats["graph_files"] += 1
                    except Exception as exc:
                        stats["graph_failed"] += 1
                        stats["errors"].append(f"graph {rel}: {exc}")

                since_persist += 1
                if since_persist >= persist_every:
                    persist()
                    since_persist = 0
        finally:
            persist()

        # BM25 snapshot mirrors the accumulated ingested set (``done``), so it
        # always matches what is actually in Milvus and both retrieval legs
        # search the same document universe.  The same pass yields each
        # document's graph id in corpus order, which is exactly what the NEXT
        # chain needs, so the corpus is parsed only once here.
        done_files = [
            p for p in all_files
            if str(p.relative_to(root)).replace("\\", "/") in done
        ]
        doc_ids = self._write_snapshot(output_dir, done_files, root)
        stats["snapshot_missing"] = len(done_files) - len(doc_ids)

        if self.graph is not None:
            ordered = [
                doc_id for rel, doc_id in doc_ids.items() if rel in graph_done
            ]
            try:
                stats["graph_links"] = self.graph.link_sequence(ordered)
            except Exception as exc:
                stats["errors"].append(f"graph link_sequence: {exc}")

        stats["total_files"] = len(files)
        stats["total_chunks"] = stats["chunks"]
        return stats

    def _write_snapshot(
        self, output_dir: str, files: list[Path], root: Path
    ) -> dict[str, str]:
        """Write a JSONL of KnowledgeHit for rebuilding the BM25 index.

        ``files`` is the accumulated ingested set (see ``progress.json``), which
        keeps the BM25 leg aligned with the Milvus vector leg so their
        candidates are directly comparable.  Returns ``{relative path: graph
        document id}`` in corpus order; a path missing from the mapping is one
        that failed to re-parse here (and so contributes no BM25 rows).
        """
        out = Path(output_dir) / "chunks.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        doc_ids: dict[str, str] = {}
        with out.open("w", encoding="utf-8") as fh:
            for fp in files:
                rel = str(fp.relative_to(root)).replace("\\", "/")
                try:
                    parsed = parse_document(str(fp), str(root))
                    chunks = chunk_document(
                        parsed,
                        chunk_size=self.chunk_size,
                        overlap=self.overlap,
                        child_window=self.child_window,
                    )
                except Exception:
                    continue
                doc_ids[rel] = Neo4jGraphStore.doc_id_for(parsed.source_rel)
                for c in chunks:
                    fh.write(
                        json.dumps(
                            {
                                "id": c.id,
                                "title": c.title,
                                "content": c.text,
                                "source": c.source_rel,
                                "score": 0.0,
                                "parent_id": c.parent_id,
                                "parent_text": c.parent_text,
                                "metadata_payload": c.metadata_payload,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
        return doc_ids


def load_chunk_snapshot(path: str) -> list[KnowledgeHit]:
    """Load a ``chunks.jsonl`` snapshot into KnowledgeHit objects (for BM25)."""
    hits: list[KnowledgeHit] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            hits.append(
                KnowledgeHit(
                    id=rec["id"],
                    title=rec.get("title", ""),
                    content=rec["content"],
                    score=float(rec.get("score", 0.0)),
                    source=rec.get("source", ""),
                    parent_id=rec.get("parent_id", "") or "",
                    parent_text=rec.get("parent_text", "") or "",
                    metadata_payload=rec.get("metadata_payload", {}),
                )
            )
    return hits