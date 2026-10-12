"""Shared data structures for the knowledge ingestion pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chunk:
    """One retrievable unit produced by the chunker.

    Carries enough provenance to (a) write to Milvus together with its
    embedding, (b) anchor to a Neo4j document node, and (c) be cited in
    retrieval results.
    """

    id: str                       # stable deterministic id (sha1 of source+index)
    text: str
    source_path: str              # absolute source file path
    source_rel: str               # path relative to corpus root (display / cite)
    title: str                    # legal name derived from the source filename
    file_no: str                  # sequence number of the source file ("" when none)
    doc_type: str                 # inferred kind (law/notice/...)
    chunk_index: int              # 0-based position within the document
    section: str = ""             # nearest "第X条" / heading captured by chunker
    # Parent/child indexing: ``text`` is the retrievable child window, while
    # ``parent_id`` names the enclosing clause ("第X条") it was cut from.
    # ``parent_text`` is the whole clause and is only populated when the clause
    # had to be split -- a single-child parent would just duplicate ``text``
    # across every Milvus row, so it stays empty and readers fall back to
    # ``text``.  Parents are never indexed for retrieval; they exist purely to
    # give callers whole-clause context and to let sibling children collapse.
    parent_id: str = ""
    parent_text: str = ""
    metadata_payload: dict = field(default_factory=dict)


@dataclass
class IngestedDocResult:
    """Per-document ingestion outcome (for progress logging / resume)."""

    source_rel: str
    title: str
    num_chunks: int
    ok: bool
    error: str | None = None