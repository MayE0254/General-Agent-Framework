"""Chinese-aware chunking for the fiscal/tax regulatory corpus.

Strategy is structural first: legal/regulatory text is dominated by
"第X条" clauses and chapter headings, so we treat those as preferred chunk
boundaries.  When a document has no clause markers (e.g. an Excel table or a
short notice) we fall back to plain sentence-window chunking.

Chunking is **two-level (parent/child)**, which is what makes "retrieve small,
return big" possible downstream:

- **Parent** = one clause ("第X条"), or a sentence-boundary slice of a clause
  that exceeds ``chunk_size_chars``.  Parents partition the clause with no
  overlap -- overlap is a retrieval concern and belongs to the child level,
  where duplicated text would otherwise burn candidate slots.
- **Child** = a ``child_window_chars`` sentence window inside one parent.  This
  is the unit that gets embedded, written to Milvus and BM25-scored, so ranking
  sees a small, focused passage instead of a whole clause.  A child carries
  ``parent_id`` (its clause) and ``parent_text`` (the whole clause) so the
  retrieval layer can hand back the full clause.

Parents are deliberately *not* indexed: indexing them would recreate exactly
the "long statute monopolises the candidate pool" problem the child window is
meant to avoid.

Every chunk inherits provenance (source path, title, file_no, doc_type) from
its source document, so retrieval results can be cited precisely.
"""
from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

from app.knowledge.ingest.models import Chunk
from app.knowledge.ingest.parsers import ParsedDocument

# Clause headers: 第(一|二|...|一百零一)条   or  第 1 条 / 第1条
_CLAUSE_RE = re.compile(r"^\s*(第\s*[0-9一二三四五六七八九十百千万零〇两]*\s*条)\s*.*$")
# Chapter / section headings, e.g. 第一章 总则 / 第一节 一般规定
_SECTION_RE = re.compile(r"^\s*(第\s*[0-9一二三四五六七八九十百千万零〇两]*\s*[章节])\s*(.*)$")
# Sentence enders used when splitting an over-long clause.
_SENT_END_RE = re.compile(r"[。！？；;]")
_ENDER_CHARS = "。！？；;"
_NUM_RE = re.compile(r"[0-9]+|[一二三四五六七八九十百千万零〇两]+")


def chunk_document(
    doc: ParsedDocument,
    *,
    chunk_size: int,
    overlap: int,
    child_window: int,
) -> list[Chunk]:
    """Cut one parsed document into ordered, provenance-carrying chunks.

    Returns only the *child* chunks (the retrievable units); each one points at
    its parent clause via ``parent_id`` / ``parent_text``.  ``chunk_size``
    bounds the parent, ``child_window`` bounds the child, so a ``child_window``
    larger than ``chunk_size`` would collapse every clause to a single child
    and silently disable the two-level behaviour -- that is rejected rather
    than tolerated.
    """
    if child_window > chunk_size:
        raise ValueError(
            f"child_window ({child_window}) must not exceed chunk_size "
            f"({chunk_size}): every clause would degenerate to a single child "
            f"and parent/child retrieval would silently become a no-op."
        )

    paragraphs = doc.paragraphs
    clauses: list[tuple[str, str]] = []  # (section_label, clause_text)

    current_section = "全文"
    buffer: list[str] = []
    current_label = "全文"

    def flush() -> None:
        text = " ".join(buffer).strip()
        if text:
            clauses.append((current_label, text))

    for line in paragraphs:
        stripped = line.strip()
        if not stripped:
            continue
        sec_match = _SECTION_RE.match(stripped)
        if sec_match and _is_short_heading(stripped):
            current_section = stripped
            continue
        clause_match = _CLAUSE_RE.match(stripped)
        if clause_match and len(stripped) < 120:
            # New clause boundary: flush previous, use only the clause number
            # as the section label (not the full clause text).
            flush()
            current_label = clause_match.group(1).strip()
            buffer = [stripped]
            continue
        buffer.append(stripped)

    flush()

    chunks: list[Chunk] = []
    child_index = 0
    parent_index = 0
    for label, text in clauses:
        # Parents partition the clause (no overlap).
        for parent_text in _split_text(text, chunk_size=chunk_size, overlap=0):
            parent_id = _parent_id(doc.source_path, parent_index)
            parent_index += 1
            children = _split_text(
                parent_text, chunk_size=child_window, overlap=overlap
            )
            for piece in children:
                chunks.append(
                    Chunk(
                        id=_chunk_id(doc.source_path, child_index, piece),
                        text=piece,
                        source_path=doc.source_path,
                        source_rel=doc.source_rel,
                        title=doc.title,
                        file_no=doc.file_no,
                        doc_type=doc.doc_type,
                        chunk_index=child_index,
                        section=label,
                        parent_id=parent_id,
                        # Only materialise the parent text when this child is
                        # really a fragment of it (see Chunk.parent_text).
                        parent_text="" if piece == parent_text else parent_text,
                        metadata_payload={
                            "section": label,
                            "parent_id": parent_id,
                            "parent_char_len": len(parent_text),
                            "paragraph_count": len(_split_sentences(piece)),
                            "char_len": len(piece),
                        },
                    )
                )
                child_index += 1
    return chunks


def _is_short_heading(line: str) -> bool:
    """A section/clause heading is short; a long line just starting with 第...章
    is likely body text and must not be swallowed."""
    return len(line) <= 60


def _split_text(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    """Split one clause/block into pieces, preferring sentence boundaries."""
    if len(text) <= chunk_size:
        return [text]

    sentences = _split_sentences(text)
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= chunk_size or not current:
            current = f"{current}\n{sentence}".strip()
        else:
            pieces.append(current)
            current = current[-overlap:] if overlap > 0 else ""  # carry tail
            # Make sure a single huge sentence is hard-cut even if longer than chunk_size
            while len(sentence) > chunk_size:
                pieces.append(sentence[:chunk_size])
                sentence = sentence[chunk_size - overlap:] if overlap > 0 else sentence[chunk_size:]
                if len(sentence) <= chunk_size:
                    current = sentence
                    sentence = ""
            if sentence:
                current = f"{current}\n{sentence}".strip()
    if current.strip():
        pieces.append(current)
    return pieces


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on zh punctuation, keeping the marker.

    The ender is retained rather than dropped.  Chunking runs twice (clause ->
    parents, parent -> children) and the parent pass re-joins its fragments with
    ``\\n``; if the marker were discarded there, the child pass would find no
    sentence boundary in a split parent and silently return it whole, making
    ``child_window`` a no-op for exactly the long clauses it exists for.
    """
    sentences: list[str] = []
    start = 0
    for match in _SENT_END_RE.finditer(text):
        segment = text[start:match.end()].strip()
        if segment and segment.strip(_ENDER_CHARS):
            sentences.append(segment)
        elif segment and sentences:
            # A run of consecutive enders (e.g. "……。。") belongs to the
            # sentence before it rather than forming its own one-char segment.
            sentences[-1] += segment
        start = match.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def _chunk_id(source_path: str, index: int, piece: str) -> str:
    raw = f"{source_path}#{index}#{piece[:64]}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def _parent_id(source_path: str, parent_index: int) -> str:
    """Stable id of a parent clause.

    Namespaced with a ``p`` prefix so it can never collide with a child id,
    which is derived from a plain integer child index.
    """
    return _chunk_id(source_path, f"p{parent_index}", "")