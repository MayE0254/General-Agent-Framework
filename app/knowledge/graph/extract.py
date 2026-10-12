"""Deterministic structural extraction for the fiscal/tax knowledge graph.

The graph's first layer is **structural, not semantic**: it records what the
corpus literally says about itself -- which clause cites which other clause,
which regulation implements which law, which document names which other
document.  No LLM is involved, so this layer is exact, free, and re-runnable.

That is not a shortcut.  Measured on the live corpus
(``data/knowledge/processed/chunks.jsonl``), the bridges the multi-hop
evaluation set actually needs are textual:

* 企税法第二十七条 reads ``（五）本法第三条第三款规定的所得`` -- an explicit
  cross-reference, resolvable inside the document;
* 实施条例第九十一条 reads ``企业所得税法第二十七条第(一)项`` -- a law name
  without 书名号 and with half-width parentheses, resolvable across documents;
* 国务院关于提高个人所得税有关专项附加扣除标准的通知 reads
  ``依据《中华人民共和国个人所得税法》有关规定`` -- document-level citation.

So the edges this module produces are read off the text, never inferred.  What
they cannot express is exactly what the semantic layer is for: two notices that
share a *subject* without citing each other (国发〔2022〕8号 与 国发〔2023〕13号
both cover 3 岁以下婴幼儿照护扣除, and neither names the other).

Everything here is a pure function over strings/dicts so it can be unit-tested
without Neo4j, and the build script stays a thin orchestrator.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

# --------------------------------------------------------------------------
# Chinese numerals
# --------------------------------------------------------------------------
_CN_DIGITS = {
    "零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}
_CN_UNITS = {"十": 10, "百": 100, "千": 1000}
_CN_BIG = {"万": 10000}
# Article numbers may be written either way in this corpus ("第二十七条" for
# statutes, "第 10 条" in the odd notice).
_ARTICLE_CHARS = "0-9一二三四五六七八九十百千万零〇两"
_ARTICLE_CLASS = f"[{_ARTICLE_CHARS}]"


def cn_to_int(text: str) -> int | None:
    """Parse an Arabic or Chinese numeral ("27", "二十七") into an int.

    Returns ``None`` for anything that does not parse, so a malformed article
    reference is dropped rather than silently becoming article 0.
    """
    s = re.sub(r"\s+", "", text or "")
    if not s:
        return None
    if s.isdigit():
        return int(s)
    total = 0       # accumulated 万-groups
    section = 0     # current group (up to 9999)
    number = 0      # pending digit
    for ch in s:
        if ch in _CN_DIGITS:
            number = _CN_DIGITS[ch]
        elif ch in _CN_UNITS:
            # "十二" means 10 + 2, i.e. a bare unit implies one of it.
            section += (number or 1) * _CN_UNITS[ch]
            number = 0
        elif ch in _CN_BIG:
            section = (section + number) * _CN_BIG[ch]
            total += section
            section = 0
            number = 0
        else:
            return None
    return total + section + number


# --------------------------------------------------------------------------
# Titles
# --------------------------------------------------------------------------
_PREFIXES = ("中华人民共和国", "中国人民", "国务院")


def normalize_title(title: str) -> str:
    """Strip 书名号 and whitespace from a document name."""
    return re.sub(r"\s+", "", title or "").strip("《》")


def short_title(title: str) -> str:
    """The name without its leading 中华人民共和国, i.e. how the corpus
    abbreviates it mid-sentence (``企业所得税法`` for
    ``中华人民共和国企业所得税法``)."""
    t = normalize_title(title)
    for prefix in _PREFIXES:
        if t.startswith(prefix) and len(t) > len(prefix) + 1:
            return t[len(prefix):]
    return t


_IMPLEMENTING_SUFFIXES = ("实施条例", "实施细则")


def implementing_law(title: str) -> str | None:
    """``X实施条例`` -> ``X``; ``None`` when the title is not an implementation.

    Restricted to 实施条例/实施细则 because both are unambiguous: a regulation
    named that way is implementing the law named by the remainder.  The looser
    suffixes (暂行条例, 实施办法) are deliberately excluded -- they routinely
    stand alone and the edge would be a guess.
    """
    t = normalize_title(title)
    for suffix in _IMPLEMENTING_SUFFIXES:
        if t.endswith(suffix) and len(t) > len(suffix):
            return t[: -len(suffix)]
    return None


class TitleIndex:
    """Maps a law name as written in a sentence to the document that owns it."""

    def __init__(self, title_to_doc: dict[str, str]) -> None:
        self._index: dict[str, str] = {}
        for title, doc_id in title_to_doc.items():
            self.add(title, doc_id)

    def add(self, title: str, doc_id: str) -> None:
        for key in (normalize_title(title), short_title(title)):
            if key and key not in self._index:
                self._index[key] = doc_id

    def __len__(self) -> int:
        return len(self._index)

    def resolve(self, name: str) -> str | None:
        """Longest-suffix resolution of a name found inside a sentence.

        The extraction pattern cannot know where the name starts -- in
        ``依据中华人民共和国企业所得税法`` it captures the 依据 too -- so the
        longest suffix that is a known title wins, which is also the most
        specific reading.
        """
        name = normalize_title(name)
        if not name:
            return None
        for start in range(len(name)):
            hit = self._index.get(name[start:])
            if hit is not None:
                return hit
        return None


# --------------------------------------------------------------------------
# Citations
# --------------------------------------------------------------------------
# The law token is one of: a self-reference (本法/本条例/本办法), a bracketed
# name (《...》), or a bare name ending in a legal suffix.  A *bare* "第X条"
# without any of these must never match -- every clause opens with its own
# number, so allowing it would make each clause cite itself.
#
# Two details are load-bearing:
# * the self-reference is 本 + a whole word (条例/办法/…), not 本 + one char,
#   or 本条例 and 本办法 in the 实施条例 family would not be recognised;
# * the bare-name branch refuses 法 when it is preceded by 本, so that the
#   "取得本法第三条" in 企税法第四条 resolves to the self token 本法 instead of
#   to a phantom law named 取得本法.
_SELF_TOKEN = r"本(?:法|条例|办法|细则|规定|规则)"
_LAW_TOKEN = (
    _SELF_TOKEN
    + r"|《[^》]{2,60}》"
    + r"|[\u4e00-\u9fff]{2,40}?(?:条例|办法|细则|规定|通知|公告|决定|规则|(?<!本)法)"
)

CITATION_RX = re.compile(
    rf"(?P<law>{_LAW_TOKEN})"
    rf"\s*第\s*(?P<article>{_ARTICLE_CLASS}+)\s*条"
    rf"(?:\s*第\s*(?P<paragraph>{_ARTICLE_CLASS}+)\s*款)?"
    rf"(?:\s*第?\s*[（(]\s*(?P<item>{_ARTICLE_CLASS}+)\s*[)）]\s*项)?"
)

TITLE_REF_RX = re.compile(r"《([^》]{2,60})》")

# Every clause starts with its own article header ("第二十七条 ..."), so the
# section label is parsed with the same numeral alphabet.
_SECTION_ART_RX = re.compile(rf"^\s*第\s*({_ARTICLE_CLASS}+)\s*条")


@dataclass(frozen=True)
class Citation:
    """One ``<law> 第X条(第X款)(第X项)`` occurrence in a clause's text."""

    raw: str
    law_token: str
    article: int
    paragraph: int | None = None
    item: int | None = None

    @property
    def is_self(self) -> bool:
        """Whether the token points at the containing document (本法/本条例/本办法)."""
        return self.law_token.startswith("本")


def find_citations(text: str) -> list[Citation]:
    """Extract every resolvable-shaped cross-reference from ``text``.

    Resolution (which document / which clause) is *not* done here: this only
    reads the sentence.  Unresolvable tokens are kept so the caller can count
    what it had to drop instead of the loss being invisible.
    """
    out: list[Citation] = []
    for match in CITATION_RX.finditer(text or ""):
        article = cn_to_int(match.group("article"))
        if article is None:
            continue
        paragraph = cn_to_int(match.group("paragraph") or "")
        item = cn_to_int(match.group("item") or "")
        out.append(
            Citation(
                raw=match.group(0).strip(),
                law_token=match.group("law").strip(),
                article=article,
                paragraph=paragraph,
                item=item,
            )
        )
    return out


def find_title_references(text: str) -> list[str]:
    """Names written inside 书名号, in order of first appearance."""
    seen: dict[str, None] = {}
    for name in TITLE_REF_RX.findall(text or ""):
        normalized = normalize_title(name)
        if normalized:
            seen.setdefault(normalized, None)
    return list(seen)


def parse_section_article(section: str) -> int | None:
    """Article number of a ``第X条`` section label, or ``None`` for 全文."""
    match = _SECTION_ART_RX.match(section or "")
    if not match:
        return None
    return cn_to_int(match.group(1))


# --------------------------------------------------------------------------
# Snapshot -> graph shape
# --------------------------------------------------------------------------
@dataclass
class ClauseRecord:
    """One retrievable clause, i.e. one node of the structural layer.

    Keyed by the chunker's ``parent_id`` rather than by article number: a
    clause longer than ``chunk_size_chars`` is split into several parents that
    share a section label but have distinct ids, and those parents are the
    bounded units ``expand_to_parent`` hands back.  Merging them by label would
    produce nodes several times the size the rest of the stack expects.
    """

    clause_id: str
    doc_id: str
    source_rel: str
    title: str
    section: str
    article: int | None
    text: str
    chunk_ids: list[str] = field(default_factory=list)


@dataclass
class CitationEdge:
    src_clause_id: str
    dst_clause_id: str
    props: dict[str, Any]


@dataclass
class StructuralExtraction:
    clauses: list[ClauseRecord] = field(default_factory=list)
    citations: list[CitationEdge] = field(default_factory=list)
    implements: list[tuple[str, str]] = field(default_factory=list)
    references: list[tuple[str, str]] = field(default_factory=list)
    doc_titles: dict[str, str] = field(default_factory=dict)
    stats: dict[str, int] = field(default_factory=dict)


def doc_id_for(source_rel: str) -> str:
    """Same id derivation as ``Neo4jGraphStore.doc_id_for`` (must not drift)."""
    return "doc:" + (source_rel or "").replace("\\", "/")


def extract_structure(records: Iterable[dict]) -> StructuralExtraction:
    """Turn ``chunks.jsonl`` records into the structural layer's node/edge sets.

    Expects the snapshot fields written by ``IngestionPipeline._write_snapshot``
    (``id``/``title``/``content``/``source``/``parent_id``/``parent_text``/
    ``metadata_payload``).  Pure: no I/O, no Neo4j.
    """
    groups: dict[str, list[dict]] = {}
    order: list[str] = []
    doc_titles: dict[str, str] = {}

    for rec in records:
        source_rel = rec.get("source", "") or ""
        doc_id = doc_id_for(source_rel)
        doc_titles.setdefault(doc_id, rec.get("title", "") or "")
        clause_id = (rec.get("parent_id") or "").strip() or rec["id"]
        if clause_id not in groups:
            groups[clause_id] = []
            order.append(clause_id)
        groups[clause_id].append(rec)

    clauses: list[ClauseRecord] = []
    by_article: dict[tuple[str, int], str] = {}
    for clause_id in order:
        group = groups[clause_id]
        head = group[0]
        source_rel = head.get("source", "") or ""
        # All children of a parent carry the same parent_text; a single-child
        # parent carries none and its content *is* the clause.  Taking the
        # longest candidate is robust to the mixed case.
        candidates = [r.get("parent_text") or "" for r in group]
        text = (
            max(candidates, key=len)
            if any(candidates)
            else max((r.get("content", "") for r in group), key=len)
        )
        section = str((head.get("metadata_payload") or {}).get("section", "") or "")
        article = parse_section_article(section)
        doc_id = doc_id_for(source_rel)
        clause = ClauseRecord(
            clause_id=clause_id,
            doc_id=doc_id,
            source_rel=source_rel,
            title=head.get("title", "") or "",
            section=section,
            article=article,
            text=text,
            chunk_ids=[r["id"] for r in group],
        )
        clauses.append(clause)
        if article is not None and (doc_id, article) not in by_article:
            by_article[(doc_id, article)] = clause_id

    index = TitleIndex(
        {title: doc_id for doc_id, title in doc_titles.items() if title}
    )

    citations: list[CitationEdge] = []
    unresolved_law = 0
    unresolved_clause = 0
    for clause in clauses:
        for cite in find_citations(clause.text):
            if cite.is_self:
                target_doc = clause.doc_id
            else:
                target_doc = index.resolve(cite.law_token)
                if target_doc is None:
                    unresolved_law += 1
                    continue
            target_clause = by_article.get((target_doc, cite.article))
            if target_clause is None or target_clause == clause.clause_id:
                unresolved_clause += 1
                continue
            citations.append(
                CitationEdge(
                    src_clause_id=clause.clause_id,
                    dst_clause_id=target_clause,
                    props={
                        "raw": cite.raw,
                        "law_token": cite.law_token,
                        "article": cite.article,
                        "paragraph": cite.paragraph,
                        "item": cite.item,
                        "cross_doc": target_doc != clause.doc_id,
                    },
                )
            )

    implements: list[tuple[str, str]] = []
    for doc_id, title in doc_titles.items():
        law_name = implementing_law(title)
        if law_name is None:
            continue
        law_doc = index.resolve(law_name)
        if law_doc and law_doc != doc_id:
            implements.append((doc_id, law_doc))

    references: list[tuple[str, str]] = []
    seen_ref: set[tuple[str, str]] = set()
    for clause in clauses:
        for name in find_title_references(clause.text):
            target_doc = index.resolve(name)
            if not target_doc or target_doc == clause.doc_id:
                continue
            if (clause.doc_id, target_doc) in seen_ref:
                continue
            seen_ref.add((clause.doc_id, target_doc))
            references.append((clause.doc_id, target_doc))

    stats = {
        "clauses": len(clauses),
        "documents": len(doc_titles),
        "citations": len(citations),
        "citations_cross_doc": sum(1 for e in citations if e.props["cross_doc"]),
        "citations_unresolved_law": unresolved_law,
        "citations_unresolved_clause": unresolved_clause,
        "implements": len(implements),
        "references": len(references),
        "title_index_keys": len(index),
    }
    return StructuralExtraction(
        clauses=clauses,
        citations=citations,
        implements=implements,
        references=references,
        doc_titles=doc_titles,
        stats=stats,
    )
