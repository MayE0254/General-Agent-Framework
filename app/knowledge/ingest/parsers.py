"""Document parsers for the fiscal/tax regulatory corpus.

Each parser turns one source file into a :class:`ParsedDocument`:
normalized text plus provenance metadata (source path, file number and the
file-name-derived legal/authority name).  The file name is the primary
metadata source for this corpus -- ``085_中华人民共和国个人所得税法实施条例.docx``
carries both a sequence number (``085``) and the document title.

Supported formats (primary = ``.docx``, plus ``.pdf`` and ``.xlsx``):

- ``.docx``  python-docx, paragraphs + tables concatenated, blank lines kept.
- ``.pdf``   PyMuPDF (``pymupdf``) text layer per page.
- ``.xlsx``  pandas, each sheet flattened to a markdown-ish table so row/col
             semantics survive text-only retrieval.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

_ACCEPTED_EXTENSIONS = {".docx", ".pdf", ".xlsx"}


@dataclass
class ParsedDocument:
    """Normalized text of one source file plus its provenance metadata."""

    source_path: str          # absolute path of the source file
    source_rel: str           # path relative to the corpus root (for display)
    title: str                # file-name-derived title / legal name
    file_no: str              # leading sequence number, "" when absent
    doc_type: str             # inferred doc kind (law/regulation/notice/...)
    text: str                 # normalized full text (newlines preserved)
    paragraphs: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


# Leading sequence number, e.g. "085_..." -> "085", "1000_..." -> "1000".
_SEQ_RE = re.compile(r"^\s*(\d{1,4})\s*[_\-．.、]?\s*(.*)$")
_DOC_TYPE_KEYWORDS = [
    (("法", "条例", "办法", "规定", "细则", "决定", "通知", "公告", "批复",
      "答复", "意见", "函", "办法", "规划", "标准", "应急预案"), "其他"),
]


def parse_document(path: str, corpus_root: str) -> ParsedDocument:
    """Dispatch to the right parser by extension and build metadata."""
    ext = os.path.splitext(path)[1].lower()
    if ext not in _ACCEPTED_EXTENSIONS:
        raise ValueError(f"Unsupported extension {ext!r} for {path}")

    rel = os.path.relpath(path, corpus_root)
    filename = os.path.basename(path)
    title, file_no, doc_type = _metadata_from_filename(filename)

    if ext == ".docx":
        paragraphs = _parse_docx(path)
    elif ext == ".pdf":
        paragraphs = _parse_pdf(path)
    else:
        paragraphs = _parse_xlsx(path)

    lines = [p.rstrip() for p in paragraphs if p and p.strip()]
    text = "\n".join(lines)

    meta = {
        "source_path": path,
        "source_rel": rel,
        "filename": filename,
        "extension": ext,
        "file_no": file_no,
        "title": title,
        "doc_type": doc_type,
    }
    return ParsedDocument(
        source_path=path,
        source_rel=rel,
        title=title,
        file_no=file_no,
        doc_type=doc_type,
        text=text,
        paragraphs=lines,
        meta=meta,
    )


def _metadata_from_filename(filename: str) -> tuple[str, str, str]:
    """Parse ``SEQ_TITLE.ext`` into (title, file_no, doc_type)."""
    base = os.path.splitext(filename)[0]
    title = base
    file_no = ""
    m = _SEQ_RE.match(base)
    if m and m.group(1):
        file_no = m.group(1)
        title = (m.group(2) or base).strip()
    # Clean BBCode-ish stray markers seen in the corpus (underscore doubling).
    title = re.sub(r"_+", " ", title).strip()
    return title, file_no, _infer_doc_type(title)


def _infer_doc_type(title: str) -> str:
    lower = title
    if "实施条例" in lower or "暂行条例" in lower or "条例" in lower:
        return "条例"
    if "法" in lower and "办法" not in lower:
        return "法律"
    if "办法" in lower or "规定" in lower or "细则" in lower:
        return "规章/办法"
    if "通知" in lower or "公告" in lower:
        return "通知/公告"
    if "批复" in lower or "答复" in lower or "意见" in lower:
        return "批复/答复"
    return "其他"


def _parse_docx(path: str) -> list[str]:
    from docx import Document  # python-docx

    doc = Document(path)
    out: list[str] = []
    # Iterate body blocks in document order so tables interleave as read.
    from docx.oxml.ns import qn

    body = doc.element.body
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            para = Paragraph(child, doc)
            out.append(para.text or "")
        elif child.tag == qn("w:tbl"):
            table = Table(child, doc)
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells]
                out.append(" | ".join(cells))
    return out


def _parse_pdf(path: str) -> list[str]:
    import pymupdf

    doc = pymupdf.open(path)
    out: list[str] = []
    for page in doc:
        out.append(page.get_text("text") or "")
    doc.close()
    return out


def _parse_xlsx(path: str) -> list[str]:
    import pandas as pd

    out: list[str] = []
    sheets = pd.read_excel(path, sheet_name=None, header=None, dtype=str)
    for sheet_name, frame in sheets.items():
        out.append(f"[Sheet: {sheet_name}]")
        for _, row in frame.iterrows():
            cells = [str(c).strip() for c in row.tolist() if str(c).strip() != "nan"]
            if cells:
                out.append(" | ".join(cells))
    return out