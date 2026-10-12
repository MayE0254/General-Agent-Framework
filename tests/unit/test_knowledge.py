"""Unit tests for the knowledge retrieval abstraction and tool wiring."""

import asyncio
import re

import pytest

from app.knowledge import (
    BaseKnowledgeRetriever,
    InMemoryKnowledgeRetriever,
    KnowledgeHit,
)
from app.knowledge.base import QUOTA_OVERFLOW_KEY
from app.knowledge.diversity import (
    cap_per_document,
    collapse_by_parent,
    expand_to_parent,
)
from app.knowledge.fusion import reciprocal_rank_fusion
from app.knowledge.ingest.chunker import chunk_document
from app.knowledge.ingest.parsers import ParsedDocument
from app.knowledge.rag import (
    INSUFFICIENT_EVIDENCE_MARKER,
    SUPPLEMENTARY_HEADING,
    EmptyAnswerError,
    IncompleteAnswerError,
    KnowledgeAnswerer,
    TruncatedAnswerError,
    _is_refusal,
    build_prompt,
    build_system_prompt,
)
from app.knowledge.rerank import CrossEncoderReranker
from app.llm import LLMRequest, LLMResponse, LLMUsage
from app.tools.bootstrap import create_default_tool_registry
from app.tools.system import KnowledgeSearchTool

DOCS = [
    {
        "id": "kb-001",
        "title": "Platform overview",
        "content": "multi-agent platform with orchestrator and tools",
        "tags": ["platform"],
    },
    {
        "id": "kb-002",
        "title": "Memory guide",
        "content": "session memory and durable memory scopes",
        "tags": ["memory"],
    },
    {
        "id": "kb-003",
        "title": "Deployment",
        "content": "docker compose deployment for the platform",
        "tags": ["deploy"],
    },
]


def test_retriever_ranks_by_token_overlap() -> None:
    retriever = InMemoryKnowledgeRetriever(DOCS)

    hits = asyncio.run(retriever.retrieve("multi-agent platform", top_k=3))

    assert hits, "expected at least one hit for an overlapping query"
    assert hits[0].id == "kb-001"
    assert hits[0].score >= hits[-1].score
    assert all(0.0 <= hit.score <= 1.0 for hit in hits)


def test_retriever_respects_top_k_and_min_score() -> None:
    # Partial overlap (1/3 query tokens) survives the default threshold...
    partial_hits = asyncio.run(
        InMemoryKnowledgeRetriever(DOCS).retrieve(
            "platform blockchain quantum", top_k=5
        )
    )
    assert partial_hits
    assert all(hit.id in {"kb-001", "kb-003"} for hit in partial_hits)

    # ...but is filtered out by a strict threshold. Exact matches (score 1.0)
    # still pass, so the query must be a partial-overlap one.
    strict = InMemoryKnowledgeRetriever(DOCS, min_score=0.9)
    assert (
        asyncio.run(strict.retrieve("platform blockchain quantum", top_k=5)) == []
    )

    limited = InMemoryKnowledgeRetriever(DOCS)
    hits = asyncio.run(limited.retrieve("platform deployment docker", top_k=1))
    assert len(hits) == 1
    assert hits[0].id == "kb-003"


def test_retriever_handles_empty_inputs() -> None:
    retriever = InMemoryKnowledgeRetriever([])
    assert asyncio.run(retriever.retrieve("anything")) == []

    empty_query = InMemoryKnowledgeRetriever(DOCS)
    assert asyncio.run(empty_query.retrieve("", top_k=3)) == []


def test_retriever_rerank_hook_keeps_order_by_default() -> None:
    retriever = InMemoryKnowledgeRetriever(DOCS)
    hits = asyncio.run(retriever.retrieve("platform", top_k=2))
    reranked = retriever.rerank("platform", hits)
    assert [hit.id for hit in reranked] == [hit.id for hit in hits]


def test_knowledge_hit_citation_ref() -> None:
    hit = KnowledgeHit(id="kb-9", content="text")
    assert hit.citation_ref("inmemory") == "inmemory:kb-9"


def test_knowledge_search_tool_uses_configured_documents() -> None:
    tool = KnowledgeSearchTool(
        config={
            "documents": DOCS,
            "top_k": 2,
            "min_score": 0.05,
        }
    )
    result = asyncio.run(
        tool._run(context=None, arguments={"query": "multi-agent platform"})
    )

    assert result.status.value == "success"
    assert result.content["backend"] == "inmemory"
    assert 1 <= result.content["hits"] <= 2
    assert result.content["results"]
    assert result.content["results"][0]["id"] == "kb-001"


def test_knowledge_search_tool_without_documents_returns_zero_hits() -> None:
    tool = KnowledgeSearchTool()
    result = asyncio.run(tool._run(context=None, arguments={"query": "anything"}))

    assert result.status.value == "success"
    assert result.content["hits"] == 0
    assert result.content["results"] == []


def test_tool_registry_binds_knowledge_settings() -> None:
    from app.core.settings import KnowledgeDocument, KnowledgeSettings

    settings = KnowledgeSettings(
        documents=[
            KnowledgeDocument(id="kb-1", content="enterprise agents platform")
        ]
    )
    registry = create_default_tool_registry(knowledge_settings=settings)
    tool = registry.create("knowledge_search")

    result = asyncio.run(
        tool._run(context=None, arguments={"query": "enterprise agents"})
    )

    assert result.content["hits"] == 1
    assert result.content["results"][0]["id"] == "kb-1"


def test_retriever_is_swap_for_base_protocol() -> None:
    assert issubclass(InMemoryKnowledgeRetriever, BaseKnowledgeRetriever)


# ---------------------------------------------------------------------------
# RRF pre-fusion (pure function, no external services)
# ---------------------------------------------------------------------------
def _hit(cid: str, **payload) -> KnowledgeHit:
    return KnowledgeHit(id=cid, content=f"content-{cid}", metadata_payload=payload)


def test_rrf_prefers_hits_found_by_both_legs() -> None:
    # "a" is #1 in the vector leg and #2 in the BM25 leg, so its summed RRF
    # score beats anything surfaced by a single leg.
    vector = [_hit("a"), _hit("b")]
    keyword = [_hit("c"), _hit("a")]

    fused = reciprocal_rank_fusion([vector, keyword], k=60)

    assert [h.id for h in fused] == ["a", "c", "b"]


def test_rrf_merges_leg_metadata_and_writes_rank() -> None:
    vector = [_hit("a", vector_score=0.91)]
    keyword = [_hit("a", bm25_score=7.5)]

    fused = reciprocal_rank_fusion([vector, keyword], k=60)

    payload = fused[0].metadata_payload
    assert payload["vector_score"] == 0.91
    assert payload["bm25_score"] == 7.5
    assert payload["rrf_rank"] == 1
    assert payload["rrf_score"] == pytest.approx(2 / 61)


def test_rrf_preserves_parent_linkage_for_expansion() -> None:
    # End-to-end at unit level: RRF -> collapse -> expand.  If the fused hit
    # loses parent_id/parent_text, collapse_by_parent degrades to deduping by
    # id and expand_to_parent passes every hit through, so the retriever ships
    # child fragments instead of whole clauses while looking perfectly healthy.
    child = KnowledgeHit(
        id="c1",
        content="……基础电信服务，税率为百分之九。",
        parent_id="p1",
        parent_text="第十条 增值税税率：……基础电信服务，税率为百分之九。",
        score=0.5,
    )

    fused = reciprocal_rank_fusion([[child], [_hit("c2")]], k=60)
    expanded = expand_to_parent(collapse_by_parent(fused))

    hit = next(item for item in expanded if item.id == "c1")
    assert hit.parent_id == "p1"
    assert hit.content == child.parent_text
    assert hit.metadata_payload["expanded_to_parent"] is True


def test_rerank_preserves_parent_linkage() -> None:
    # The reranker is the last stage before expand_to_parent, so it cannot drop
    # the field either -- rebuilding the hit field by field was how the parent
    # linkage went missing in the first place.
    class _Model:
        def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
            return [0.2, 0.9]

    reranker = CrossEncoderReranker("stub")
    reranker._model = _Model()
    hits = [
        KnowledgeHit(
            id="c1", content="子块甲", parent_id="p1", parent_text="第一条 甲。"
        ),
        KnowledgeHit(
            id="c2", content="子块乙", parent_id="p2", parent_text="第二条 乙。"
        ),
    ]

    reranked = reranker.rerank("问题", hits)

    assert [item.id for item in reranked] == ["c2", "c1"]
    assert reranked[0].parent_id == "p2"
    assert expand_to_parent(reranked)[0].content == "第二条 乙。"
    assert reranked[0].metadata_payload["rerank_score"] == 0.9


def test_rrf_handles_empty_legs() -> None:
    assert reciprocal_rank_fusion([[], []], k=60) == []
    assert reciprocal_rank_fusion([[_hit("a")], []], k=60)[0].id == "a"


def test_rrf_rejects_non_positive_k() -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[_hit("a")]], k=0)


# ---------------------------------------------------------------------------
# Parent/child chunking (pure function, no external services)
# ---------------------------------------------------------------------------
def _doc(*paragraphs: str) -> ParsedDocument:
    return ParsedDocument(
        source_path="C:/corpus/001_test.docx",
        source_rel="001_test.docx",
        title="测试法规",
        file_no="001",
        doc_type="law",
        text="\n".join(paragraphs),
        paragraphs=list(paragraphs),
    )


def _norm(text: str) -> str:
    """Compare chunk *content*, not the chunker's internal line joining.

    ``_split_text`` re-joins sentence fragments with ``\\n``, so raw string
    equality would test the formatting rather than the parent/child
    relationship.  Sentence enders are kept by the chunker and therefore are
    *not* stripped here -- doing so would hide a lost-punctuation regression.
    """
    return re.sub(r"\s", "", text)


def test_chunker_single_child_clause_carries_no_parent_text() -> None:
    chunks = chunk_document(
        _doc("第一条 本办法自发布之日起施行。"),
        chunk_size=900,
        overlap=120,
        child_window=400,
    )

    assert len(chunks) == 1
    assert chunks[0].section == "第一条"
    assert chunks[0].parent_id, "every child must be addressable to its clause"
    # The child *is* the whole clause, so storing parent_text would only
    # duplicate the same string on every Milvus row.
    assert chunks[0].parent_text == ""
    assert chunks[0].metadata_payload["parent_char_len"] == len(chunks[0].text)


def test_chunker_splits_long_clause_into_siblings_sharing_a_parent() -> None:
    # A clause header is its own paragraph in real .docx sources; the body
    # follows.  The header is what establishes section/parent identity.
    body = "".join(f"第{n}项规定的具体情形应当依照本办法执行。" for n in range(1, 30))
    chunks = chunk_document(
        _doc("第二条", body),
        chunk_size=900,
        overlap=0,
        child_window=200,
    )

    assert len(chunks) > 1, "an over-long clause must be split into child windows"
    assert len({c.parent_id for c in chunks}) == 1, "siblings share one parent"
    for chunk in chunks:
        assert len(chunk.text) <= 200
        # Each fragment must be able to rebuild the whole clause on its own.
        assert chunk.parent_text
        assert _norm(chunk.text) in _norm(chunk.parent_text)
        assert chunk.section == "第二条"


def test_chunker_parents_partition_the_clause_without_overlap() -> None:
    # Parents must partition the clause exactly: overlap duplication is what
    # the parent collapse relies on being confined to the child level.
    body = "".join(f"第{n}段内容。" for n in range(1, 120))
    chunks = chunk_document(
        _doc("第三条", body),
        chunk_size=300,
        overlap=50,
        child_window=150,
    )

    parents = list(dict.fromkeys(c.parent_id for c in chunks))
    assert len(parents) > 1, "a clause longer than chunk_size must yield several parents"

    parent_texts: list[str] = []
    for parent in parents:
        group = [c for c in chunks if c.parent_id == parent]
        text = group[0].parent_text or group[0].text
        assert all((c.parent_text or c.text) == text for c in group)
        parent_texts.append(text)

    # Exact concatenation proves parents do not overlap...
    assert "".join(_norm(text) for text in parent_texts) == _norm("第三条 " + body)
    # ...while children inside a parent do (that is what helps recall).
    second_group = [c for c in chunks if c.parent_id == parents[1]]
    assert len(second_group) > 1
    assert sum(len(_norm(c.text)) for c in second_group) > len(_norm(parent_texts[1]))


def test_chunker_keeps_document_order_across_clauses() -> None:
    chunks = chunk_document(
        _doc("第一条 甲。", "第二条 乙。", "第三条 丙。"),
        chunk_size=900,
        overlap=120,
        child_window=400,
    )

    assert [c.chunk_index for c in chunks] == [0, 1, 2]
    assert [c.section for c in chunks] == ["第一条", "第二条", "第三条"]


def test_chunker_ids_are_stable_and_collision_free() -> None:
    doc = _doc("第一条 甲。", "第二条 乙。")
    first = chunk_document(doc, chunk_size=900, overlap=120, child_window=400)
    again = chunk_document(doc, chunk_size=900, overlap=120, child_window=400)

    assert [c.id for c in first] == [c.id for c in again], "re-ingest must be idempotent"
    ids = {c.id for c in first} | {c.parent_id for c in first}
    assert len(ids) == 2 * len(first), "parent and child ids must never collide"


def test_chunker_rejects_child_window_larger_than_parent() -> None:
    # A silently degenerate configuration would make the whole parent/child
    # feature a no-op, so it is rejected loudly instead.
    with pytest.raises(ValueError, match="child_window"):
        chunk_document(
            _doc("第一条 甲。"),
            chunk_size=300,
            overlap=0,
            child_window=400,
        )


# ---------------------------------------------------------------------------
# Diversity controls (pure functions, no external services)
# ---------------------------------------------------------------------------
def _child(cid: str, *, parent: str, source: str, score: float = 0.0) -> KnowledgeHit:
    return KnowledgeHit(
        id=cid,
        content=f"content-{cid}",
        score=score,
        source=source,
        parent_id=parent,
        parent_text=f"parent-of-{parent}",
    )


def test_collapse_by_parent_keeps_best_ranked_sibling() -> None:
    hits = [
        _child("c1", parent="p1", source="a"),
        _child("c2", parent="p1", source="a"),
        _child("c3", parent="p2", source="a"),
    ]

    collapsed = collapse_by_parent(hits)

    assert [h.id for h in collapsed] == ["c1", "c3"]
    assert collapsed[0].metadata_payload["collapsed_siblings"] == 1
    # A parent with no siblings keeps a clean payload (no noisy zero).
    assert "collapsed_siblings" not in collapsed[1].metadata_payload


def test_collapse_by_parent_treats_missing_parent_as_unique() -> None:
    hits = [
        KnowledgeHit(id="solo-1", content="x"),
        KnowledgeHit(id="solo-2", content="y"),
    ]

    assert [h.id for h in collapse_by_parent(hits)] == ["solo-1", "solo-2"]


def test_cap_per_document_quota_stops_one_document_monopolising() -> None:
    # A single long statute supplies every candidate, which is exactly the
    # "大部头挤压" case the quota exists to break.
    hits = [_child(f"big-{n}", parent=f"p{n}", source="民法典") for n in range(10)]
    hits += [_child(f"small-{n}", parent=f"q{n}", source="通知A") for n in range(3)]

    capped, overflow = cap_per_document(hits, max_per_document=3, limit=6)

    per_source: dict[str, int] = {}
    for him in capped:
        per_source[him.source] = per_source.get(him.source, 0) + 1
    assert per_source == {"民法典": 3, "通知A": 3}
    # Rank order is preserved, so relevance inside a document is never traded
    # for a weaker document's top hit.
    assert capped[0].id == "big-0"
    assert capped[3].id == "small-0"
    # Overflow is opt-in; with overflow_limit unset nothing is handed back.
    assert overflow == []


def test_cap_per_document_backfills_instead_of_shrinking_the_pool() -> None:
    # Only one document matches: the quota must not return 3 results where the
    # caller asked for a pool of 5.
    hits = [_child(f"only-{n}", parent=f"p{n}", source="唯一法规") for n in range(5)]

    capped, overflow = cap_per_document(hits, max_per_document=3, limit=5)

    assert [h.id for h in capped] == [f"only-{n}" for n in range(5)]
    # Backfilled hits were returned after all, so they are not overflow: handing
    # them back a second time as "rejected" would double-count them.
    assert overflow == []


def test_cap_per_document_groups_uncited_hits_by_their_own_id() -> None:
    hits = [KnowledgeHit(id=f"x{n}", content="t") for n in range(4)]

    capped, _ = cap_per_document(hits, max_per_document=1, limit=4)

    assert [h.id for h in capped] == ["x0", "x1", "x2", "x3"]


def test_cap_per_document_reports_the_quota_rejected_remainder_as_overflow() -> None:
    # The vat-001 shape measured on the live corpus: one statute fills the pool
    # (so backfilling has no slots left), and the clause that actually holds the
    # answer was rejected purely for being its 4th-ranked clause.
    hits = [_child(f"big-{n}", parent=f"p{n}", source="增值税法") for n in range(6)]
    hits += [_child(f"other-{n}", parent=f"q{n}", source=f"通知{n}") for n in range(12)]

    capped, overflow = cap_per_document(
        hits, max_per_document=3, limit=5, overflow_limit=3
    )

    assert len(capped) == 5
    # Quota-rejected, in rank order, and bounded by overflow_limit.
    assert [h.id for h in overflow] == ["big-3", "big-4", "big-5"]


def test_cap_per_document_overflow_excludes_hits_that_lost_on_rank() -> None:
    # Candidates past the pool cutoff lost the ranking competition, not the
    # quota -- they were never judged relevant enough to place, so calling them
    # "relevant but excluded" would be a lie the caller then shows to the LLM.
    hits = [
        _child("a1", parent="p1", source="民法典"),
        _child("b1", parent="p2", source="通知A"),
        _child("c1", parent="p3", source="通知B"),
        _child("a2", parent="p4", source="民法典"),
    ]

    capped, overflow = cap_per_document(
        hits, max_per_document=1, limit=2, overflow_limit=5
    )

    assert [h.id for h in capped] == ["a1", "b1"]
    # a2 is also from 民法典 and would have hit the quota, but the walk stopped
    # at the cutoff before reaching it.
    assert overflow == []


def test_cap_per_document_overflow_is_capped_by_its_own_budget() -> None:
    hits = [_child(f"big-{n}", parent=f"p{n}", source="民法典") for n in range(9)]

    capped, overflow = cap_per_document(
        hits, max_per_document=1, limit=3, overflow_limit=2
    )

    # Six candidates are quota-rejected; two are backfilled into the underfilled
    # pool, so the budget applies to what is left.
    assert [h.id for h in capped] == ["big-0", "big-1", "big-2"]
    assert [h.id for h in overflow] == ["big-3", "big-4"]


def test_cap_per_document_rejects_invalid_quota() -> None:
    with pytest.raises(ValueError):
        cap_per_document([], max_per_document=0, limit=5)


def test_cap_per_document_rejects_negative_overflow_limit() -> None:
    with pytest.raises(ValueError, match="overflow_limit"):
        cap_per_document([], max_per_document=1, limit=5, overflow_limit=-1)


def test_expand_to_parent_returns_whole_clause_and_keeps_child_id() -> None:
    hit = _child("c1", parent="p1", source="a")

    expanded = expand_to_parent([hit])[0]

    assert expanded.content == "parent-of-p1"
    # The id stays on the indexed child (that is what exists in Milvus/BM25),
    # while parent_id identifies the clause for citation.
    assert expanded.id == "c1"
    assert expanded.parent_id == "p1"
    assert expanded.metadata_payload["matched_child_id"] == "c1"
    assert expanded.metadata_payload["expanded_to_parent"] is True


def test_expand_to_parent_passes_through_hits_without_a_parent() -> None:
    hit = KnowledgeHit(id="solo", content="already-whole")

    expanded = expand_to_parent([hit])[0]

    assert expanded.content == "already-whole"
    assert "expanded_to_parent" not in expanded.metadata_payload


# ---------------------------------------------------------------------------
# Grounded answering chain: retrieve -> prompt -> generate
# ---------------------------------------------------------------------------
class _StubLLM:
    """Records each request and answers with a canned string per call.

    ``contents`` hands out one reply per call, which is what the answerer's
    refusal retry needs; a bare ``content`` is reused for every call.
    """

    def __init__(
        self,
        content: str = "结论：可以抵扣。[1]",
        finish_reason: str = "stop",
        contents: list[str] | None = None,
    ) -> None:
        self.content = content
        self.contents = list(contents or [])
        self.finish_reason = finish_reason
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        content = self.contents.pop(0) if self.contents else self.content
        return LLMResponse(
            model=request.model,
            content=content,
            finish_reason=self.finish_reason,
            usage=LLMUsage(prompt_tokens=11, completion_tokens=7, total_tokens=18),
            latency_ms=5,
        )


class _StubRetriever(BaseKnowledgeRetriever):
    name = "stub"

    def __init__(self, hits: list[KnowledgeHit]) -> None:
        self._hits = hits

    async def retrieve(self, query: str, *, top_k: int = 5) -> list[KnowledgeHit]:
        return self._hits[:top_k]


def _answerer(
    hits: list[KnowledgeHit], llm: _StubLLM, **kwargs
) -> KnowledgeAnswerer:
    return KnowledgeAnswerer(
        _StubRetriever(hits), llm, model="deepseek/deepseek-v4-flash", **kwargs
    )


def test_build_prompt_numbers_blocks_and_reports_what_fits() -> None:
    hits = [
        KnowledgeHit(
            id="c1",
            title="增值税法",
            content="第一条 内容甲。",
            source="03_增值税法.docx",
            metadata_payload={"section": "第一条"},
        ),
        KnowledgeHit(
            id="c2",
            title="增值税法",
            content="第二条 内容乙。",
            source="03_增值税法.docx",
            metadata_payload={"section": "第二条"},
        ),
    ]

    message, included = build_prompt("能抵扣吗", hits, max_context_chars=10_000)

    assert [hit.id for hit in included] == ["c1", "c2"]
    assert "[1] 03_增值税法.docx ｜ 增值税法 ｜ 第一条" in message
    assert "[2] 03_增值税法.docx ｜ 增值税法 ｜ 第二条" in message
    assert message.endswith("【用户问题】\n能抵扣吗")


def test_build_prompt_drops_over_budget_hits_from_both_prompt_and_citations() -> None:
    # A hit that did not reach the prompt must not be citable either: judging an
    # answer against context the model never saw would score the wrong thing.
    hits = [
        KnowledgeHit(id="c1", content="甲" * 100),
        KnowledgeHit(id="c2", content="乙" * 100),
    ]

    message, included = build_prompt("问题", hits, max_context_chars=150)

    assert [hit.id for hit in included] == ["c1"]
    assert "c2" not in message


def test_build_prompt_keeps_one_block_even_when_it_exceeds_the_budget() -> None:
    # An empty context would turn a grounded answer into a refusal, so the
    # budget is a soft cap: one oversized block still gets through.
    hits = [KnowledgeHit(id="c1", content="甲" * 500)]

    _, included = build_prompt("问题", hits, max_context_chars=10)

    assert [hit.id for hit in included] == ["c1"]


def test_build_prompt_renders_quota_overflow_in_its_own_section() -> None:
    hits = [
        KnowledgeHit(
            id="c1",
            title="增值税法",
            content="第一条 内容甲。",
            source="03_增值税法.docx",
            metadata_payload={"section": "第一条"},
        ),
        KnowledgeHit(
            id="c2",
            title="增值税法",
            content="第十条 税率甲。",
            source="03_增值税法.docx",
            metadata_payload={QUOTA_OVERFLOW_KEY: True},
        ),
    ]

    message, included = build_prompt("税率分几档", hits, max_context_chars=10_000)

    assert [hit.id for hit in included] == ["c1", "c2"]
    # Numbering runs on across the sections, so [2] still resolves to a block.
    assert "[2] 03_增值税法.docx ｜ 增值税法\n第十条 税率甲。" in message
    # The hint comes after the ranked clauses and says what it is.
    assert (
        message.index("【检索到的法规片段】")
        < message.index(SUPPLEMENTARY_HEADING)
        < message.index("【用户问题】")
    )
    # Rule 1 confines the answer to the listed fragments, so a heading the
    # system prompt never mentions would make the section unusable.
    assert SUPPLEMENTARY_HEADING in build_system_prompt(with_supplementary=True)


def test_system_prompt_names_only_the_sources_the_call_carries() -> None:
    # The rules name their sources, so the prompt has to match the user message
    # it is sent with: on a first pass there is no supplementary section, and
    # naming one that is not there measured worse Faithfulness (RAGAS doc §7.6).
    plain = build_system_prompt(with_supplementary=False)

    assert SUPPLEMENTARY_HEADING not in plain
    assert "【检索到的法规片段】" in plain


def test_system_prompt_quotes_the_refusal_marker_verbatim() -> None:
    # The answerer detects a refusal by substring, and a refusal is what triggers
    # the supplementary retry -- so a reworded rule 3 would quietly turn the
    # retry off without failing anything else.
    for with_supplementary in (False, True):
        assert INSUFFICIENT_EVIDENCE_MARKER in build_system_prompt(
            with_supplementary=with_supplementary
        )


def test_build_prompt_keeps_the_supplementary_section_within_the_budget() -> None:
    hits = [
        KnowledgeHit(id="c1", content="甲" * 100),
        KnowledgeHit(
            id="c2",
            content="乙" * 100,
            metadata_payload={QUOTA_OVERFLOW_KEY: True},
        ),
    ]

    message, included = build_prompt("问题", hits, max_context_chars=150)

    assert [hit.id for hit in included] == ["c1"]
    assert SUPPLEMENTARY_HEADING not in message


def test_answerer_answers_from_retrieved_clauses_with_citations() -> None:
    hits = [
        KnowledgeHit(
            id="c1",
            title="增值税法",
            content="第一条 可以抵扣。",
            source="03_增值税法.docx",
            score=0.9,
            parent_id="p1",
            metadata_payload={"section": "第一条"},
        )
    ]
    llm = _StubLLM()

    answer = asyncio.run(_answerer(hits, llm).answer("能抵扣吗"))

    assert answer.answer == "结论：可以抵扣。[1]"
    assert answer.contexts == ["第一条 可以抵扣。"]
    assert answer.retrieved == 1
    assert answer.citations[0].index == 1
    assert answer.citations[0].chunk_id == "c1"
    assert answer.citations[0].section == "第一条"
    assert answer.prompt_tokens == 11

    sent = llm.requests[0]
    assert sent.messages[0].content == build_system_prompt(with_supplementary=False)
    # The invariant that makes Faithfulness meaningful: the judged contexts are
    # verbatim what the model was given.
    for context in answer.contexts:
        assert context in sent.messages[1].content


def test_answerer_forwards_extra_body_to_the_generation_call() -> None:
    # The generation leg runs with thinking disabled (a thinking chain shares
    # max_tokens with the answer and only eats the budget), and the switch is
    # configuration rather than hardcoded, so a non-DeepSeek generation model
    # does not receive a DeepSeek-only field.
    hits = [KnowledgeHit(id="c1", content="第一条 可以抵扣。", score=0.9)]
    llm = _StubLLM()

    asyncio.run(
        _answerer(
            hits, llm, extra_body={"thinking": {"type": "disabled"}}
        ).answer("能抵扣吗")
    )

    assert llm.requests[0].extra_body == {"thinking": {"type": "disabled"}}


def test_answerer_refuses_without_retrieval_and_never_calls_the_llm() -> None:
    llm = _StubLLM()

    answer = asyncio.run(_answerer([], llm).answer("火星上的税率是多少"))

    assert answer.answer == "检索到的法规片段不足以回答该问题。"
    assert answer.contexts == []
    assert answer.citations == []
    assert llm.requests == [], "no evidence must mean no generation call"


def test_answerer_raises_on_an_empty_completion_instead_of_scoring_it() -> None:
    # Observed with deepseek-v4-flash at max_tokens=1200: the thinking chain ate
    # the whole budget, the call was billed (completion_tokens == max_tokens)
    # and content came back empty.  Returning that string made the evaluation
    # record it as a real sample with Answer Relevancy 0.0.
    hits = [KnowledgeHit(id="c1", content="第一条 可以抵扣。", score=0.9)]
    llm = _StubLLM(content="")

    with pytest.raises(EmptyAnswerError) as excinfo:
        asyncio.run(_answerer(hits, llm).answer("能抵扣吗"))

    assert llm.requests, "the call was still made (and billed)"
    assert "returned no answer text" in str(excinfo.value)


def test_answerer_rejects_a_truncated_completion() -> None:
    # vat-005 shipped like this: non-empty answer, but "length" says max_tokens
    # cut it mid-clause, so it graded a sentence the model never finished.
    hits = [KnowledgeHit(id="c1", content="第十条 应税行为。", score=0.9)]
    llm = _StubLLM(content="纳税人应当凭法律、行政法规", finish_reason="length")

    with pytest.raises(TruncatedAnswerError) as excinfo:
        asyncio.run(_answerer(hits, llm).answer("如何抵扣"))

    assert llm.requests, "the call was still made (and billed)"
    assert "stopped at max_tokens=" in str(excinfo.value)
    # Both failure modes are one class for a caller that only needs "not usable".
    assert isinstance(excinfo.value, IncompleteAnswerError)


def test_answerer_accepts_an_answer_that_ends_without_a_reported_reason() -> None:
    # Providers that omit finish_reason must not be read as truncated.
    hits = [KnowledgeHit(id="c1", content="第一条 可以抵扣。", score=0.9)]
    llm = _StubLLM(content="结论：可以抵扣。[1]", finish_reason="")

    answer = asyncio.run(_answerer(hits, llm).answer("能抵扣吗"))

    assert answer.answer == "结论：可以抵扣。[1]"


def test_answerer_holds_back_supplementary_clauses_when_the_first_pass_answers() -> None:
    # The whole point of the trigger: 10 of 15 evaluation samples got these
    # clauses while only 2 needed them, and the ones that did not need them paid
    # for it in precision and latency.
    hits = [
        KnowledgeHit(id="c1", content="第十二条 分别核算。", source="03.docx"),
        KnowledgeHit(
            id="c2",
            content="第十条 增值税税率：",
            source="03.docx",
            metadata_payload={QUOTA_OVERFLOW_KEY: True},
        ),
    ]
    llm = _StubLLM(content="结论：分别核算。[1]")

    answer = asyncio.run(_answerer(hits, llm).answer("要分别核算吗"))

    assert len(llm.requests) == 1
    assert SUPPLEMENTARY_HEADING not in llm.requests[0].messages[1].content
    assert answer.contexts == ["第十二条 分别核算。"]
    assert [c.supplementary for c in answer.citations] == [False]
    assert answer.insufficient_evidence is False
    # The held-back clause is still reported as retrieved, so a reader can tell
    # the retrieval offered it and generation declined it.
    assert answer.retrieved == 2


def test_answerer_retries_with_supplementary_clauses_after_a_refusal() -> None:
    hits = [
        KnowledgeHit(id="c1", content="第十二条 分别核算。", source="03.docx"),
        KnowledgeHit(
            id="c2",
            content="第十条 增值税税率：",
            source="03.docx",
            metadata_payload={QUOTA_OVERFLOW_KEY: True},
        ),
    ]
    llm = _StubLLM(
        contents=[f"{INSUFFICIENT_EVIDENCE_MARKER}，缺少税率条款。", "结论：四档。[2]"]
    )

    answer = asyncio.run(_answerer(hits, llm).answer("税率分几档"))

    assert len(llm.requests) == 2
    retry = llm.requests[1].messages[1].content
    assert SUPPLEMENTARY_HEADING in retry
    assert "第十条 增值税税率：" in retry
    assert retry.index("第十二条 分别核算。") < retry.index(SUPPLEMENTARY_HEADING)
    # Each call is told about the sources it carries: the first pass has no
    # supplementary section, so its prompt must not pretend otherwise.
    assert SUPPLEMENTARY_HEADING not in llm.requests[0].messages[0].content
    assert SUPPLEMENTARY_HEADING in llm.requests[1].messages[0].content
    # The answered pass is the one that counts, and it says why it could answer.
    assert answer.answer == "结论：四档。[2]"
    assert answer.contexts == ["第十二条 分别核算。", "第十条 增值税税率："]
    assert [c.supplementary for c in answer.citations] == [False, True]
    assert answer.insufficient_evidence is False
    # Both calls were billed and both took time, so the answer reports the sum.
    assert answer.latency_ms == 10


def test_answerer_retries_on_a_rephrased_refusal() -> None:
    # What the model actually returned (vat-001, 2026-09-24): the demanded
    # sentence with the missing evidence folded into it.  An exact-substring
    # test of the marker passes the verbatim case above and fails this one --
    # and this one is what occurs, so failing it would switch the retry off in
    # production without failing anything else.
    hits = [
        KnowledgeHit(id="c1", content="第十二条 分别核算。", source="03.docx"),
        KnowledgeHit(
            id="c2",
            content="第十条 增值税税率：",
            source="03.docx",
            metadata_payload={QUOTA_OVERFLOW_KEY: True},
        ),
    ]
    llm = _StubLLM(
        contents=[
            "检索到的法规片段不足以完整回答该问题，缺少现行税率档次条款。",
            "结论：四档。[2]",
        ]
    )

    answer = asyncio.run(_answerer(hits, llm).answer("税率分几档"))

    assert len(llm.requests) == 2
    assert answer.answer == "结论：四档。[2]"
    assert [c.supplementary for c in answer.citations] == [False, True]


def test_refusal_check_ignores_a_grounded_answer() -> None:
    # The check is a substring test, so its negative case is what keeps the
    # retry from firing on every question.
    assert _is_refusal("结论：增值税分四档，分别适用……[1][2]") is False


def test_answerer_reports_a_verdict_it_could_not_overturn() -> None:
    hits = [
        KnowledgeHit(id="c1", content="第十二条 分别核算。", source="03.docx"),
        KnowledgeHit(
            id="c2",
            content="第十条 增值税税率：",
            source="03.docx",
            metadata_payload={QUOTA_OVERFLOW_KEY: True},
        ),
    ]
    refusal = f"{INSUFFICIENT_EVIDENCE_MARKER}，缺少税率条款。"
    llm = _StubLLM(contents=[refusal, refusal])

    answer = asyncio.run(_answerer(hits, llm).answer("税率分几档"))

    assert len(llm.requests) == 2
    assert answer.insufficient_evidence is True


def test_answerer_does_not_retry_a_refusal_with_nothing_to_add() -> None:
    hits = [KnowledgeHit(id="c1", content="第十二条 分别核算。", source="03.docx")]
    llm = _StubLLM(content=f"{INSUFFICIENT_EVIDENCE_MARKER}。")

    answer = asyncio.run(_answerer(hits, llm).answer("税率分几档"))

    assert len(llm.requests) == 1
    assert answer.insufficient_evidence is True
