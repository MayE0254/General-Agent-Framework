"""Unit tests for the deterministic structural extraction (no Neo4j, no LLM).

The strings below are verbatim excerpts from the live corpus, not invented
examples: each one is a bridge the multi-hop evaluation set needs, and it was
read out of ``data/knowledge/processed/chunks.jsonl`` before this module was
written.  A regression that drops one of these edges is therefore a regression
on a measured question, not a hypothetical one.
"""

from app.knowledge.graph.extract import (
    TitleIndex,
    cn_to_int,
    doc_id_for,
    extract_structure,
    find_citations,
    find_title_references,
    implementing_law,
    normalize_title,
    parse_section_article,
    short_title,
)


# --------------------------------------------------------------------------
# Chinese numerals
# --------------------------------------------------------------------------
def test_cn_to_int_handles_both_writing_systems() -> None:
    assert cn_to_int("二十七") == 27
    assert cn_to_int("三") == 3
    assert cn_to_int("十") == 10
    assert cn_to_int("十二") == 12
    assert cn_to_int("二十") == 20
    assert cn_to_int("一百零一") == 101
    assert cn_to_int("一百二十") == 120
    assert cn_to_int("三千二百") == 3200
    assert cn_to_int("27") == 27
    assert cn_to_int(" 十 一 ") == 11


def test_cn_to_int_returns_none_instead_of_guessing() -> None:
    # A malformed reference must be dropped, never silently become article 0.
    assert cn_to_int("") is None
    assert cn_to_int("abc") is None
    assert cn_to_int("第") is None


def test_parse_section_article() -> None:
    assert parse_section_article("第二十七条") == 27
    assert parse_section_article("第 10 条") == 10
    # Notices and Excel-derived documents are not split into clauses at all.
    assert parse_section_article("全文") is None
    assert parse_section_article("") is None


# --------------------------------------------------------------------------
# Titles
# --------------------------------------------------------------------------
def test_normalize_and_short_title() -> None:
    assert normalize_title("《中华人民共和国企业所得税法》") == "中华人民共和国企业所得税法"
    assert short_title("中华人民共和国企业所得税法") == "企业所得税法"
    assert short_title("中华人民共和国个人所得税法实施条例") == "个人所得税法实施条例"
    # Nothing to strip: the name is returned unchanged rather than truncated.
    assert short_title("增值税法") == "增值税法"


def test_implementing_law_only_claims_unambiguous_suffixes() -> None:
    assert (
        implementing_law("中华人民共和国企业所得税法实施条例")
        == "中华人民共和国企业所得税法"
    )
    assert (
        implementing_law("中华人民共和国个人所得税法实施条例")
        == "中华人民共和国个人所得税法"
    )
    assert implementing_law("中华人民共和国环境保护税法实施条例") == "中华人民共和国环境保护税法"
    # 暂行条例 / 实施办法 routinely stand alone; claiming an implementation
    # edge from them would be a guess.
    assert implementing_law("中华人民共和国增值税暂行条例") is None
    assert implementing_law("个人所得税专项附加扣除暂行办法") is None
    assert implementing_law("中华人民共和国企业所得税法") is None


def test_title_index_resolves_the_longest_suffix() -> None:
    index = TitleIndex(
        {
            "中华人民共和国企业所得税法": "doc:law",
            "中华人民共和国企业所得税法实施条例": "doc:reg",
        }
    )

    # The name as it appears mid-sentence, with the lead-in the pattern cannot
    # know to exclude.
    assert index.resolve("依据中华人民共和国企业所得税法") == "doc:law"
    assert index.resolve("企业所得税法") == "doc:law"
    assert index.resolve("中华人民共和国企业所得税法实施条例") == "doc:reg"
    assert index.resolve("中华人民共和国增值税法") is None


# --------------------------------------------------------------------------
# Citations
# --------------------------------------------------------------------------
def test_find_citations_reads_a_self_reference_with_a_paragraph() -> None:
    # 中华人民共和国企业所得税法 第四条, verbatim.
    cites = find_citations("第四条企业所得税的税率为25％。 非居民企业取得本法第三条第三款规定的所得，适用税率为20％。")

    assert len(cites) == 1
    cite = cites[0]
    assert cite.law_token == "本法"
    assert cite.article == 3
    assert cite.paragraph == 3
    assert cite.item is None
    assert cite.is_self is True


def test_find_citations_reads_a_cross_document_reference() -> None:
    # 中华人民共和国企业所得税法实施条例 第九十一条, verbatim: the law name
    # carries no 书名号 here, which is why the bare-name branch exists.
    cites = find_citations(
        "第九十一条 非居民企业取得企业所得税法第二十七条第（五）项规定的所得，"
        "减按10%的税率征收企业所得税。"
    )

    assert len(cites) == 1
    assert cites[0].article == 27
    assert cites[0].item == 5
    assert cites[0].is_self is False
    # The lead-in before the law name survives in the token; the index resolves
    # it by longest suffix instead of requiring a clean capture.
    assert cites[0].law_token.endswith("企业所得税法")


def test_find_citations_accepts_halfwidth_parentheses() -> None:
    # 实施条例 第八十六条 writes 第(一)项 with ASCII parentheses while the law
    # itself writes （五）; both spellings occur in the corpus.
    half = find_citations("第八十六条 企业所得税法第二十七条第(一)项规定的企业从事")
    full = find_citations("（五）本法第三条第三款规定的所得")

    assert half and half[0].item == 1
    assert half[0].article == 27
    assert full and full[0].paragraph == 3


def test_find_citations_never_matches_a_clause_header() -> None:
    # Every clause opens with its own number, so a bare 第X条 must not match --
    # if it did, every clause would cite itself and the CITES edges would be
    # almost entirely self-loops.
    assert find_citations("第二十七条企业的下列所得，可以免征、减征企业所得税：") == []
    assert find_citations("第十条增值税税率：") == []
    assert find_citations("第一条 为了规范增值税的征收和缴纳，制定本法。") == []


def test_find_title_references() -> None:
    text = (
        "依据《中华人民共和国个人所得税法》有关规定，国务院决定，"
        "提高3岁以下婴幼儿照护等三项个人所得税专项附加扣除标准。"
        "四、……按照《个人所得税专项附加扣除暂行办法》有关规定执行。"
        "（另见《中华人民共和国个人所得税法》第六条。）"
    )

    refs = find_title_references(text)

    # Deduplicated, order-preserving.
    assert refs == ["中华人民共和国个人所得税法", "个人所得税专项附加扣除暂行办法"]


# --------------------------------------------------------------------------
# Snapshot -> graph shape, on a miniature of the mh-cit-001 chain
# --------------------------------------------------------------------------
LAW = "13_中华人民共和国企业所得税法.docx"
REG = "086_中华人民共和国企业所得税法实施条例.docx"
IIT = "14_中华人民共和国个人所得税法.docx"
NOTICE = "141_国务院关于提高个人所得税有关专项附加扣除标准的通知.docx"


def _rec(source, title, section, parent_id, content, cid=None) -> dict:
    return {
        "id": cid or f"{parent_id}-c0",
        "title": title,
        "content": content,
        "source": source,
        "parent_id": parent_id,
        "parent_text": "",
        "metadata_payload": {"section": section},
    }


def _snapshot() -> list[dict]:
    return [
        _rec(
            LAW,
            "中华人民共和国企业所得税法",
            "第三条",
            "p3",
            "第三条 居民企业应当就其来源于中国境内、境外的所得缴纳企业所得税。"
            "非居民企业在中国境内未设立机构、场所的，或者虽设立机构、场所但取得的所得"
            "与其所设机构、场所没有实际联系的，应当就其来源于中国境内的所得缴纳企业所得税。",
        ),
        _rec(
            LAW,
            "中华人民共和国企业所得税法",
            "第四条",
            "p4",
            "第四条企业所得税的税率为25％。 非居民企业取得本法第三条第三款规定的所得，"
            "适用税率为20％。",
        ),
        _rec(
            LAW,
            "中华人民共和国企业所得税法",
            "第二十七条",
            "p27",
            "第二十七条企业的下列所得，可以免征、减征企业所得税："
            "（五）本法第三条第三款规定的所得。",
        ),
        _rec(
            REG,
            "中华人民共和国企业所得税法实施条例",
            "第九十一条",
            "p91",
            "第九十一条 非居民企业取得企业所得税法第二十七条第（五）项规定的所得，"
            "减按10%的税率征收企业所得税。",
        ),
        _rec(
            IIT,
            "中华人民共和国个人所得税法",
            "第六条",
            "p6",
            "第六条 专项附加扣除，包括子女教育、继续教育、大病医疗、住房贷款利息或者"
            "住房租金、赡养老人等支出，具体范围、标准和实施步骤由国务院确定，"
            "并报全国人民代表大会常务委员会备案。",
        ),
        _rec(
            NOTICE,
            "国务院关于提高个人所得税有关专项附加扣除标准的通知",
            "全文",
            "pfull",
            "国务院关于提高个人所得税有关专项附加扣除标准的通知 "
            "为进一步减轻家庭生育养育和赡养老人的支出负担，依据《中华人民共和国个人所得税法》"
            "有关规定，国务院决定，提高3岁以下婴幼儿照护等三项个人所得税专项附加扣除标准。 "
            "四、……按照《个人所得税专项附加扣除暂行办法》有关规定执行。",
        ),
    ]


def _edge_pairs(extraction) -> set[tuple[str, str]]:
    return {(e.src_clause_id, e.dst_clause_id) for e in extraction.citations}


def test_extract_structure_builds_the_measured_citation_chain() -> None:
    extraction = extract_structure(_snapshot())
    pairs = _edge_pairs(extraction)

    # The two intra-document hops mh-cit-001 needs...
    assert ("p4", "p3") in pairs, "第四条 must cite 第三条"
    assert ("p27", "p3") in pairs, "第二十七条 must cite 第三条"
    # ...and the cross-document hop that was missing from retrieval, which only
    # exists because a bare (unbracketed) law name resolves by title.
    assert ("p91", "p27") in pairs, "实施条例第九十一条 must cite 企税法第二十七条"

    cross = [e for e in extraction.citations if e.props["cross_doc"]]
    assert len(cross) == 1
    assert cross[0].props["item"] == 5
    assert cross[0].props["law_token"].endswith("企业所得税法")


def test_extract_structure_links_a_notice_to_the_law_it_invokes() -> None:
    extraction = extract_structure(_snapshot())

    # 个税法第六条 itself never names the notice (it delegates to 国务院), so
    # the only handle is the notice citing the law -- which is why expansion
    # has to be direction-agnostic rather than following edges forward.
    assert extraction.references == [
        (doc_id_for(NOTICE), doc_id_for(IIT))
    ]


def test_extract_structure_links_a_regulation_to_its_law() -> None:
    extraction = extract_structure(_snapshot())

    assert extraction.implements == [(doc_id_for(REG), doc_id_for(LAW))]


def test_extract_structure_keeps_one_node_per_parent_not_per_article() -> None:
    records = _snapshot()
    # A clause longer than chunk_size_chars is split into several parents that
    # share a section label; each must stay its own node, otherwise a node
    # several times the expected size is handed back by expand_to_parent.
    records.append(
        _rec(
            LAW,
            "中华人民共和国企业所得税法",
            "第二十七条",
            "p27b",
            "第二十七条（续）其他所得，依照本法第三条第三款的规定处理。",
            cid="p27b-c0",
        )
    )

    extraction = extract_structure(records)

    assert {c.clause_id for c in extraction.clauses} >= {"p27", "p27b"}
    node = next(c for c in extraction.clauses if c.clause_id == "p27b")
    assert node.section == "第二十七条"
    assert node.article == 27


def test_extract_structure_prefers_parent_text_and_keeps_every_child_id() -> None:
    records = [
        {
            "id": "c0",
            "title": "测试法",
            "content": "第一条 前半句。",
            "source": "x_测试法.docx",
            "parent_id": "px",
            "parent_text": "第一条 前半句。后半句。",
            "metadata_payload": {"section": "第一条"},
        },
        {
            "id": "c1",
            "title": "测试法",
            "content": "后半句。",
            "source": "x_测试法.docx",
            "parent_id": "px",
            "parent_text": "第一条 前半句。后半句。",
            "metadata_payload": {"section": "第一条"},
        },
    ]

    extraction = extract_structure(records)

    assert len(extraction.clauses) == 1
    clause = extraction.clauses[0]
    assert clause.text == "第一条 前半句。后半句。"
    assert clause.chunk_ids == ["c0", "c1"]


def test_extract_structure_reports_what_it_could_not_resolve() -> None:
    extraction = extract_structure(_snapshot())

    # 《个人所得税专项附加扣除暂行办法》 is cited but is not in this mini
    # corpus: the name resolves to no document and the edge is dropped.  The
    # count exists so that loss is visible instead of silent.
    assert extraction.stats["references"] == 1
    assert extraction.stats["documents"] == 4
    assert extraction.stats["clauses"] == 6
    assert extraction.stats["implements"] == 1


def test_extract_structure_handles_an_empty_corpus() -> None:
    extraction = extract_structure([])

    assert extraction.clauses == []
    assert extraction.citations == []
    assert extraction.references == []
    assert extraction.stats["clauses"] == 0
