"""实体层质量度量的单元测试（不连 Neo4j、不调模型）。

``app.knowledge.graph.audit`` 里的每个函数都是纯函数，输入是快照记录、条款列表
与预计算的计数。这里锁的是三件容易悄悄错掉的事：

* **分母语义** —— 空分母必须是 ``None`` 而不是 ``0.0``，否则「没问过」会被印成
  「0%」；
* **上限的指纹** —— ``6+`` 桶恒为 0 是 ``MAX_ENTITIES_PER_CLAUSE`` 咬住长枚举
  条款的证据，分桶口径一变，这个证据就消失了；
* **两个机制不能合并** —— ``cites`` 与 ``mentions`` 各自连不连得上，指向完全
  不同的修法（结构层正则 vs 实体层抽取），合成一个「图覆盖率」就再也分不清。
"""

from app.knowledge.graph.audit import (
    BUCKETS,
    Endpoint,
    anchor_coverage,
    classify_mention,
    clause_mention_counts,
    degree_summary,
    hop_coverage,
    last_wins,
    mention_literal_audit,
    mentions_per_clause,
    parse_reference_endpoints,
    ratio,
    squeeze_text,
    summarise_snapshot,
    target_ranks,
)
from app.knowledge.graph.extract import ClauseRecord


def _clause(
    clause_id: str,
    *,
    doc_id: str = "doc:企业所得税法.docx",
    text: str = "第一条 在中华人民共和国境内，企业和其他取得收入的组织为企业所得税的纳税人。",
    article: int | None = 1,
) -> ClauseRecord:
    return ClauseRecord(
        clause_id=clause_id,
        doc_id=doc_id,
        source_rel="企业所得税法.docx",
        title="中华人民共和国企业所得税法",
        section=f"第{article}条" if article else "全文",
        article=article,
        text=text,
    )


def _record(doc_id: str, *, status: str = "ok", stats: dict | None = None,
            mentions: list[dict] | None = None) -> dict:
    return {
        "doc_id": doc_id,
        "status": status,
        "stats": stats or {},
        "mentions": mentions or [],
    }


# --------------------------------------------------------------------------
# ratio / last_wins
# --------------------------------------------------------------------------
def test_ratio_returns_none_for_an_empty_denominator() -> None:
    # 「没问过」与「全都没命中」在报告里不能长成一个样子。
    assert ratio(3, 0) is None
    assert ratio(0, 0) is None
    assert ratio(1, 4) == 0.25
    assert ratio(0, 4) == 0.0


def test_last_wins_matches_the_builder_snapshot_rule() -> None:
    # 快照是按行追加的，重跑一篇会再追加一行；读侧必须与 build_entities 同规则。
    first = _record("doc:a", stats={"mentions": 1}, mentions=[{"clause_id": "c1"}])
    second = _record("doc:b")
    retry = _record("doc:a", stats={"mentions": 9}, mentions=[{"clause_id": "c2"}])
    assert last_wins([first, second, retry]) == {"doc:a": retry, "doc:b": second}


def test_last_wins_skips_rows_without_a_doc_id() -> None:
    assert last_wins([{"status": "ok"}, None, _record("doc:a")]) == {
        "doc:a": _record("doc:a")
    }


# --------------------------------------------------------------------------
# 账本 P3
# --------------------------------------------------------------------------
def test_summarise_snapshot_sums_discards_and_keeps_the_ratio() -> None:
    docs = {
        "doc:a": _record(
            "doc:a",
            stats={
                "refs_sent": 20, "refs_returned": 20, "refs_missing": 0,
                "raw_entities": 10, "unknown_types": 1, "invalid_names": 2,
                "duplicates": 3, "over_limit": 4, "mentions": 5, "entities": 4,
            },
        ),
        "doc:b": _record("doc:b", stats={"refs_sent": 10, "refs_returned": 8,
                                         "refs_missing": 2, "raw_entities": 6,
                                         "over_limit": 0}),
        "doc:c": _record("doc:c", status="error"),
    }
    row = summarise_snapshot(docs)
    assert row["docs_total"] == 3
    assert row["docs_ok"] == 2
    assert row["docs_error"] == 1
    assert row["stats"]["refs_sent"] == 30
    # 失败文档不产生任何计数：它的 0 必须是"没跑"，不能混进分母。
    assert row["stats"]["refs_missing"] == 2
    assert row["discards"] == {
        "unknown_types": 1, "invalid_names": 2, "duplicates": 3, "over_limit": 4,
    }
    assert row["discarded"] == 10
    assert row["discard_ratio"] == 10 / 16
    assert row["ref_loss"] == 2 / 30


def test_summarise_snapshot_with_no_documents_has_no_rate() -> None:
    row = summarise_snapshot({})
    assert row["discard_ratio"] is None
    assert row["ref_loss"] is None


def test_clause_mention_counts_ignores_failed_documents() -> None:
    docs = {
        "doc:a": _record("doc:a", mentions=[{"clause_id": "c1"}, {"clause_id": "c1"}]),
        "doc:b": _record("doc:b", status="error", mentions=[{"clause_id": "c2"}]),
    }
    assert clause_mention_counts(docs) == {"c1": 2}


# --------------------------------------------------------------------------
# 锚点数分布
# --------------------------------------------------------------------------
def test_mentions_per_clause_splits_long_enumerations_from_ordinary_clauses() -> None:
    # 长枚举条款 = 正文含 ≥8 个顿号；普通条款是它 52.4% 无锚点的对照。
    enum_text = "、".join(["货物"] * 9)
    texts = {"enum": enum_text, "plain": "第一条 企业所得税的税率为25%。"}
    counts = {"enum": 5, "plain": 0}
    row = mentions_per_clause(texts, counts)
    assert row["groups"]["长枚举"]["3-5"] == 1
    assert row["groups"]["普通"]["0"] == 1
    assert row["groups"]["长枚举"]["合计"] == 1
    assert row["groups"]["普通"]["有锚点"] == 0


def test_mentions_per_clause_counts_clauses_absent_from_the_snapshot_as_zero() -> None:
    row = mentions_per_clause({"c1": "短短一句", "c2": "另一句"}, {"c1": 1})
    assert row["groups"]["普通"]["0"] == 1
    assert row["groups"]["普通"]["1-2"] == 1
    assert row["anchored"] == 1
    assert sum(row["groups"]["普通"][b] for b in BUCKETS) == 2


# --------------------------------------------------------------------------
# 锚点覆盖 L1
# --------------------------------------------------------------------------
def test_anchor_coverage_narrows_the_denominator_layer_by_layer() -> None:
    clauses = [
        _clause("long-anchored", text="第一条 " + "纳税人" * 20),
        _clause("long-bare", text="第二条 " + "计税依据" * 20),
        # 这一条**故意写长**：它要能活过"≥40 字"那层，只能被程序性规则拦下，
        # 否则这条测试就分不清是长度在起作用还是规则在起作用。
        _clause("procedural", doc_id="doc:通知.docx",
                text="本通知自2023年1月1日起施行，此前有关规定与本通知不一致的，"
                     "以本通知为准，同时废止。"),
        _clause("short", doc_id="doc:通知.docx", text="特此公告"),
        _clause("failed-doc", doc_id="doc:坏.docx", text="第三条 " + "税" * 40),
    ]
    row = anchor_coverage(
        clauses, {"long-anchored", "procedural", "failed-doc"},
        error_docs={"doc:坏.docx"},
    )
    all_layer, no_error, long_only, no_procedural = row["layers"]

    assert (all_layer["clauses"], all_layer["anchored"]) == (5, 3)
    # 抽取失败的文档整篇无锚点，那是调用失败而不是"模型说没有实体"。
    assert (no_error["clauses"], no_error["anchored"]) == (4, 2)
    assert long_only["clauses"] == 3
    # 施行/废止句里出现"废止"字样，但它是程序性的，不该拉低覆盖率。
    assert no_procedural["clauses"] == 2
    assert no_procedural["anchored"] == 1
    assert no_procedural["coverage"] == 0.5
    assert row["anchored_total"] == 3
    assert row["anchored_orphan"] == 0
    # 程序性规则筛掉的那批单独记账，分母是**它自己**（第三层里被它去掉的那些），
    # 不是全部条款：这里只有 procedural 一条（short 早在长度那层就出去了），
    # 而它恰好有锚点 → 密度 1.0 > 第三层的 2/3。这与实测语料同向（87% 有锚点），
    # 正是这一栏存在的意义：让「筛掉的都是讲不出实体的条款」这个假设可被否证。
    assert row["procedural_dropped"] == {
        "clauses": 1, "anchored": 1, "coverage": 1.0,
    }


def test_anchor_coverage_flags_anchors_that_point_at_missing_clauses() -> None:
    # 快照里指向不存在条款的锚点 = 两份快照不同代，此时任何覆盖率都不可信。
    row = anchor_coverage([_clause("c1")], {"c1", "已经从chunk快照消失的条款"})
    assert row["anchored_orphan"] == 1
    assert row["anchored_total"] == 1


def test_anchor_coverage_with_no_clauses_reports_none_coverage() -> None:
    row = anchor_coverage([], set())
    assert row["layers"][0]["coverage"] is None


# --------------------------------------------------------------------------
# 逐跳覆盖 L2
# --------------------------------------------------------------------------
def _resolver(mapping: dict[tuple[str, int], str], docs: dict[str, list[str]]):
    def resolve_clause(law_token: str, article: int) -> str | None:
        return mapping.get((law_token, article))

    def resolve_doc(name: str) -> list[str]:
        return docs.get(name, [])

    return resolve_clause, resolve_doc


def test_parse_reference_endpoints_reads_citations_and_whole_documents() -> None:
    reference = (
        "法定税率为20%。依据：《中华人民共和国企业所得税法》第四条，"
        "以及《中华人民共和国企业所得税法实施条例》第九十一条。"
        "详见《适用9增值税税率货物范围注释》。"
    )
    resolve_clause, resolve_doc = _resolver(
        {
            ("《中华人民共和国企业所得税法》", 4): "c-cit-4",
            ("《中华人民共和国企业所得税法实施条例》", 91): "c-impl-91",
        },
        {"适用9增值税税率货物范围注释": ["n1", "n2"]},
    )
    endpoints, unresolved = parse_reference_endpoints(
        reference, resolve_clause=resolve_clause, resolve_doc=resolve_doc
    )
    assert [e.clause_ids for e in endpoints] == [
        ("c-cit-4",), ("c-impl-91",), ("n1", "n2"),
    ]
    assert unresolved == []


def test_parse_reference_endpoints_reports_what_it_could_not_resolve() -> None:
    # 解析不出的一律原样返回：算不了的跳与不成立的跳必须能区分。
    reference = "《中华人民共和国企业所得税法》第二十七条第（五）项规定本法第三条第三款规定的所得。"
    resolve_clause, resolve_doc = _resolver({}, {})
    endpoints, unresolved = parse_reference_endpoints(
        reference, resolve_clause=resolve_clause, resolve_doc=resolve_doc
    )
    assert endpoints == []
    # 「本法」在评测集作者的综述文字里没有宿主，只能原样报出来。
    assert any(item.startswith("本法第三条") for item in unresolved)
    assert any(item.startswith("《中华人民共和国企业所得税法》第二十七条")
               for item in unresolved)


def test_parse_reference_endpoints_does_not_emit_a_document_twice() -> None:
    # 已经按条号落地的法规，不再以"整篇"重复计入，否则端点集会被整篇条款淹没。
    reference = "《中华人民共和国个人所得税法》第六条与《中华人民共和国个人所得税法》。"
    resolve_clause, resolve_doc = _resolver(
        {("《中华人民共和国个人所得税法》", 6): "c-iit-6"},
        {"中华人民共和国个人所得税法": ["c-iit-6", "c-iit-7"]},
    )
    endpoints, _ = parse_reference_endpoints(
        reference, resolve_clause=resolve_clause, resolve_doc=resolve_doc
    )
    assert [e.clause_ids for e in endpoints] == [("c-iit-6",)]


def test_hop_coverage_keeps_the_two_mechanisms_apart() -> None:
    a = Endpoint("甲条", ("a",))
    b = Endpoint("乙条", ("b",))
    c = Endpoint("丙条", ("c",))
    row = hop_coverage(
        [a, b, c],
        cites_links=lambda left, right: 1 if {"a", "b"} == {left[0], right[0]} else 0,
        mentions_shared=lambda left, right: 2 if {"a", "c"} == {left[0], right[0]} else 0,
        min_shared=2,
    )
    assert row["pairs_total"] == 3
    assert row["pairs_by_cites"] == 1
    assert row["pairs_by_mentions"] == 1
    assert row["pairs_linked"] == 2
    assert row["pairs_unlinked"] == 1
    assert row["complete"] is False
    assert [p["linked_by_mentions"] for p in row["pairs"]] == [False, True, False]


def test_hop_coverage_applies_the_production_threshold() -> None:
    # 覆盖率必须按生产路径的 min_shared_entities 判定，否则会得到
    # "桥存在但从不过滤器"的假阳性。
    endpoints = [Endpoint("甲", ("a",)), Endpoint("乙", ("b",))]
    shared = {1: 1, 2: 0, 3: 0}
    for threshold, expected in shared.items():
        row = hop_coverage(
            endpoints,
            cites_links=lambda *_: 0,
            mentions_shared=lambda *_: 1,
            min_shared=threshold,
        )
        assert row["pairs_by_mentions"] == (1 if expected else 0)


def test_hop_coverage_with_a_single_endpoint_is_not_complete() -> None:
    row = hop_coverage(
        [Endpoint("甲", ("a",))],
        cites_links=lambda *_: 0,
        mentions_shared=lambda *_: 0,
        min_shared=1,
    )
    assert row["pairs_total"] == 0
    assert row["complete"] is False


# --------------------------------------------------------------------------
# 度分布 P2
# --------------------------------------------------------------------------
def test_degree_summary_reports_both_ends_of_the_distribution() -> None:
    row = degree_summary([1, 2, 3, 4, 5], thresholds=(2, 4))
    assert row["entities"] == 5
    assert row["p50"] == 3
    assert row["p90"] == 5
    assert row["max"] == 5
    assert row["mean"] == 3.0
    assert row["singleton"] == 1
    assert row["singleton_ratio"] == 0.2
    assert [(h["threshold"], h["entities"]) for h in row["hubs"]] == [(2, 4), (4, 2)]


def test_degree_summary_of_an_empty_graph_has_no_percentiles() -> None:
    row = degree_summary([])
    assert row["entities"] == 0
    assert row["p50"] is None
    assert row["max"] is None
    assert row["mean"] is None
    assert row["singleton_ratio"] is None
    assert all(h["ratio"] is None for h in row["hubs"])


# --------------------------------------------------------------------------
# 预算内可达 L3
# --------------------------------------------------------------------------
def test_target_ranks_reports_the_first_hit_and_the_count() -> None:
    row = target_ranks(["x", "t1", "y", "t2"], {"t1", "t2"})
    assert row == {"total_neighbors": 4, "hits": 2, "first_rank": 2}


def test_target_ranks_distinguishes_absent_from_never_run() -> None:
    # 名次为 None 有两种含义——腿没跑、跑了没召回——调用方必须自己区分，
    # 这里只保证"跑了一次空结果"不会被误报成命中。
    assert target_ranks(["x", "y"], {"t"}) == {
        "total_neighbors": 2, "hits": 0, "first_rank": None,
    }
    assert target_ranks([], {"t"})["first_rank"] is None


# --------------------------------------------------------------------------
# 字面性审计 P1a
# --------------------------------------------------------------------------
def test_squeeze_text_removes_whitespace_but_keeps_punctuation() -> None:
    # 删标点会把「，」「、」两侧的词粘成一个不存在的名字，凭空造出命中。
    assert squeeze_text("第一条，企业所得税\t的\n税率为25% 。") == \
        "第一条，企业所得税的税率为25%。"
    assert squeeze_text("") == ""
    assert squeeze_text(None) == ""


def test_classify_mention_three_ways() -> None:
    text = squeeze_text("第一条 企业所得税的税率为25%")
    # 全称是正文子串 → exact（《》在归一化时被剥掉）。
    assert classify_mention("企业所得税", text) == "exact"
    assert classify_mention("《企业所得税》", text) == "exact"
    # 全称不在，但 3 字片段在正文里 → partial（正文用了简称）。
    assert classify_mention("增值税小规模纳税人", squeeze_text("小规模纳税人适用3%征收率")) \
        == "partial"
    # 一个字都对不上 → absent。
    assert classify_mention("环境保护税", text) == "absent"


def test_classify_mention_treats_empty_names_as_absent() -> None:
    # 归一化后为空的名字不能被当成"命中空串"而落进 exact。
    assert classify_mention("", "任何正文") == "absent"
    assert classify_mention("  ", "任何正文") == "absent"


def test_classify_mention_does_not_partial_match_short_names() -> None:
    # 2 字名称够不到 3 字窗口：不给它猜的机会，直接 absent。放宽这一条会让
    # 「印花」去蹭满篇的「花」字，absent 上界立刻失去意义。
    assert classify_mention("印花", squeeze_text("企业所得税的税率为25%")) == "absent"


def test_mention_literal_audit_on_empty_input_has_no_rates() -> None:
    row = mention_literal_audit({}, {})
    assert row["mentions"] == 0
    assert row["resolved"] == 0
    assert row["matched"] == {"exact": 0, "partial": 0, "absent": 0}
    assert row["absent_ratio"] is None
    assert row["partial_ratio"] is None
    assert row["by_type"] == {}
    # 空输入下每一带都要在，且是全键，不能留空洞。
    assert [band for band, _ in row["by_degree"].items()] == \
        ["1", "2-3", "4-9", "10-49", "50+"]
    assert all(group["mentions"] == 0 and group["absent_ratio"] is None
               for group in row["by_degree"].values())
    assert all(items == [] for items in row["samples"].values())


def test_mention_literal_audit_counts_edges_whose_clause_is_gone() -> None:
    # 边指向 chunk 快照里不存在的条款 = 两份快照不同代；这类边不能混进比例。
    docs = {"doc:a": _record("doc:a", mentions=[
        {"clause_id": "c1", "type": "tax_type", "name": "企业所得税"},
        {"clause_id": "已消失", "type": "tax_type", "name": "企业所得税"},
    ])}
    row = mention_literal_audit(docs, {"c1": "第一条 企业所得税。"})
    assert row["mentions"] == 2
    assert row["unresolved_clauses"] == 1
    assert row["resolved"] == 1


def test_mention_literal_audit_tabulates_by_type_and_degree() -> None:
    clause_texts = {
        "c1": "第一条 企业所得税的税率为25%，非居民企业适用20%的税率。",
        "c2": "第二条 依照本法缴纳企业所得税的居民企业。",
    }
    docs = {"doc:a": _record("doc:a", mentions=[
        {"clause_id": "c1", "type": "tax_type", "name": "企业所得税"},   # exact
        {"clause_id": "c1", "type": "taxpayer", "name": "非居民企业"},   # exact
        {"clause_id": "c1", "type": "tax_matter", "name": "环境保护税"},  # absent
        {"clause_id": "c2", "type": "tax_type", "name": "企业所得税"},   # exact（度=2）
        {"clause_id": "c2", "type": "tax_type", "name": "增值税"},       # absent
    ])}
    row = mention_literal_audit(docs, clause_texts)
    assert row["matched"] == {"exact": 3, "partial": 0, "absent": 2}
    assert row["resolved"] == 5
    assert row["absent_ratio"] == 2 / 5
    assert row["partial_ratio"] == 0.0

    by_type = row["by_type"]
    assert by_type["tax_type"] == {
        "exact": 2, "partial": 0, "absent": 1, "mentions": 3,
        "absent_ratio": 1 / 3,
    }
    assert by_type["taxpayer"]["absent_ratio"] == 0.0
    assert by_type["tax_matter"]["absent_ratio"] == 1.0

    # 度由快照自己算：企业所得税 出现在 2 条边 → 落 2-3 带；其余各 1 条 → 1 带。
    assert row["by_degree"]["1"] == {
        "exact": 1, "partial": 0, "absent": 2, "mentions": 3,
        "absent_ratio": 2 / 3,
    }
    assert row["by_degree"]["2-3"] == {
        "exact": 2, "partial": 0, "absent": 0, "mentions": 2,
        "absent_ratio": 0.0,
    }
    assert row["by_degree"]["50+"]["mentions"] == 0

    assert [item["name"] for item in row["samples"]["absent"]] == \
        ["环境保护税", "增值税"]
    assert row["samples"]["partial"] == []


def test_mention_literal_audit_caps_the_sample_list() -> None:
    docs = {"doc:a": _record("doc:a", mentions=[
        {"clause_id": "c1", "type": "tax_type", "name": f"税种{i}"} for i in range(4)
    ])}
    row = mention_literal_audit(docs, {"c1": "第一条 企业所得税。"}, sample_limit=2)
    assert row["matched"]["absent"] == 4
    assert len(row["samples"]["absent"]) == 2
