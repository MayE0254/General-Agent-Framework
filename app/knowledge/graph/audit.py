"""实体层质量的**零 LLM**度量：覆盖率与精度形状。

实体层存在的唯一理由是让 ``Clause -MENTIONS-> Entity <-MENTIONS- Clause`` 这条
两跳把「共用主题却互不引用」的条款连起来（国发〔2022〕8号 与 国发〔2023〕13号
都讲 3 岁以下婴幼儿照护扣除，谁也没提谁）。因此它的好坏不是通用 NER 的 P/R：

* **覆盖率** —— 评测集真正需要的那几跳，图里存在几条、能不能活过扩散腿的上限
  （``neighbors_per_seed`` / ``max_results``）。桥存在但排在第 40 名，对检索
  而言与桥不存在是同一件事。
* **精度形状** —— 写进去的实体里哪些无害、哪些在稀释两跳邻域。hub 实体
  （``增值税`` 被上千条款提到）**事实正确却有害**：共享实体数当权重是 3~5，
  合并排序时系统性压掉权重为 1 的 CITES 邻居——这正是 2026-09-30 实测到的
  mentions 腿负收益机制。

本模块不调用任何模型。所有数字都来自抽取快照（``entities.jsonl``）、chunk 快照
与图，所以可复算、可 diff——而评判模型对逐字相同输入的波动实测 0.4~0.58，比实体
层的任何改动都大；用 RAGAS 分辨这一层等于拿噪声当尺子。

每个比例都连同分子分母一起返回，由调用方打印：只给百分比看不出是哪一侧动了。
"""
from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from app.knowledge.graph.entities import normalize_entity_name
from app.knowledge.graph.extract import ClauseRecord, find_citations, find_title_references

# --------------------------------------------------------------------------
# 抽取快照账本：模型被问了什么、丢掉了什么
# --------------------------------------------------------------------------
# ``parse_extraction`` 逐文档记下的计数项，这里只是把它们汇总起来。
STAT_KEYS: tuple[str, ...] = (
    "refs_sent",
    "refs_returned",
    "refs_missing",
    "refs_unknown",
    "raw_entities",
    "unknown_types",
    "invalid_names",
    "duplicates",
    "over_limit",
    "mentions",
    "entities",
)
# 四项「丢弃」的原因：模型给了我们没要的东西（unknown_types）、名字不合法
# （invalid_names）、同条款内重复（duplicates）、超过每条款上限（over_limit）。
# 分开列是因为它们指向完全不同的修法：词表、正则、去重、上限。
DISCARD_KEYS: tuple[str, ...] = ("unknown_types", "invalid_names", "duplicates", "over_limit")


def ratio(part: int, whole: int) -> float | None:
    """``part/whole``；分母为空时返回 ``None``。

    返回 ``None`` 而不是 ``0.0``：空分母意味着这个问题**没有被问过**，把
    「没问」印成「0%」是测量变成假陈述的经典路径（评测里缺失的指标同理，见
    ``app.knowledge.evaluation.runner``）。
    """
    return None if not whole else part / whole


def last_wins(records: Iterable[dict]) -> dict[str, dict]:
    """按 ``doc_id`` 取最后一行，与 ``build_entities._load_results`` 同规则。

    ``build_entities.py`` 的快照是**追加**写的，重跑一篇文档会再追加一行，
    读侧以最后一行为准。这里必须复刻同一条规则，否则同一个快照在脚本里和在
    建图脚本里是两个不同的东西。
    """
    out: dict[str, dict] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        doc_id = record.get("doc_id")
        if doc_id:
            out[doc_id] = record
    return out


def summarise_snapshot(docs: dict[str, dict]) -> dict:
    """汇总整份抽取快照的账本与丢弃分解。

    注意口径：一篇文档可能被切成多个批次调用，``refs_*`` 是**逐次调用**的计数，
    因此这里的和是「调用次数之和」而不是「文档数」。``refs_missing`` 是其中唯一
    直接指向模型行为的项——我们给了 N 个条款编号，它只回执了 M 个。
    """
    totals = {key: 0 for key in STAT_KEYS}
    ok = 0
    error = 0
    for record in docs.values():
        if record.get("status") != "ok":
            error += 1
            continue
        ok += 1
        stats = record.get("stats") or {}
        for key in STAT_KEYS:
            totals[key] += int(stats.get(key) or 0)
    discarded = sum(totals[key] for key in DISCARD_KEYS)
    return {
        "docs_total": len(docs),
        "docs_ok": ok,
        "docs_error": error,
        "stats": totals,
        "discards": {key: totals[key] for key in DISCARD_KEYS},
        "discarded": discarded,
        "discard_ratio": ratio(discarded, totals["raw_entities"]),
        "ref_loss": ratio(totals["refs_missing"], totals["refs_sent"]),
    }


def clause_mention_counts(docs: dict[str, dict]) -> dict[str, int]:
    """``clause_id -> 该条款被抽出的实体个数``（失败文档不计）。"""
    counts: dict[str, int] = {}
    for record in docs.values():
        if record.get("status") != "ok":
            continue
        for mention in record.get("mentions") or []:
            clause_id = str(mention.get("clause_id") or "")
            if clause_id:
                counts[clause_id] = counts.get(clause_id, 0) + 1
    return counts


# §5.3 的分桶口径：含 ≥8 个顿号的正文算「长枚举」条款。
ENUM_MARKS = 8
BUCKETS: tuple[str, ...] = ("0", "1-2", "3-5", "6+")


def _bucket(count: int) -> str:
    if count <= 0:
        return "0"
    if count <= 2:
        return "1-2"
    if count <= 5:
        return "3-5"
    return "6+"


def mentions_per_clause(
    clause_texts: dict[str, str],
    counts: dict[str, int],
    *,
    enum_marks: int = ENUM_MARKS,
) -> dict:
    """条款的锚点数分布，长枚举条款与普通条款分开。

    分开不是装饰：长枚举条款是通往「注释 / 清单 / 目录」类附件文档的那一类，
    而 ``MAX_ENTITIES_PER_CLAUSE = 5`` 的上限**只在这类条款上咬得住**。合在一起
    算平均，上限的痕迹就被普通条款的稀疏平均掉了。``6+`` 恒为 0 就是上限的
    指纹（2026-09-30 实体层文档 §5.3）。
    """
    groups: dict[str, dict[str, int]] = {
        "长枚举": {key: 0 for key in BUCKETS},
        "普通": {key: 0 for key in BUCKETS},
    }
    for clause_id, text in clause_texts.items():
        name = "长枚举" if (text or "").count("、") >= enum_marks else "普通"
        groups[name][_bucket(counts.get(clause_id, 0))] += 1
    for name, buckets in groups.items():
        buckets["合计"] = sum(buckets.values())
        buckets["有锚点"] = buckets["合计"] - buckets["0"]
    return {
        "enum_marks": enum_marks,
        "groups": groups,
        "clauses": len(clause_texts),
        "anchored": len({cid for cid in counts if cid in clause_texts}),
    }


# --------------------------------------------------------------------------
# 字面性审计 P1a：一条边在条款正文里有没有字面依据（零模型、全量）
# --------------------------------------------------------------------------
# 这是 P1（边正确性）里**不需要模型的那一半**，而且它比需要模型的那一半更结实：
# 抽取 prompt 的规则 1 就是"只抽字面出现的实体"，所以"字面出没出现"是个字符串
# 问题，可以在**全量 47k 条边**上精确回答，不需要抽样、不需要置信区间。
#
# 但它**只回答"有没有依据"，不回答"对不对"**，两者必须分开报：
#   * 依据在，类型仍可能归错（把"环境保护税"归成 tax_base）；
#   * 依据不在，边仍可能正确——prompt 规则 3 要求写规范全称（写"个人所得税"不写
#     "个税"），正文用简称时全称就不是子串了。
# 所以 ``absent`` 是「无据边」的**上界**，不是结论；把上界拆成"模型幻觉"与
# "全称改写"要靠模型，那是 P1b 的双盲判官（选项里必须有"正文里根本没有这个名称"）。
# 本轮先把它全量钉死，P1b 就只需在 ``absent`` 层抽样，样本量小得多也更集中。
MENTION_MATCH_KEYS: tuple[str, ...] = ("exact", "partial", "absent")
# 片段匹配的窗口：3 字。2 字在中文里几乎必然误命中（"企业""所得税"满篇都是），
# 5 字又会让"减按10%的税率"这类带数字的名字整段失配。
PARTIAL_NGRAM = 3


def squeeze_text(text: str) -> str:
    """正文归一化：只去空白，**不动标点**。

    删标点看着更"干净"，实则有害：``《``/``（`` 夹在名字里的情形（``《增值税法》``
    与 ``增值税法``）由子串匹配自然处理，而删标点会把 ``,`` ``、`` 两侧的词粘成
    一个不存在的名字，凭空造出命中。
    """
    return re.sub(r"\s+", "", text or "")


def classify_mention(name: str, squeezed_text: str) -> str:
    """``exact`` / ``partial`` / ``absent``，见本节顶部的口径说明。

    ``squeezed_text`` 必须已经过 ``squeeze_text``（正文很长，逐边归一化是 O(边数
    × 正文长度) 的无谓开销）；``name`` 在这里归一化，它只有几个字。
    """
    normalized = normalize_entity_name(name)
    if not normalized:
        return "absent"
    if normalized in squeezed_text:
        return "exact"
    if len(normalized) >= PARTIAL_NGRAM and any(
        normalized[i : i + PARTIAL_NGRAM] in squeezed_text
        for i in range(len(normalized) - PARTIAL_NGRAM + 1)
    ):
        return "partial"
    return "absent"


# 实体度的分带。用 P2 的 hub 阈值切（10/50/100/500），这样"字面性"与"度分布"
# 两张表说的是同一批实体，可以对着读：若 hub 的 absent 明显低于长尾，说明 hub
# 是"事实正确却有害"，而长尾才是"模型凑数"。
DEGREE_BANDS: tuple[tuple[str, int, int | None], ...] = (
    ("1", 1, 1),
    ("2-3", 2, 3),
    ("4-9", 4, 9),
    ("10-49", 10, 49),
    ("50+", 50, None),
)


def _degree_band(degree: int) -> str:
    for label, low, high in DEGREE_BANDS:
        if degree >= low and (high is None or degree <= high):
            return label
    return DEGREE_BANDS[0][0]


def _with_ratios(row: dict[str, int]) -> dict[str, int | float | None]:
    """给一张 kind 计数表补上合计与 absent 占比；空行也返回全键，不留空洞。"""
    filled = {key: row.get(key, 0) for key in MENTION_MATCH_KEYS}
    total = sum(filled.values())
    return {**filled, "mentions": total, "absent_ratio": ratio(filled["absent"], total)}


def mention_literal_audit(
    docs: dict[str, dict],
    clause_texts: dict[str, str],
    *,
    sample_limit: int = 5,
) -> dict:
    """全量逐边分类，并按类型与实体度交叉制表。

    实体度**由快照自己算**，不查图：``parse_extraction`` 已按条款去重，而
    ``write_entities`` 的 MERGE 边数就是 (条款, 实体) 对数，所以快照里的
    ``count(entity_id)`` 与图中 ``MENTIONS`` 度数同口径。这让本节完全离线。
    """
    squeezed = {cid: squeeze_text(text) for cid, text in clause_texts.items()}
    degrees: dict[str, int] = {}
    for record in docs.values():
        if record.get("status") != "ok":
            continue
        for mention in record.get("mentions") or []:
            entity = f"{mention.get('type') or ''}:{mention.get('name') or ''}"
            degrees[entity] = degrees.get(entity, 0) + 1

    matched = {key: 0 for key in MENTION_MATCH_KEYS}
    by_type: dict[str, dict[str, int]] = {}
    by_degree: dict[str, dict[str, int]] = {}
    samples: dict[str, list[dict]] = {key: [] for key in MENTION_MATCH_KEYS}
    total = 0
    unresolved = 0

    for record in docs.values():
        if record.get("status") != "ok":
            continue
        for mention in record.get("mentions") or []:
            clause_id = str(mention.get("clause_id") or "")
            name = str(mention.get("name") or "")
            type_key = str(mention.get("type") or "")
            total += 1
            text = squeezed.get(clause_id)
            if text is None:
                # 边指向 chunk 快照里不存在的条款：两份快照不同代（与 L1 的
                # ``anchored_orphan`` 同一个漂移探测器），此时任何比例都不可信。
                unresolved += 1
                continue
            kind = classify_mention(name, text)
            matched[kind] += 1
            degree = degrees.get(f"{type_key}:{name}", 0)
            for table, key in ((by_type, type_key), (by_degree, _degree_band(degree))):
                row = table.setdefault(key, {k: 0 for k in MENTION_MATCH_KEYS})
                row[kind] += 1
            if kind != "exact" and len(samples[kind]) < sample_limit:
                samples[kind].append(
                    {"clause_id": clause_id, "type": type_key, "name": name,
                     "degree": degree}
                )

    resolved = sum(matched.values())
    return {
        "mentions": total,
        "resolved": resolved,
        "unresolved_clauses": unresolved,
        "matched": matched,
        "absent_ratio": ratio(matched["absent"], resolved),
        "partial_ratio": ratio(matched["partial"], resolved),
        "by_type": {key: _with_ratios(row) for key, row in sorted(by_type.items())},
        "by_degree": {
            label: _with_ratios(by_degree.get(label, {}))
            for label, _, _ in DEGREE_BANDS
        },
        "samples": samples,
    }


# --------------------------------------------------------------------------
# 锚点覆盖
# --------------------------------------------------------------------------
# 客观上讲不出实体的条款：施行/废止/特此一类纯程序句，句子里没有纳税人、税种、
# 事项、征税对象或征管事项。这是一张**刻意收窄**的规则表——放宽它只会把分母
# 悄悄缩小到覆盖率看起来漂亮为止，那正是这条指标最容易被自欺的地方。
PROCEDURAL_RX = re.compile(
    r"(自[^，。；]{1,20}起(?:施行|执行|生效))"
    r"|((?:予以|同时|决定)?废止)"
    r"|(特此(?:公告|通知|说明))"
)
MIN_ANCHOR_CHARS = 40


def anchor_coverage(
    clauses: Sequence[ClauseRecord],
    anchored: Iterable[str],
    *,
    error_docs: Iterable[str] = (),
    min_chars: int = MIN_ANCHOR_CHARS,
) -> dict:
    """有实体锚点的条款占比，**逐层收窄分母**。

    单个「41.4% 无锚点」是没有分母语义的数字：施行日期条款本就该是空的。所以
    这里给的是四层，让读者自己选一个可辩护的分母，而不是由本模块替它断言一个
    阈值。

    但第四层（程序性规则）**不能被当作"更诚实的分母"**：规则匹配的是"句子里出现
    施行/废止/特此"，而这类句子恰恰长在有实质内容的长条款末尾，于是它删掉的多是
    *有锚点*的条款（2026-09-30 实测的 87%，见 ``docs/2026-09-30_*_实体层零LLM质量
    报告_update.md``）。判定办法在返回值里而不是在这里：``procedural_dropped``
    的 ``coverage`` 一旦高于第三层，就说明这条规则在删真东西，该层的读数不能用。

    剥离出的 ``anchored_orphan`` 是漂移探测器：快照里的锚点若指向 chunk 快照中
    不存在的条款，说明两份快照不同代（重建过图却没重建实体，或反之），此时
    任何覆盖率都不可信。
    """
    clause_ids = {c.clause_id for c in clauses}
    anchored_set = set(anchored)
    failed = set(error_docs)

    layers: list[tuple[str, Callable[[ClauseRecord], bool]]] = [
        ("全部条款（等于图中 Clause 总量）", lambda c: True),
        ("排除抽取失败的文档", lambda c: c.doc_id not in failed),
        (
            f"再排除正文 < {min_chars} 字",
            lambda c: c.doc_id not in failed and len(c.text or "") >= min_chars,
        ),
        (
            "再排除纯程序性条款（施行/废止/特此）",
            lambda c: c.doc_id not in failed
            and len(c.text or "") >= min_chars
            and not PROCEDURAL_RX.search(c.text or ""),
        ),
    ]
    rows = []
    kept_per_layer: list[list[ClauseRecord]] = []
    for name, keep in layers:
        kept = [c for c in clauses if keep(c)]
        kept_per_layer.append(kept)
        hit = sum(1 for c in kept if c.clause_id in anchored_set)
        rows.append(
            {
                "layer": name,
                "clauses": len(kept),
                "anchored": hit,
                "coverage": ratio(hit, len(kept)),
            }
        )
    # 第四层筛掉那批的锚点密度：高于第三层即证明这条规则在删有实质内容的条款。
    dropped = [c for c in kept_per_layer[-2] if not layers[-1][1](c)]
    dropped_anchored = sum(1 for c in dropped if c.clause_id in anchored_set)
    return {
        "layers": rows,
        "clauses_total": len(clauses),
        "anchored_total": len(anchored_set & clause_ids),
        "anchored_orphan": len(anchored_set - clause_ids),
        "procedural_dropped": {
            "clauses": len(dropped),
            "anchored": dropped_anchored,
            "coverage": ratio(dropped_anchored, len(dropped)),
        },
    }


# --------------------------------------------------------------------------
# 逐跳覆盖：评测集说它需要哪些条款
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Endpoint:
    """评测样本声称答案要用到的一处证据：单条条款，或整篇文档。"""

    label: str
    clause_ids: tuple[str, ...]


def parse_reference_endpoints(
    text: str,
    *,
    resolve_clause: Callable[[str, int], str | None],
    resolve_doc: Callable[[str], Sequence[str]],
) -> tuple[list[Endpoint], list[str]]:
    """从样本的 ``reference`` 里机械地读出「这题需要哪些条款」。

    用 reference 而不是手写一张跳表：reference 是**仓库里已有的、人工撰写的
    金标准**，从它派生出来的跳列表不可能与题目脱节，而手写表会。读法直接复用
    结构层的 ``find_citations``——同一个读者，就不可能对「《中华人民共和国
    企业所得税法》第二十七条」指的是哪一条产生两种解释。

    书名号里出现、但不带条号的名字（如《适用9增值税税率货物范围注释》）按
    **整篇文档**计：题目要的是那份附件里的条款，不是它的某一条。

    解析不出的一律进 ``unresolved`` 原样返回，绝不静默丢弃——「这题需要的跳
    我们没算」和「这题的跳不成立」必须能区分开。
    """
    endpoints: list[Endpoint] = []
    unresolved: list[str] = []
    seen_unresolved: set[str] = set()
    seen_clauses: set[str] = set()
    covered_titles: set[str] = set()

    def note(raw: str) -> None:
        # 去重但保序：同一处未解析在多个条款里重复出现时，报一次就够。
        if raw not in seen_unresolved:
            seen_unresolved.add(raw)
            unresolved.append(raw)

    for cite in find_citations(text or ""):
        if cite.is_self:
            # reference 是评测集作者的综述文字，「本法/本条例」在其中没有宿主。
            note(cite.raw)
            continue
        clause_id = resolve_clause(cite.law_token, cite.article)
        if clause_id is None:
            note(cite.raw)
            continue
        covered_titles.add(cite.law_token.strip("《》").replace(" ", ""))
        if clause_id in seen_clauses:
            continue
        seen_clauses.add(clause_id)
        endpoints.append(
            Endpoint(label=f"{cite.law_token}第{cite.article}条", clause_ids=(clause_id,))
        )

    for name in find_title_references(text or ""):
        if name in covered_titles:
            continue
        ids = tuple(resolve_doc(name))
        if not ids:
            note(f"《{name}》")
            continue
        endpoints.append(Endpoint(label=f"《{name}》(整篇 {len(ids)} 条)", clause_ids=ids))

    return endpoints, unresolved


def hop_coverage(
    endpoints: Sequence[Endpoint],
    *,
    cites_links: Callable[[Sequence[str], Sequence[str]], int],
    mentions_shared: Callable[[Sequence[str], Sequence[str]], int],
    min_shared: int,
) -> dict:
    """端点两两之间：结构层连得上吗？实体层连得上吗？

    两个机制分开统计，因为它们的修法完全不同：``cites`` 缺是结构层的问题（正则
    没读出这条引用），``mentions`` 缺是实体层的问题（条款没被抽到共享实体）。
    合成一个「图覆盖率」就再也分不清该修哪边。

    ``mentions`` 用 ``共享实体数 >= min_shared`` 判定，与扩散腿的 Cypher 同口径：
    覆盖率必须按**生产路径的阈值**算，否则会得到「桥存在但从不过滤器」的假阳性。
    """
    pairs = []
    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            left, right = endpoints[i], endpoints[j]
            cited = cites_links(left.clause_ids, right.clause_ids)
            shared = mentions_shared(left.clause_ids, right.clause_ids)
            pairs.append(
                {
                    "left": left.label,
                    "right": right.label,
                    "cites": cited,
                    "mentions_shared": shared,
                    "linked_by_cites": cited > 0,
                    "linked_by_mentions": shared >= min_shared,
                }
            )
    linked = sum(1 for p in pairs if p["linked_by_cites"] or p["linked_by_mentions"])
    return {
        "endpoints": len(endpoints),
        "pairs": pairs,
        "pairs_total": len(pairs),
        "pairs_by_cites": sum(1 for p in pairs if p["linked_by_cites"]),
        "pairs_by_mentions": sum(1 for p in pairs if p["linked_by_mentions"]),
        "pairs_linked": linked,
        "pairs_unlinked": len(pairs) - linked,
        "complete": bool(pairs) and linked == len(pairs),
    }


# --------------------------------------------------------------------------
# 精度形状：实体的度分布
# --------------------------------------------------------------------------
HUB_THRESHOLDS: tuple[int, ...] = (10, 50, 100, 500)


def _percentile(values: Sequence[int], q: float) -> int | None:
    """最近秩分位数；空输入返回 ``None``。"""
    if not values:
        return None
    index = min(len(values) - 1, int(round(q * (len(values) - 1))))
    return values[index]


def degree_summary(
    degrees: Iterable[int], *, thresholds: Sequence[int] = HUB_THRESHOLDS
) -> dict:
    """实体度的分位数与 hub / singleton 计数。

    两端都是垃圾，但坏法不同：hub 是**事实正确却有害**（两跳邻域退化成全库、
    权重还压过 CITES），singleton 多半是长尾噪声或一次性专名。所以这两个计数
    都要能单独断言，而不是只报一个「平均度」把两端抹平。

    hub 阈值不在这里拍板：可辩护的做法是从逐跳覆盖反推——扫阈值，看还剩几条桥、
    目标条款是否仍在预算内可达（见 ``scripts/report_entities.py``）。
    """
    values = sorted(int(d) for d in degrees)
    total = len(values)
    return {
        "entities": total,
        "singleton": sum(1 for v in values if v == 1),
        "singleton_ratio": ratio(sum(1 for v in values if v == 1), total),
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p99": _percentile(values, 0.99),
        "max": values[-1] if values else None,
        "mean": (sum(values) / total) if total else None,
        "hubs": [
            {
                "threshold": int(k),
                "entities": sum(1 for v in values if v >= k),
                "ratio": ratio(sum(1 for v in values if v >= k), total),
            }
            for k in thresholds
        ],
    }


def target_ranks(neighbors: Sequence[str], targets: Iterable[str]) -> dict:
    """目标条款在扩散腿返回列表里的名次。

    ``first_rank`` 为 ``None`` 有两种含义——「腿没跑」与「跑了但没召回」——
    调用方必须自己区分并把腿的状态一起打印，否则一次未接入的腿会被读成
    一次干净的失败。
    """
    wanted = set(targets)
    order = list(neighbors)
    first = next((i for i, cid in enumerate(order, start=1) if cid in wanted), None)
    return {
        "total_neighbors": len(order),
        "hits": sum(1 for cid in order if cid in wanted),
        "first_rank": first,
    }
