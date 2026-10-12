"""实体层的零 LLM 质量报告：覆盖率 + 精度形状。

为什么要有这个脚本：2026-09-30 的两臂各 3 遍评测给出一个硬结论——GLM-5.3 在
``temperature=0`` 下对逐字相同输入逐题可差 0.4~0.58，比实体层的任何改动都大。
于是「实体层改好了没有」不能用 RAGAS 回答，只能用**离散的、零 LLM 的**判据回答
（见 ``docs/2026-09-30_1407_*.md`` §10）。

报告分六节，各节的输入与代价都不同：

| 节 | 量 | 输入 | 是否联网 |
| --- | --- | --- | --- |
| 账本 P3 | 模型丢了多少、按原因分解 | ``entities.jsonl`` | 否 |
| 锚点覆盖 L1 | 有锚点的条款占比（四层分母） | ``entities.jsonl`` + chunk 快照 | 否 |
| 字面性审计 P1a | 每条边在条款正文里有没有字面依据（全量，非抽样） | 同上 | 否 |
| 度分布 P2 | 实体度分位数 / hub / singleton | Neo4j | 是 |
| 逐跳覆盖 L2 | 评测集需要的每一跳，结构层与实体层各自连不连得上 | Neo4j + 评测集 | 是 |
| 预算内可达 L3 | 端点作种子时，目标条款在扩散腿返回列表里排第几 | Neo4j + chunk 快照 | 是 |

P1a 与 L1 都完全离线且**全量**，所以它们给出的是精确计数而不是估计值——P1（边
正确性）里需要抽样、需要置信区间的那一半只剩"类型归得对不对"，那必须问模型，
见 ``audit.MENTION_MATCH_KEYS`` 的口径说明。

L3 的口径必须说清楚：种子取自 ``reference`` 里的**金标准条款**，而生产环境的种子
来自词法腿。所以它是**乐观上界**——「定义条款只要进了种子，图能不能在预算内把
其余证据捞回来」。上界都做不到，生产更做不到；上界做到了也不代表生产能做到。

L3 对每一对端点跑两遍：一遍走生产路径（每种子 ``LIMIT`` + 总预算），一遍拆掉每
种子 ``LIMIT``。两次都没进邻域，才是**实体层真的没连上**；进了邻域却没进预算，
问题在排序与截断——这两种失败的修法完全不同，只报一个「没召回」等于把可修的
和不可修的混成一堆。

从仓库根目录运行::

    .venv\\Scripts\\python scripts\\report_entities.py
    .venv\\Scripts\\python scripts\\report_entities.py --skip-graph   # Neo4j 不可达时只跑前两节
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_SNAPSHOT = "data/knowledge/processed/chunks.jsonl"
DEFAULT_ENTITIES = "data/knowledge/graph/entities.jsonl"
DEFAULT_QUESTIONS = "data/knowledge/eval/questions_multihop.jsonl"
DEFAULT_REPORT = "data/knowledge/eval/report_entities.json"

# 结构层已覆盖的跳：方向无关，长度 1~2（第四条引第三条、第二十七条引第三条 →
# 第四条 与 第二十七条 由 CITES 连成两跳）。长度不放大到 3：跳数一多，任何两个
# 条款都能被"连通"，这条指标就失去分辨力了。
_CITES_HOPS_QUERY = """
MATCH (a:Clause)-[:CITES*1..2]-(b:Clause)
WHERE a.clause_id IN $left AND b.clause_id IN $right
RETURN count(*) AS c
"""

# 实体层能提供的跳。取**跨端点对的最大共享实体数**，与扩散腿 min_shared_entities
# 同口径：一端是整篇文档时有几十条条款，逐对列举既慢又没意义，而"最好的那一对
# 共享几个实体"正是过滤器要问的问题。
_SHARED_ENTITIES_QUERY = """
MATCH (a:Clause)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(b:Clause)
WHERE a.clause_id IN $left AND b.clause_id IN $right AND a.clause_id <> b.clause_id
WITH a, b, count(DISTINCT e) AS shared
RETURN max(shared) AS c
"""

# L3 的「不限预算」版扩散腿：与 ``expand._MENTIONS_QUERY`` 逐条同构，**只去掉
# 每种子 LIMIT**。存在的理由是必须把两种失败分开——「目标排在预算之外」与
# 「目标根本不是邻居」的修法完全不同（调 per_seed / 治 hub，vs 去补实体层）。
# 排序不在这里复制：行喂给生产用的 ``rank_neighbors``，否则报告与检索会对同一批
# 邻居给出两种名次。
_MENTIONS_UNCAPPED_QUERY = """
UNWIND $seeds AS sid
MATCH (c:Clause {clause_id: sid})-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(n:Clause)
WHERE NOT n.clause_id IN $seeds
WITH sid AS seed, n.clause_id AS clause_id, count(DISTINCT e) AS weight
WHERE weight >= $min_shared
RETURN seed, clause_id, weight
"""

_DEGREES_QUERY = """
MATCH (e:Entity)<-[:MENTIONS]-()
WITH e, count(*) AS d
RETURN d
"""

_TOP_HUBS_QUERY = """
MATCH (e:Entity)<-[:MENTIONS]-()
WITH e, count(*) AS d
RETURN e.type AS type, e.name AS name, d AS degree
ORDER BY d DESC, name
LIMIT $limit
"""

# ``min_shared_entities`` 的扫参点。只看生产值 1 无法回答"这个值该定几"：邻域规模
# 与「真桥还剩几条」都随它单调变化，只有把整条曲线画出来才知道两者谁掉得快。
MIN_SHARED_SWEEP: tuple[int, ...] = (1, 2, 3, 4)


def _inject_user_env(name: str) -> None:
    """把 User 作用域的环境变量提升到进程内，供配置里的 ``${VAR}`` 解析。"""
    if os.environ.get(name):
        return
    try:
        import subprocess

        probe = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f"[Environment]::GetEnvironmentVariable('{name}','User')"],
            capture_output=True, text=True, timeout=15,
        )
        if probe.returncode == 0 and probe.stdout.strip():
            os.environ[name] = probe.stdout.strip()
    except Exception:
        pass


def _stream_jsonl(path: Path):
    """逐行 yield，不整份读进内存（chunk 快照 4 万行以上）。"""
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.1%}"


def _print_ledger(row: dict) -> None:
    print("\n--- 账本 P3：模型被问了什么、丢掉了什么 ---")
    print(f"文档         : {row['docs_total']}（成功 {row['docs_ok']} / 失败 {row['docs_error']}）")
    stats = row["stats"]
    print(f"条款编号回执 : {stats['refs_returned']}/{stats['refs_sent']}"
          f"（缺 {stats['refs_missing']}，回执缺失率 {_fmt(row['ref_loss'])}）")
    print(f"原始实体     : {stats['raw_entities']} → 采纳 {stats['mentions']} 条 mention"
          f"（丢弃 {row['discarded']}，{_fmt(row['discard_ratio'])}）")
    print(f"实体内存     : 各文档内去重后合计 {stats['entities']}"
          "（同一实体在多篇文档各计一次，**不等于**图中的 Entity 节点数）")
    for key, value in row["discards"].items():
        print(f"    · {key:<14s}: {value}")
    if stats["refs_unknown"]:
        print(f"    · refs_unknown: {stats['refs_unknown']}（模型回了不存在的条款编号）")


def _print_histogram(row: dict) -> None:
    print("\n--- 锚点数分布（长枚举 vs 普通）---")
    for name, group in row["groups"].items():
        total = group["合计"]
        # 桶的顺序就是 ``audit`` 里的插入顺序（BUCKETS），合计项排除在外。
        cells = " | ".join(
            f"{bucket}: {group[bucket]}" + (f"({group[bucket] / total:.1%})" if total else "")
            for bucket in group
            if bucket not in ("合计", "有锚点")
        )
        print(f"{name}  条款 {total}  | {cells}")
    print("  `6+` 恒为 0 即 MAX_ENTITIES_PER_CLAUSE 上限的指纹"
          "（长枚举条款 72.0% 恰好停在 3-5）")


def _print_anchor(row: dict) -> None:
    print("\n--- 锚点覆盖 L1（逐层收窄分母）---")
    for layer in row["layers"]:
        print(f"{layer['layer']}：条款 {layer['clauses']}，有锚点 {layer['anchored']}，"
              f"覆盖率 {_fmt(layer['coverage'])}")
    print(f"  锚点总数 {row['anchored_total']}；其中指向不存在的条款 {row['anchored_orphan']}"
          "（>0 说明两份快照不同代，覆盖率不可信）")
    dropped = row["procedural_dropped"]
    third = row["layers"][-2]["coverage"]
    verdict = (
        "**高于**上一层，即这条规则在删有实质内容的条款，最后一层不可用"
        if dropped["coverage"] is not None
        and third is not None
        and dropped["coverage"] > third
        else "不高于上一层，最后一层可以读"
    )
    print(f"  程序性规则筛掉 {dropped['clauses']} 条，其中 {dropped['anchored']} 条有锚点"
          f"（{_fmt(dropped['coverage'])}）：{verdict}；"
          "可辩护的分母是「再排除正文 < 40 字」那一层")


def _print_literal(row: dict) -> None:
    print("\n--- 字面性审计 P1a（零 LLM，全量：边在条款正文里有没有字面依据）---")
    print(f"边总数       : {row['mentions']}（正文缺失 {row['unresolved_clauses']} 条边）")
    if row["unresolved_clauses"]:
        print("  [警告] 有边指向 chunk 快照里不存在的条款：两份快照不同代，"
              "下面的比例不可信（先重跑 build_graph.py / build_entities.py）")
    matched = row["matched"]
    exact_ratio = matched["exact"] / row["resolved"] if row["resolved"] else None
    print(f"字面依据     : exact {matched['exact']}（{_fmt(exact_ratio)}）"
          f" / partial {matched['partial']}（{_fmt(row['partial_ratio'])}）"
          f" / absent {matched['absent']}（{_fmt(row['absent_ratio'])}）")
    print("  释义：exact = 归一化名字是正文子串；partial = 名字的 3 字片段出现在正文里；"
          "absent = 都不出现。")
    print("  **absent 是「无据边」的上界，不是结论**：prompt 规则 3 要求写规范全称"
          "（写「个人所得税」不写「个税」），正文用简称时正确的边也会落进 absent。")
    print("  把上界拆成「模型幻觉」与「全称改写」要靠模型，那是 P1b 的双盲判官。")
    print("  按类型:")
    for key, group in row["by_type"].items():
        print(f"    {key:<10s} 边 {group['mentions']:>6} | exact {group['exact']:>6} | "
              f"partial {group['partial']:>5} | absent {group['absent']:>5}"
              f"（{_fmt(group['absent_ratio'])}）")
    print("  按实体度（度数由快照算出，与图中 MENTIONS 同口径）:")
    for label, group in row["by_degree"].items():
        print(f"    度 {label:<6s} 边 {group['mentions']:>6} | exact {group['exact']:>6} | "
              f"partial {group['partial']:>5} | absent {group['absent']:>5}"
              f"（{_fmt(group['absent_ratio'])}）")
    for kind in ("absent", "partial"):
        items = row["samples"].get(kind) or []
        if not items:
            continue
        # 抽样必须带上 clause_id：这一节的全部价值就是「报告里的数字可以被逐条核对」，
        # 没有条款定位的抽样只是一串名字，核不了。
        print(f"  {kind} 抽样（可据此回查正文）:")
        for item in items:
            print(f"    {item['type']:<10s} {item['name']}（度 {item['degree']}）"
                  f" @ {item['clause_id']}")


def _print_degree(summary: dict, top: list[dict]) -> None:
    print("\n--- 度分布 P2 ---")
    mean = "-" if summary["mean"] is None else f"{summary['mean']:.2f}"
    print(f"实体 {summary['entities']}；度 p50={summary['p50']} p90={summary['p90']} "
          f"p99={summary['p99']} max={summary['max']} mean={mean}")
    print(f"仅出现一次 : {summary['singleton']}（{_fmt(summary['singleton_ratio'])}）")
    hubs = "，".join(f"度≥{h['threshold']}: {h['entities']}（{_fmt(h['ratio'])}）"
                     for h in summary["hubs"])
    print(f"hub 计数   : {hubs}")
    print("  度最高的实体（两跳邻域被它们决定）:")
    for item in top:
        print(f"    {item['degree']:>6}  [{item['type']}] {item['name']}")


def _build_resolvers(extraction):
    """标题/条号 → clause_id 的解析器，全部复用结构层已有的索引。

    ``TitleIndex.resolve`` 做的是**最长后缀**匹配，正是结构层读引用时用的那套，
    所以「企税法」与「中华人民共和国企业所得税法」在这里解析到同一篇文档——
    两条路径对同一个名字不可能有两种解释。
    """
    from app.knowledge.graph.extract import TitleIndex

    title_index = TitleIndex(
        {title: doc_id for doc_id, title in extraction.doc_titles.items() if title}
    )
    by_article: dict[tuple[str, int], str] = {}
    by_doc: dict[str, list[str]] = {}
    for clause in extraction.clauses:
        by_doc.setdefault(clause.doc_id, []).append(clause.clause_id)
        if clause.article is not None:
            by_article.setdefault((clause.doc_id, clause.article), clause.clause_id)

    def resolve_clause(law_token: str, article: int) -> str | None:
        doc_id = title_index.resolve(law_token)
        if doc_id is None:
            return None
        return by_article.get((doc_id, article))

    def resolve_doc(name: str) -> list[str]:
        doc_id = title_index.resolve(name)
        return by_doc.get(doc_id, []) if doc_id else []

    return resolve_clause, resolve_doc, title_index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--snapshot", default=DEFAULT_SNAPSHOT, help="chunk 快照")
    parser.add_argument("--entities", default=DEFAULT_ENTITIES,
                        help="实体抽取快照（build_entities.py 的输出）")
    parser.add_argument("--questions", default=DEFAULT_QUESTIONS, help="评测问题集")
    parser.add_argument("--report", default=DEFAULT_REPORT, help="报告输出路径（JSON，可 diff）")
    parser.add_argument("--skip-graph", action="store_true",
                        help="跳过需要 Neo4j 的三节（账本与锚点覆盖仍照常计算）")
    args = parser.parse_args(argv)

    _inject_user_env("NEO4J_PASSWORD")

    from app.knowledge.graph.audit import (
        anchor_coverage,
        clause_mention_counts,
        degree_summary,
        hop_coverage,
        last_wins,
        mention_literal_audit,
        mentions_per_clause,
        parse_reference_endpoints,
        summarise_snapshot,
        target_ranks,
    )
    from app.knowledge.graph.expand import (
        GRAPH_WEIGHT_KEY,
        KIND_MENTIONS,
        ClauseTextIndex,
        GraphExpander,
        clause_key,
        rank_neighbors,
    )
    from app.knowledge.graph.extract import extract_structure

    snapshot_path = ROOT / args.snapshot
    entities_path = ROOT / args.entities
    if not snapshot_path.exists():
        print(f"chunk 快照不存在：{snapshot_path}；请先运行 scripts/ingest_knowledge.py。")
        return 1
    if not entities_path.exists():
        print(f"实体快照不存在：{entities_path}；请先运行 scripts/build_entities.py。")
        return 1

    print(f"chunk 快照   : {args.snapshot}")
    print(f"实体快照     : {args.entities}")

    records = last_wins(_stream_jsonl(entities_path))
    ledger = summarise_snapshot(records)
    _print_ledger(ledger)

    extraction = extract_structure(_stream_jsonl(snapshot_path))
    clause_texts = {c.clause_id: c.text for c in extraction.clauses}
    counts = clause_mention_counts(records)
    histogram = mentions_per_clause(clause_texts, counts)
    _print_histogram(histogram)

    error_docs = {doc_id for doc_id, rec in records.items() if rec.get("status") != "ok"}
    anchor = anchor_coverage(extraction.clauses, counts, error_docs=error_docs)
    _print_anchor(anchor)

    literal = mention_literal_audit(records, clause_texts)
    _print_literal(literal)

    report: dict = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "snapshot": args.snapshot,
            "entities": args.entities,
            "questions": args.questions,
        },
        "ledger": ledger,
        "mentions_per_clause": histogram,
        "anchor_coverage": anchor,
        "mention_literal": literal,
    }

    if args.skip_graph:
        print("\n--skip-graph : 未连接 Neo4j，度分布 / 逐跳覆盖 / 预算内可达三节未计算")
    else:
        from app.core import load_settings
        from app.knowledge.graph.neo4j import Neo4jGraphStore
        from app.knowledge.ingest.pipeline import load_chunk_snapshot

        settings = load_settings(ROOT / "configs" / "application.toml")
        graph = Neo4jGraphStore(settings.knowledge.graph)
        try:
            degrees = [int(row["d"]) for row in graph.query_all(_DEGREES_QUERY)]
            summary = degree_summary(degrees)
            top = graph.query_all(_TOP_HUBS_QUERY, {"limit": 15})
            _print_degree(summary, top)
            report["degree"] = {"summary": summary, "top": top}

            resolve_clause, resolve_doc, _ = _build_resolvers(extraction)
            min_shared = settings.knowledge.graph.expand.min_shared_entities
            expand_settings = settings.knowledge.graph.expand.model_copy(
                update={"enabled": True, "use_cites": False, "use_mentions": True}
            )
            chunks = load_chunk_snapshot(str(snapshot_path))
            expander = GraphExpander(
                expand_settings, graph.query_all, ClauseTextIndex(chunks)
            )

            def cites_links(left, right) -> int:
                row = graph.query_one(
                    _CITES_HOPS_QUERY, {"left": list(left), "right": list(right)}
                )
                return int(row["c"] or 0) if row else 0

            def mentions_shared(left, right) -> int:
                # 聚合函数在没有匹配行时**仍然返回一行**，值为 null——所以这里
                # 必须同时防 None，只判 `if row` 会漏掉"查询成功但一条都没匹配"。
                row = graph.query_one(
                    _SHARED_ENTITIES_QUERY, {"left": list(left), "right": list(right)}
                )
                return int(row["c"] or 0) if row else 0

            samples = list(_stream_jsonl(ROOT / args.questions))
            hops: dict[str, dict] = {}
            reach: dict[str, dict] = {}
            unresolved: dict[str, list[str]] = {}
            print("\n--- 逐跳覆盖 L2 / 预算内可达 L3 ---")
            print(f"（min_shared_entities={min_shared}，"
                  f"per_seed={expand_settings.neighbors_per_seed}，"
                  f"max_results={expand_settings.max_results}，"
                  f"seed_top_n={expand_settings.seed_top_n}）")
            for sample in samples:
                sample_id = sample.get("id") or sample.get("question", "")[:24]
                endpoints, missing = parse_reference_endpoints(
                    sample.get("reference", ""),
                    resolve_clause=resolve_clause,
                    resolve_doc=resolve_doc,
                )
                if missing:
                    unresolved[sample_id] = missing
                row = hop_coverage(
                    endpoints,
                    cites_links=cites_links,
                    mentions_shared=mentions_shared,
                    min_shared=min_shared,
                )
                hops[sample_id] = row

                # L3：以金标准端点为种子（乐观上界），看目标端点能否在预算内被捞回。
                # 同一对种子/目标跑两遍：一遍走生产路径（每种子 LIMIT + 总预算），
                # 一遍拆掉每种子 LIMIT。两次的名次差就是"被排序截断"的部分，两次
                # 都没进邻域才是"实体层根本没连上"——修法完全不同（调 per_seed /
                # 治 hub，vs 去补实体层），所以不能只报一个"没召回"。
                #
                # 邻域只由**种子**决定、与目标无关，所以按种子聚合一次；再对
                # min_shared_entities 扫参——它是把「主题簇」收成「核心」的唯一旋钮，
                # 扫参也是唯一能给出可辩护取值的做法（见 audit.degree_summary）。
                capped_ids: dict[str, list[str]] = {}
                top_by_seed: dict[str, list[dict]] = {}
                sweep_ids: dict[str, dict[str, list[str]]] = {}
                for left in endpoints:
                    if left.label in capped_ids:
                        continue
                    seeds = list(left.clause_ids)[: expand_settings.seed_top_n]
                    hits = expander.expand(seeds)
                    capped_ids[left.label] = [clause_key(hit) for hit in hits]
                    # 把预算内实际胜出的邻居记下来：只报"目标排第 129"没法判断该
                    # 治谁，看到前 8 条是同法兄弟姐妹还是长尾才决定改权重还是改阈值。
                    top_by_seed[left.label] = [
                        {
                            "rank": rank,
                            "title": hit.title,
                            "section": str(hit.metadata_payload.get("section") or ""),
                            "weight": hit.metadata_payload.get(GRAPH_WEIGHT_KEY),
                        }
                        for rank, hit in enumerate(hits[:8], start=1)
                    ]
                    sweep_ids[left.label] = {}
                    for threshold in MIN_SHARED_SWEEP:
                        rows = graph.query_all(
                            _MENTIONS_UNCAPPED_QUERY,
                            {"seeds": seeds, "min_shared": threshold},
                        )
                        sweep_ids[left.label][str(threshold)] = [
                            neighbor.clause_id
                            for neighbor in rank_neighbors(
                                rows, kind=KIND_MENTIONS, max_results=len(rows)
                            )
                        ]

                ranked: list[dict] = []
                for i, left in enumerate(endpoints):
                    for j, right in enumerate(endpoints):
                        if i == j:
                            continue
                        capped = target_ranks(capped_ids[left.label], right.clause_ids)
                        uncapped = target_ranks(
                            sweep_ids[left.label][str(min_shared)], right.clause_ids
                        )
                        ranked.append(
                            {
                                "seed": left.label,
                                "target": right.label,
                                "capped_rank": capped["first_rank"],
                                "uncapped_rank": uncapped["first_rank"],
                                "neighbors_capped": capped["total_neighbors"],
                                "neighbors_uncapped": uncapped["total_neighbors"],
                                "rank_by_min_shared": {
                                    threshold: target_ranks(
                                        ids, right.clause_ids
                                    )["first_rank"]
                                    for threshold, ids in sweep_ids[left.label].items()
                                },
                            }
                        )
                reach[sample_id] = {
                    "pairs": ranked,
                    "top_by_seed": top_by_seed,
                    "neighbourhood_size_by_min_shared": {
                        label: {key: len(ids) for key, ids in by_threshold.items()}
                        for label, by_threshold in sweep_ids.items()
                    },
                }

                flag = "PASS" if row["complete"] else "FAIL"
                print(f"[{flag}] {sample_id}: 端点 {row['endpoints']}，"
                      f"跳 {row['pairs_linked']}/{row['pairs_total']}"
                      f"（cites {row['pairs_by_cites']} / mentions {row['pairs_by_mentions']}）")
                for pair in row["pairs"]:
                    if pair["linked_by_cites"] or pair["linked_by_mentions"]:
                        continue
                    print(f"       未连通: {pair['left']} <-?-> {pair['right']}"
                          f"（共享实体 {pair['mentions_shared']}）")
                if missing:
                    print(f"       未解析: {'; '.join(missing)}")
                reached = sum(1 for item in ranked if item["capped_rank"] is not None)
                available = sum(1 for item in ranked if item["uncapped_rank"] is not None)
                truncated = available - reached
                unwired = len(ranked) - available
                sizes = [item["neighbors_capped"] for item in ranked]
                span = f"{min(sizes)}~{max(sizes)}" if sizes else "-"
                print(f"       L3 预算内可达: {reached}/{len(ranked)} 对"
                      f"（种子取金标准端点，乐观上界；腿每次返回邻居 {span} 条）")
                print(f"         邻域内存在 {available} 对 → 被排序截断 {truncated} 对"
                      f"（调 per_seed / max_results 可救），"
                      f"根本不在邻域 {unwired} 对"
                      f"（共享实体数 < {min_shared}，是实体层的洞）")
                for label, items in top_by_seed.items():
                    shown = "，".join(
                        f"{item['rank']}.{item['title']}·{item['section']}(w{item['weight']})"
                        for item in items
                    )
                    print(f"         种子「{label}」预算内的邻居: {shown}")
                cells = []
                for threshold in MIN_SHARED_SWEEP:
                    key = str(threshold)
                    sizes = sorted(
                        len(by_threshold[key])
                        for by_threshold in sweep_ids.values()
                    )
                    span = f"{sizes[0]}~{sizes[-1]}" if sizes else "-"
                    present = sum(
                        1 for item in ranked
                        if item["rank_by_min_shared"].get(key) is not None
                    )
                    cells.append(f"t={threshold}: 邻域 {span}，目标在内 {present}/{len(ranked)}")
                print("         min_shared 扫参: " + " | ".join(cells))
                print("           （调高阈值只缩邻域、不找目标：若「目标在内」掉得比邻域规模"
                      "快，真桥本身就是低权重的，卡权重会连桥一起砍掉）")

            report["hops"] = {
                "min_shared_entities": min_shared,
                "samples": hops,
                "unresolved": unresolved,
            }
            report["reachability"] = {
                "seed_top_n": expand_settings.seed_top_n,
                "max_results": expand_settings.max_results,
                "neighbors_per_seed": expand_settings.neighbors_per_seed,
                "min_shared_sweep": list(MIN_SHARED_SWEEP),
                "samples": reach,
                "totals": {
                    "pairs": sum(len(s["pairs"]) for s in reach.values()),
                    "capped": sum(
                        1 for s in reach.values() for p in s["pairs"]
                        if p["capped_rank"] is not None
                    ),
                    "uncapped": sum(
                        1 for s in reach.values() for p in s["pairs"]
                        if p["uncapped_rank"] is not None
                    ),
                },
            }
            report["graph_counts"] = {
                "nodes": graph.counts(),
                "relations": graph.relation_counts(),
            }
        except Exception as exc:
            print(f"\nneo4j : 不可用 -> {type(exc).__name__}: {exc}")
            print("（前两节已完成；加 --skip-graph 可跳过联网的三节）")
            graph.close()
            return 1
        graph.close()

    path = ROOT / args.report
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\n报告已写入   : {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
