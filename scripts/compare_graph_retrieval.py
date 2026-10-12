"""图扩散腿的四臂对比：基线 / +CITES / +MENTIONS / 全开。

只测**检索侧机制指标**，不调用任何 LLM：对 `questions_multihop.jsonl` 的每一跳，
判定它是否出现在检索器返回的 contexts 里（判定口径与
[2026-09-28 多跳样本集](../docs/2026-09-28_1746_多跳评测样本集_update.md) §4 一致：
按「文档名 + 条款号」而不是 chunk_id，因为 `expand_to_parent` 会保留子块 id）。
RAGAS 端到端分数需要评判模型，待其可用后另行补跑。

四臂只差一个变量——图扩散腿的开关，其余检索配置（同一语料、同一 Milvus、
同一 BM25 快照、同一交叉编码器）完全一致：

| 臂 | enabled | use_cites | use_mentions |
| --- | --- | --- | --- |
| baseline | false | - | - |
| cites | true | true | false |
| mentions | true | false | true |
| both | true | true | true |

分开跑 cites / mentions 是为了把效果归因到**结构层**或**实体层**，
而不是笼统的"图"——两层的数据来源、成本与失败模式完全不同。

运行（需 Milvus 与 Neo4j 可达）：

    .venv\\Scripts\\python scripts\\compare_graph_retrieval.py
    .venv\\Scripts\\python scripts\\compare_graph_retrieval.py --report out.json

判定用的目标（文档名, 条款号）**不做模糊匹配**：`中华人民共和国企业所得税法`
是 `中华人民共和国企业所得税法实施条例` 的子串，用"包含"判定会把实施条例的
第三条算成企税法的第三条。故此处一律用标题**全等**；`None` 表示该文档整篇
只有一节（`section == "全文"`，PDF 与国务院通知均如此），只按文档判定。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 逐跳目标：(文档标题, 条款号 or None=整篇)。取自 2026-09-28 多跳样本集 §3 的
# "需要的跳"一栏，不是从检索结果反推的——否则指标会自我实现。
_TARGETS: dict[str, list[tuple[str, str | None]]] = {
    "mh-cit-001": [
        ("中华人民共和国企业所得税法", "第三条"),
        ("中华人民共和国企业所得税法", "第四条"),
        ("中华人民共和国企业所得税法实施条例", "第九十一条"),
    ],
    "mh-vat-001": [
        ("中华人民共和国增值税法", "第十条"),
        ("适用9增值税税率货物范围注释", None),
    ],
    "mh-iit-001": [
        ("中华人民共和国个人所得税法", "第六条"),
        ("国务院关于提高个人所得税有关专项附加扣除标准的通知", None),
    ],
    "mh-iit-002": [
        ("中华人民共和国个人所得税法", "第六条"),
        ("国务院关于设立3岁以下婴幼儿照护个人所得税专项附加扣除的通知", None),
        ("国务院关于提高个人所得税有关专项附加扣除标准的通知", None),
    ],
    "mh-cit-002": [
        ("中华人民共和国企业所得税法", "第二十七条"),
        ("中华人民共和国企业所得税法实施条例", "第九十一条"),
    ],
    "mh-env-001": [
        ("中华人民共和国环境保护税法", "第七条"),
        ("中华人民共和国环境保护税法实施条例", "第七条"),
    ],
}

# 判别样本 / 对照样本（§3.1 / §3.2）。对照样本是回归哨兵：任一臂让它掉分，
# 该臂就是净噪声，而不是"多跳能力"。
_DISCRIMINATING = {"mh-cit-001", "mh-vat-001", "mh-iit-001", "mh-iit-002"}

_ARMS: list[tuple[str, bool, bool, bool]] = [
    # (name, enabled, use_cites, use_mentions)
    ("baseline", False, False, False),
    ("cites", True, True, False),
    ("mentions", True, False, True),
    ("both", True, True, True),
]


def _inject_user_env(name: str) -> None:
    """Promote a User-scope env var so ``${VAR}`` refs in the config resolve."""
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


def _resolve_model_ref(model_ref: str) -> str:
    """Resolve a configured model ref, keeping transformers off the hub."""
    candidate = Path(model_ref)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    if candidate.is_dir():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        return str(candidate)
    return model_ref


def _section(hit) -> str:
    return str(hit.metadata_payload.get("section", "") or "")


def _hop_hit(hits, doc_title: str, section: str | None) -> object | None:
    """The first returned hit that proves this hop, or None."""
    for hit in hits:
        if hit.title != doc_title:
            continue
        if section is None or _section(hit) == section:
            return hit
    return None


def _load_questions(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


async def _run_question(retriever, question: str) -> list:
    return await retriever.retrieve(question, top_k=retriever.top_k)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--questions", default="data/knowledge/eval/questions_multihop.jsonl",
                        help="评测问题集（JSONL，每行一个问题）")
    parser.add_argument("--report", default="data/knowledge/eval/report_graph_arms.json",
                        help="对比结果输出路径（JSON）")
    args = parser.parse_args(argv)

    for var in ("MILVUS_PASSWORD", "NEO4J_PASSWORD"):
        _inject_user_env(var)

    from app.core import load_settings
    from app.knowledge.embeddings import LocalEmbeddingModel
    from app.knowledge.graph.expand import ClauseTextIndex, GraphExpander
    from app.knowledge.graph.neo4j import Neo4jGraphStore
    from app.knowledge.hybrid import HybridRetriever
    from app.knowledge.ingest.pipeline import load_chunk_snapshot
    from app.knowledge.rerank import CrossEncoderReranker
    from app.knowledge.vectorstore.milvus import MilvusStore

    settings = load_settings(ROOT / "configs" / "application.toml")
    kn = settings.knowledge
    if not kn.graph.enabled:
        print("[knowledge.graph].enabled = false，图不可用；本对比无意义。")
        return 1

    snapshot_path = ROOT / "data/knowledge/processed/chunks.jsonl"
    if not snapshot_path.exists():
        raise FileNotFoundError(
            f"chunk 快照不存在：{snapshot_path}；请先运行 scripts/ingest_knowledge.py"
        )
    chunks = load_chunk_snapshot(str(snapshot_path))

    store = MilvusStore(kn.milvus)
    # 与 evaluate_knowledge.py 同一理由：HybridRetriever 会吞掉半死的向量腿，
    # 那样测出来的就不是"混合检索 + 图"了。这里先校验，宁可中断。
    store.verify_schema()
    embed = LocalEmbeddingModel(_resolve_model_ref(kn.embedding_model))
    reranker = CrossEncoderReranker(_resolve_model_ref(kn.reranker_model))

    graph_store = Neo4jGraphStore(kn.graph)
    index = ClauseTextIndex(chunks)
    print(f"语料快照     : {len(chunks)} chunks，其中 {len(index)} 个条款有正文（可回填）")

    questions = _load_questions(ROOT / args.questions)
    unknown = [q["id"] for q in questions if q.get("id") not in _TARGETS]
    if unknown:
        print(f"以下问题没有逐跳目标，无法判定：{unknown}")
        return 1

    report: dict = {
        "questions": args.questions,
        "arms": {},
    }

    for name, enabled, use_cites, use_mentions in _ARMS:
        expand_settings = kn.graph.expand.model_copy(
            update={
                "enabled": enabled,
                "use_cites": use_cites,
                "use_mentions": use_mentions,
            }
        )
        expander = None
        if enabled:
            expander = GraphExpander(expand_settings, graph_store.query_all, index)
            # 图不可达时要立刻失败：否则"这一臂没有效果"与"这一臂从未运行"
            # 在结果里长得一模一样。
            print(f"[{name}] 图中条款数 : {expander.verify()['clauses']}")

        retriever = HybridRetriever(
            store,
            embed,
            reranker,
            chunk_snapshot=chunks,
            candidates=kn.hybrid_candidates,
            top_k=kn.top_k,
            rrf_k=kn.rrf_k,
            prefusion_top_n=kn.prefusion_top_n,
            max_chunks_per_document=kn.max_chunks_per_document,
            quota_overflow_top_n=kn.quota_overflow_top_n,
            graph_expander=expander,
        )

        print(f"\n===== 臂 {name} =====")
        arm_items: list[dict] = []
        for question in questions:
            qid = question["id"]
            targets = _TARGETS[qid]
            hits = asyncio.run(_run_question(retriever, question["question"]))
            hops = []
            for doc_title, section in targets:
                found = _hop_hit(hits, doc_title, section)
                hops.append(
                    {
                        "doc": doc_title,
                        "section": section,
                        "hit": found is not None,
                        "evidence": (
                            None if found is None
                            else f"{found.title} | {_section(found)}"
                        ),
                    }
                )
            graph_ctx = sum(
                1 for h in hits if h.metadata_payload.get("graph_expanded")
            )
            group = "判别" if qid in _DISCRIMINATING else "对照"
            hit_n = sum(1 for h in hops if h["hit"])
            arm_items.append(
                {
                    "id": qid,
                    "group": group,
                    "contexts": len(hits),
                    "graph_contexts": graph_ctx,
                    "hits": hit_n,
                    "targets": len(hops),
                    "hops": hops,
                }
            )
            marks = "".join("HIT " if h["hit"] else "MISS" for h in hops)
            print(f"  {qid:12s} [{group}] {hit_n}/{len(hops)}  {marks}"
                  f"  contexts={len(hits)}(图 {graph_ctx})")
            for hop in hops:
                if not hop["hit"]:
                    label = hop["section"] or "全文"
                    print(f"      MISS {hop['doc']} {label}")

        disc = [i for i in arm_items if i["group"] == "判别"]
        ctrl = [i for i in arm_items if i["group"] == "对照"]
        summary = {
            "discriminating": {
                "hits": sum(i["hits"] for i in disc),
                "targets": sum(i["targets"] for i in disc),
            },
            "control": {
                "hits": sum(i["hits"] for i in ctrl),
                "targets": sum(i["targets"] for i in ctrl),
            },
            "all": {
                "hits": sum(i["hits"] for i in arm_items),
                "targets": sum(i["targets"] for i in arm_items),
            },
            "graph_contexts": sum(i["graph_contexts"] for i in arm_items),
            "contexts": sum(i["contexts"] for i in arm_items),
        }
        report["arms"][name] = {"summary": summary, "items": arm_items}

    print("\n" + "=" * 78)
    print(f"{'臂':<10}{'判别命中':>12}{'对照命中':>12}{'合计命中':>12}{'图来源ctx':>12}")
    print("=" * 78)
    for name in report["arms"]:
        s = report["arms"][name]["summary"]
        cells = [
            f"{s['discriminating']['hits']}/{s['discriminating']['targets']}",
            f"{s['control']['hits']}/{s['control']['targets']}",
            f"{s['all']['hits']}/{s['all']['targets']}",
        ]
        print(
            f"{name:<10}"
            + "".join(f"{cell:>12}" for cell in cells)
            + f"{s['graph_contexts']:>12}"
        )

    out = ROOT / args.report
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n结果已写入   : {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
