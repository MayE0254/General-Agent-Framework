"""可视化知识库构建与混合检索全链路 Demo（手动运行，非 pytest）。

用法（在项目根目录）：

    1) 小样本入库 + 检索演示（推荐先试，需联网下载一次 bge 模型）：
       .venv\\Scripts\\python tests\\demo\\knowledge_demo.py \\
           --corpus "data/knowledge/raw/Regulations Library-20260701" \\
           --limit 5 \\
           --query "符合什么条件可以享受增值税进项税额抵扣"

    2) 只跑解析+切分可视化（不连 Milvus/Neo4j，不下载模型）：
       .venv\\Scripts\\python tests\\demo\\knowledge_demo.py --corpus "<...>" --limit 3 --chunk-only

说明：
- 本脚本**不在 pytest 自动执行**，属正式运行之外的调试/调参工具；
- 首次使用会从 HuggingFace 下载 BAAI/bge-base-zh-v1.5 与 BAAI/bge-reranker-base，
  若网络受限可设置 ``HF_ENDPOINT=https://hf-mirror.com`` 走镜像；
- ``--limit N`` 控制处理文件数（大语料请用小 N 验证效果再全量跑）。
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Promote User-scope password vars so ${VAR} refs in application.toml resolve.
for _name in ("MILVUS_PASSWORD", "NEO4J_PASSWORD"):
    if not os.environ.get(_name):
        import subprocess

        _p = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "[Environment]::GetEnvironmentVariable('" + _name + "','User')"],
            capture_output=True, text=True, timeout=15,
        )
        if _p.stdout.strip():
            os.environ[_name] = _p.stdout.strip()

from app.core import load_settings  # noqa: E402
from app.core.settings import Settings  # noqa: E402
from app.knowledge.diversity import (  # noqa: E402
    cap_per_document,
    collapse_by_parent,
    expand_to_parent,
)
from app.knowledge.embeddings import LocalEmbeddingModel  # noqa: E402
from app.knowledge.fusion import reciprocal_rank_fusion  # noqa: E402
from app.knowledge.graph.neo4j import Neo4jGraphStore  # noqa: E402
from app.knowledge.hybrid import HybridRetriever  # noqa: E402
from app.knowledge.ingest.chunker import chunk_document  # noqa: E402
from app.knowledge.ingest.parsers import parse_document  # noqa: E402
from app.knowledge.ingest.pipeline import IngestionPipeline, load_chunk_snapshot  # noqa: E402
from app.knowledge.rerank import CrossEncoderReranker  # noqa: E402
from app.knowledge.vectorstore.milvus import MilvusStore  # noqa: E402

_SETTINGS = load_settings(ROOT / "configs" / "application.toml")


# ---------------------------------------------------------------------------
# 阶段 A：解析 + 切分可视化
# ---------------------------------------------------------------------------
def show_chunking(corpus_dir: str, limit: int) -> list[dict]:
    kn = _SETTINGS.knowledge
    root = Path(corpus_dir)
    files = sorted(p for p in root.rglob("*") if p.is_file()
                   and p.suffix.lower() in {".docx", ".pdf", ".xlsx"})[:limit]

    print("\n" + "=" * 78)
    print(f"阶段 A | 文档解析与切分可视化（parent={kn.ingest.chunk_size_chars}, "
          f"child={kn.ingest.child_window_chars}, overlap={kn.ingest.chunk_overlap_chars}）")
    print("=" * 78)
    preview_records: list[dict] = []
    for fp in files:
        parsed = parse_document(str(fp), str(root))
        chunks = chunk_document(
            parsed,
            chunk_size=kn.ingest.chunk_size_chars,
            overlap=kn.ingest.chunk_overlap_chars,
            child_window=kn.ingest.child_window_chars,
        )
        parents = {c.parent_id for c in chunks}
        split_parents = {c.parent_id for c in chunks if c.parent_text}
        print(f"\n【文件】{parsed.source_rel}")
        print(f"  标题: {parsed.title}  编号: {parsed.file_no}  类型: {parsed.doc_type}")
        print(f"  段落数: {len(parsed.paragraphs)}  父块数: {len(parents)}  子块数: {len(chunks)}")
        print(f"  被切分的条款数: {len(split_parents)} "
              f"（其余 {len(parents) - len(split_parents)} 个条款本身即最小检索单元）")
        for c in chunks[:3]:
            print(f"  - [{c.chunk_index}] section={c.section!r} child_chars={len(c.text)} "
                  f"parent_chars={c.metadata_payload.get('parent_char_len', 0)}")
            print(f"      id={c.id}  parent_id={c.parent_id}")
            print(f"      text: {c.text[:120]}...")
            if c.parent_text:
                print(f"      parent_text: {c.parent_text[:120]}...")
            print(f"      meta: {c.metadata_payload}")
        preview_records.append({
            "file": parsed.source_rel,
            "title": parsed.title,
            "file_no": parsed.file_no,
            "doc_type": parsed.doc_type,
            "num_chunks": len(chunks),
            "num_parents": len(parents),
        })
    return preview_records


# ---------------------------------------------------------------------------
# 阶段 B：入库（嵌入 -> Milvus -> Neo4j）
# ---------------------------------------------------------------------------
def run_ingestion(corpus_dir: str, limit: int | None) -> dict:
    print("\n" + "=" * 78)
    print("阶段 B | 向量化入库 -> Milvus + Neo4j 骨架")
    print("=" * 78)
    kn = _SETTINGS.knowledge
    embed = LocalEmbeddingModel(kn.embedding_model)
    store = MilvusStore(_SETTINGS.knowledge.milvus)
    graph = Neo4jGraphStore(_SETTINGS.knowledge.graph)

    created = store.ensure_collection(dim=kn.embedding_dim)
    print(f"Milvus collection '{store.collection}' {'已创建' if created else '已存在'}")
    try:
        graph.ensure_constraints()
        print("Neo4j 约束已就绪")
    except Exception as exc:
        print(f"Neo4j 约束处理警告: {exc}")

    pipeline = IngestionPipeline(kn, embedding=embed, store=store, graph=graph)
    report = pipeline.run(corpus_dir, limit=limit,
                          output_dir=str(ROOT / "data/knowledge/processed"))
    print(f"入库结果: {report}")
    graph.close()
    return report


# ---------------------------------------------------------------------------
# 阶段 C：混合检索（向量 + BM25 -> 融合 -> cross-encoder 重排）
# ---------------------------------------------------------------------------
def run_hybrid_retrieval(corpus_dir: str, query: str) -> list[dict]:
    print("\n" + "=" * 78)
    print("阶段 C | 混合检索可视化：向量 + BM25 → RRF → 去重/配额 → 重排 → 展开父块")
    print("=" * 78)
    kn = _SETTINGS.knowledge
    embed = LocalEmbeddingModel(kn.embedding_model)
    store = MilvusStore(_SETTINGS.knowledge.milvus)
    reranker = CrossEncoderReranker(kn.reranker_model)

    snapshot_path = ROOT / "data/knowledge/processed/chunks.jsonl"
    if not snapshot_path.exists():
        print("未发现 chunk 快照（chunks.jsonl），请先运行入库阶段 B（不传 --chunk-only）。")
        print("先执行一次：knowledge_demo.py --corpus ... --limit N")
        return []
    hits_known = load_chunk_snapshot(str(snapshot_path))

    retriever = HybridRetriever(
        store, embed, reranker,
        chunk_snapshot=hits_known,
        candidates=kn.hybrid_candidates,
        top_k=kn.top_k,
        rrf_k=kn.rrf_k,
        prefusion_top_n=kn.prefusion_top_n,
        max_chunks_per_document=kn.max_chunks_per_document,
    )

    query_vec = embed.embed_query(query)
    print(f"\n查询: {query}")
    print(f"query embedding 维度: {len(query_vec)}")

    # 1) 向量腿
    vec_hits = store.search(query_vec, top_k=kn.hybrid_candidates)
    print(f"\n[向量腿] top-{len(vec_hits)}:")
    for i, h in enumerate(vec_hits, 1):
        print(f"  #{i} score={h.score}  [{h.source}] {h.title} | {h.metadata_payload.get('section', '')}")

    # 2) BM25 腿
    bm_hits = retriever.bm25.to_hits(retriever.bm25.search(query, top_k=kn.hybrid_candidates)) \
        if retriever.bm25 else []
    print(f"\n[关键词腿 BM25] top-{len(bm_hits)}:")
    for i, h in enumerate(bm_hits, 1):
        print(f"  #{i} score={h.score}  [{h.source}] {h.title}")

    # 3) RRF 初筛：按名次融合两腿（规避 cosine / BM25 量纲不可比），并截断控制精排规模
    fused = reciprocal_rank_fusion([vec_hits, bm_hits], k=kn.rrf_k)
    cutoff = max(kn.prefusion_top_n, kn.top_k)
    prescreened = fused[:cutoff]
    print(f"\n[RRF 初筛] rrf_k={kn.rrf_k}  融合去重后={len(fused)} 个 → "
          f"进入精排={len(prescreened)} 个（上限 prefusion_top_n={kn.prefusion_top_n}）:")
    for i, h in enumerate(prescreened, 1):
        p = h.metadata_payload
        print(f"  #{i} rrf={p.get('rrf_score', 0.0):.5f} "
              f"vec={p.get('vector_score', '-')} bm25={p.get('bm25_score', '-')}  "
              f"[{h.source}] {h.title} | {p.get('section', '')}")

    # 3b) 与 HybridRetriever 完全一致的去重 + 配额（避免两处逻辑漂移）
    collapsed = collapse_by_parent(fused)
    capped, overflow = cap_per_document(
        collapsed,
        max_per_document=kn.max_chunks_per_document,
        limit=cutoff,
        overflow_limit=kn.quota_overflow_top_n,
    )
    print(f"\n[父块折叠] {len(fused)} → {len(collapsed)} 个"
          f"（同一条款的多个子窗口合并，保留名次最高者）")
    print(f"[按源文档配额] max_chunks_per_document={kn.max_chunks_per_document} → {len(capped)} 个:")
    per_src: dict[str, int] = {}
    for h in capped:
        per_src[h.source] = per_src.get(h.source, 0) + 1
    for src, n in sorted(per_src.items(), key=lambda kv: -kv[1]):
        print(f"  {n:>3} 块  {src}")
    if overflow:
        print(f"[配额排除的补充候选] quota_overflow_top_n={kn.quota_overflow_top_n}"
              f" → {len(overflow)} 个（未经 cross-encoder，仅作提示）:")
        for i, h in enumerate(overflow, 1):
            print(f"  #{i} [{h.source}] {h.title} | {h.metadata_payload.get('section', '')}")

    # 4) cross-encoder 精排（最终排序只由它决定）
    reranked = reranker.rerank(query, capped)
    # 4b) 展开父块：命中的是子窗口，交付的是完整条款
    final = expand_to_parent(reranked)[: kn.top_k]

    print(f"\n[重排后 top-{len(final)}]（cross-encoder 打分，降序；content 已展开为父块）:")
    for i, h in enumerate(final, 1):
        p = h.metadata_payload
        print(f"  #{i} rerank={p.get('rerank_score', h.score):.4f} "
              f"rrf={p.get('rrf_score', 0.0):.5f} "
              f"vec={p.get('vector_score', '-')} bm25={p.get('bm25_score', '-')}")
        print(f"      [{h.source}] {h.title} | {p.get('section', '')}")
        print(f"      parent_id={h.parent_id}  折叠掉同父兄弟={p.get('collapsed_siblings', 0)} "
              f"匹配子块长度={p.get('matched_child_char_len', len(h.content))}")
        print(f"      {h.content[:150]}...")

    return [h.model_dump() for h in final]


# ---------------------------------------------------------------------------
# 阶段 D：指标（轻量记录式）
# ---------------------------------------------------------------------------
def show_metrics(query: str) -> None:
    print("\n" + "=" * 78)
    print("阶段 D | 简单指标记录（weak-label 提示 + 排名漂移）")
    print("=" * 78)
    import jieba
    jieba.initialize()
    toks = [t for t in jieba.cut(query) if len(t.strip()) > 1]
    print(f"查询分词: {toks}")
    print("人工对照阶段 C 输出做相关性判定；脚本辅助指标：")
    print(f"  - RRF 初筛把两腿合并后的候选数 与 进入精排的候选数 对比："
          f"（rrf_k={_SETTINGS.knowledge.rrf_k}, "
          f"prefusion_top_n={_SETTINGS.knowledge.prefusion_top_n}）")
    print("  - 初筛排名 与 重排后排名 的漂移：看 cross-encoder 是否纠正了 RRF 的名次")
    print("  - 重排后 top-k 来源文档是否覆盖了多篇相关法规 → recall 覆盖")
    print("  - 一个 chunk 同时带 vec 与 bm25 分 → 被两腿共同召回（RRF 名次更靠前）")


# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="知识库构建与混合检索 Demo")
    parser.add_argument("--corpus", required=True,
                        help="知识库语料根目录（如 data/knowledge/raw/Regulations Library-20260701）")
    parser.add_argument("--limit", type=int, default=None, help="仅处理前 N 个文件（小样本）")
    parser.add_argument("--query", default="符合什么条件可以享受增值税进项税额抵扣",
                        help="检索演示查询")
    parser.add_argument("--chunk-only", action="store_true",
                        help="只跑 A（解析+切分），不连库、不下载模型")
    args = parser.parse_args()

    if args.chunk_only:
        preview = show_chunking(args.corpus, args.limit or 3)
        total = sum(r["num_chunks"] for r in preview)
        print(f"\n切分可视化完成：{len(preview)} 篇，共 {total} 个 chunk。")
        print("（--chunk-only 未连接 Milvus/Neo4j，也未加载模型）")
        return 0

    show_chunking(args.corpus, args.limit or 3)
    run_ingestion(args.corpus, args.limit)
    run_hybrid_retrieval(args.corpus, args.query)
    show_metrics(args.query)
    print("\nDemo 完成。以上：切分形态 / metadata / 来源 / 召回候选 / 排名 / 重排 / 输出 / 指标均已打印。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())