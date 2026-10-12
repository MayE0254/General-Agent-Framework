"""Structural knowledge-graph builder for the fiscal-tax corpus.

Reads the chunk snapshot (``processed/chunks.jsonl``), derives the structural
layer with ``app.knowledge.graph.extract`` -- Clause nodes plus the
``HAS_CLAUSE`` / ``CITES`` / ``IMPLEMENTS`` / ``REFERENCES`` relations -- and
writes it into Neo4j.  No LLM, no re-embedding, no re-parsing of the raw
documents: the snapshot is already the corpus's own text, so the whole pass is
a pure function of a file and re-runnable at will.

Run from the repo root::

    .venv\\Scripts\\python scripts\\build_graph.py                 # incremental (MERGE)
    .venv\\Scripts\\python scripts\\build_graph.py --report-only    # dry run, print stats
    .venv\\Scripts\\python scripts\\build_graph.py --reset-structure
    .venv\\Scripts\\python scripts\\build_graph.py --verify        # + walk the measured chains

``--reset-structure`` only removes the ``Clause`` nodes and the four relations;
documents, chunks and their ``CONTAINS`` / ``NEXT`` edges (i.e. everything the
expensive embedding pass produced) are untouched.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# The two multi-hop chains the evaluation set actually needs, expressed as
# Cypher over titles/article numbers rather than over internal ids, so the
# check keeps meaning if the corpus is edited.  Each is (name, query, min).
_CHAIN_CHECKS: list[tuple[str, str, int]] = [
    (
        "mh-cit-001 企税法第四条 -> 第三条（同文档）",
        "MATCH (d:Document)-[:HAS_CLAUSE]->(a:Clause) "
        "MATCH (d)-[:HAS_CLAUSE]->(b:Clause) "
        "WHERE d.title = '中华人民共和国企业所得税法' "
        "  AND a.article = 4 AND b.article = 3 "
        "MATCH (a)-[r:CITES]->(b) RETURN count(r) AS c",
        1,
    ),
    (
        "mh-cit-001 实施条例第九十一条 -> 企税法第二十七条（跨文档）",
        "MATCH (reg:Document)-[:HAS_CLAUSE]->(a:Clause)-[r:CITES]->(b:Clause)"
        "<-[:HAS_CLAUSE]-(law:Document) "
        "WHERE reg.title = '中华人民共和国企业所得税法实施条例' "
        "  AND law.title = '中华人民共和国企业所得税法' "
        "  AND a.article = 91 AND b.article = 27 AND r.cross_doc "
        "RETURN count(r) AS c",
        1,
    ),
    (
        "mh-iit-001 扣除标准通知 -> 个人所得税法（文档级引用）",
        "MATCH (notice:Document)-[r:REFERENCES]->(law:Document) "
        "WHERE notice.title = '国务院关于提高个人所得税有关专项附加扣除标准的通知' "
        "  AND law.title = '中华人民共和国个人所得税法' "
        "RETURN count(r) AS c",
        1,
    ),
    (
        "跨文档 CITES 边总量",
        "MATCH ()-[r:CITES]->() WHERE r.cross_doc RETURN count(r) AS c",
        1,
    ),
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


def _stream_snapshot(path: Path):
    """Yield snapshot records one at a time (40k+ lines, never load it whole)."""
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _verify(graph) -> bool:
    """Walk the measured chains in the written graph; True when all pass."""
    print("\n--- 校验（用真实链路验证图是否连通）---")
    ok = True
    for name, query, minimum in _CHAIN_CHECKS:
        rec = graph.query_one(query)
        value = int(rec["c"]) if rec else 0
        passed = value >= minimum
        ok = ok and passed
        print(f"[{'PASS' if passed else 'FAIL'}] {name} : {value}")
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--snapshot", default="data/knowledge/processed/chunks.jsonl",
                        help="chunk 快照路径（默认 data/knowledge/processed/chunks.jsonl）")
    parser.add_argument("--batch-size", type=int, default=500,
                        help="每批 UNWIND 写入的行数")
    parser.add_argument("--report-only", action="store_true",
                        help="只抽取并打印统计，不连接 Neo4j、不写图")
    parser.add_argument("--reset-structure", action="store_true",
                        help="写入前删除 Clause 节点与四类结构边（不动 Document/Chunk 与向量库）")
    parser.add_argument("--verify", action="store_true",
                        help="写入后跑多跳链路校验（mh-cit-001 / mh-iit-001）")
    args = parser.parse_args(argv)

    _inject_user_env("NEO4J_PASSWORD")

    from app.knowledge.graph.extract import extract_structure

    snapshot = ROOT / args.snapshot
    if not snapshot.exists():
        print(f"快照不存在：{snapshot}")
        print("请先运行 scripts/ingest_knowledge.py 生成 chunks.jsonl。")
        return 1

    started = time.monotonic()
    extraction = extract_structure(_stream_snapshot(snapshot))
    elapsed = time.monotonic() - started

    print(f"snapshot     : {snapshot}")
    print(f"抽取耗时     : {elapsed:.1f}s")
    for key, value in extraction.stats.items():
        print(f"  {key:<27s}: {value}")

    if args.report_only:
        print("\n--report-only : 未连接 Neo4j，未写入任何数据")
        return 0

    from app.core import load_settings
    from app.knowledge.graph.neo4j import Neo4jGraphStore

    settings = load_settings(ROOT / "configs" / "application.toml")
    graph = Neo4jGraphStore(settings.knowledge.graph)
    try:
        graph.ensure_constraints()
        if args.reset_structure:
            removed = graph.reset_structure()
            print(f"\n--reset-structure : 已删除 Clause 节点 {removed['clauses']}"
                  f" / 结构边 {removed['doc_edges']}")
        written = graph.write_structure(extraction, batch_size=args.batch_size)
        nodes = graph.counts()
        rels = graph.relation_counts()
    except Exception as exc:
        print(f"neo4j        : 不可用 -> {type(exc).__name__}: {exc}")
        return 1

    print(f"\nneo4j        : {settings.knowledge.graph.uri}")
    print(f"本次写入     : {written}")
    print(f"节点计数     : {nodes}")
    print(f"关系计数     : {rels}")

    if args.verify and not _verify(graph):
        graph.close()
        return 1
    graph.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
