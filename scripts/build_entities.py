"""Semantic entity-layer builder for the fiscal-tax corpus.

Reads the chunk snapshot, asks an LLM (one call per batch of clauses, i.e.
per document in practice) which entities of a closed type list each clause
mentions, and writes ``Entity`` nodes plus ``MENTIONS`` edges into Neo4j.  This
is the paid, non-deterministic half of the graph; the free half is
``scripts/build_graph.py`` and must be run first (it creates the ``Clause``
nodes these edges attach to).

Two phases, which is what makes the run resumable:

1. **Extraction** -- documents whose result is not already in the snapshot are
   sent to the model and appended to ``data/knowledge/graph/entities.jsonl``.
   The snapshot is the cache: re-running skips them, ``--no-resume`` redoes
   them.
2. **Write** -- the *snapshot* (not the live results) is replayed into Neo4j
   with ``MERGE``.  So ``--report-only`` costs nothing later, a crash during
   writing is fixed by re-running without paying the model again, and the graph
   can never contain an entity the snapshot does not account for.

Run from the repo root::

    .venv\\Scripts\\python scripts\\build_entities.py --limit 20 --report-only
    .venv\\Scripts\\python scripts\\build_entities.py --reset-entities --verify
    .venv\\Scripts\\python scripts\\build_entities.py --only 个人所得税
    .venv\\Scripts\\python scripts\\build_entities.py --no-resume

Clause text is taken from ``app.knowledge.graph.extract.extract_structure``
rather than re-read from the snapshot here: picking the parent text is subtle
(a long clause is split into several parents sharing one section label), and
two code paths choosing text differently would silently produce entities
attached to a different clause than the structural layer's.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_SNAPSHOT = "data/knowledge/processed/chunks.jsonl"
DEFAULT_OUT = "data/knowledge/graph/entities.jsonl"

# Bridges the multi-hop evaluation set needs and the structural layer does not
# provide (it links 第三条->第四条 and 第二十七条->第九十一条, but nothing joins
# 第四条 to 第二十七条, and nothing joins 个税法第六条 to the notices).  A
# shared entity is exactly that missing hop, so each check asserts the hop
# exists.  min=1 means "at least one shared entity": the claim under test is
# that the LLM layer bridges these pairs at all.
_BRIDGE_CHECKS: list[tuple[str, str, int]] = [
    (
        "mh-cit-001 企税法第四条 <-共享实体-> 实施条例第九十一条（跨文档，非引用）",
        "MATCH (d1:Document)-[:HAS_CLAUSE]->(a:Clause)-[:MENTIONS]->(e:Entity)"
        "<-[:MENTIONS]-(b:Clause)<-[:HAS_CLAUSE]-(d2:Document) "
        "WHERE d1.title = '中华人民共和国企业所得税法' AND a.article = 4 "
        "  AND d2.title = '中华人民共和国企业所得税法实施条例' AND b.article = 91 "
        "RETURN count(DISTINCT e) AS c",
        1,
    ),
    (
        "mh-iit-001 个税法第六条 <-共享实体-> 提高扣除标准通知",
        "MATCH (d1:Document)-[:HAS_CLAUSE]->(a:Clause)-[:MENTIONS]->(e:Entity)"
        "<-[:MENTIONS]-(b:Clause)<-[:HAS_CLAUSE]-(d2:Document) "
        "WHERE d1.title = '中华人民共和国个人所得税法' AND a.article = 6 "
        "  AND d2.title CONTAINS '提高个人所得税有关专项附加扣除标准的通知' "
        "RETURN count(DISTINCT e) AS c",
        1,
    ),
    (
        "mh-vat-001 增值税法第十条 <-共享实体-> 适用9%税率货物范围注释",
        "MATCH (d1:Document)-[:HAS_CLAUSE]->(a:Clause)-[:MENTIONS]->(e:Entity)"
        "<-[:MENTIONS]-(b:Clause)<-[:HAS_CLAUSE]-(d2:Document) "
        "WHERE d1.title CONTAINS '增值税法' AND a.article = 10 "
        "  AND d2.title CONTAINS '货物范围注释' "
        "RETURN count(DISTINCT e) AS c",
        1,
    ),
]

# Informational: a vocabulary that collapsed onto one type, or entities that
# only ever occur once, is the shape of a model answering "something" per
# clause.  Printed, not asserted -- there is no defensible threshold yet.
_DISTRIBUTION_QUERY = (
    "MATCH (e:Entity) WITH e.type AS t, count(*) AS c "
    "RETURN collect(t + '=' + toString(c)) AS dist, sum(c) AS total"
)
_SINGLETON_QUERY = (
    "MATCH (e:Entity)<-[:MENTIONS]-() WITH e, count(*) AS c "
    "WHERE c = 1 RETURN count(e) AS c"
)


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


def _load_results(path: Path) -> dict[str, dict]:
    """Last line per document wins, so a re-run can overwrite a failure."""
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            doc_id = record.get("doc_id")
            if doc_id:
                out[doc_id] = record
    return out


def _group_clauses(extraction) -> dict[str, list]:
    """Group the structural layer's clauses by document, preserving order."""
    from app.knowledge.graph.entities import ClauseInput

    docs: dict[str, list] = {}
    for clause in extraction.clauses:
        docs.setdefault(clause.doc_id, []).append(
            ClauseInput(
                clause_id=clause.clause_id,
                doc_id=clause.doc_id,
                title=clause.title,
                section=clause.section,
                text=clause.text,
            )
        )
    return docs


def _matches(clauses: list, only: str) -> bool:
    if not only:
        return True
    needle = only.lower()
    for clause in clauses:
        if needle in clause.title.lower() or needle in clause.doc_id.lower():
            return True
    return False


async def _extract_all(
    extractor,
    docs: dict[str, list],
    done: set[str],
    out_path: Path,
    *,
    concurrency: int,
) -> tuple[int, int]:
    """Phase 1: call the model for every document not already in the snapshot.

    Returns ``(extracted, failed)``.  Results are appended as they arrive, so a
    Ctrl-C keeps everything already paid for.
    """
    pending = [(doc_id, clauses) for doc_id, clauses in docs.items() if doc_id not in done]
    if not pending:
        return 0, 0

    semaphore = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    handle = out_path.open("a", encoding="utf-8")
    counters = {"done": 0, "failed": 0}

    async def _one(doc_id: str, clauses: list) -> None:
        async with semaphore:
            title = clauses[0].title if clauses else ""
            try:
                result = await extractor.extract_document(doc_id, title, clauses)
                record = {
                    "doc_id": doc_id,
                    "title": title,
                    "status": "ok",
                    "mentions": [
                        {
                            "clause_id": m.clause_id,
                            "type": m.type_key,
                            "name": m.name,
                            "extractor": m.extractor,
                        }
                        for m in result.mentions
                    ],
                    "stats": result.stats,
                }
            except Exception as exc:  # recorded, never swallowed
                counters["failed"] += 1
                record = {
                    "doc_id": doc_id,
                    "title": title,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            async with lock:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                counters["done"] += 1
                flag = "" if record["status"] == "ok" else "  [失败]"
                if extractor.circuit_open:
                    flag += "  [降级腿]"
                print(
                    f"[{counters['done']}/{len(pending)}] {doc_id.split('/')[-1][:48]}"
                    f"  mentions={len(record.get('mentions', []))}{flag}",
                    flush=True,
                )

    await asyncio.gather(*(_one(d, c) for d, c in pending))
    handle.close()
    return counters["done"], counters["failed"]


def _write_all(graph, results: dict[str, dict], docs: dict[str, list], *, batch_size: int) -> dict:
    """Phase 2: replay the snapshot into Neo4j (idempotent MERGE)."""
    from app.knowledge.graph.entities import EntityMention, EntityExtraction

    written = {"docs": 0, "entities": 0, "mentions": 0, "skipped": 0, "errors": 0}
    for doc_id, record in results.items():
        if doc_id not in docs:
            written["skipped"] += 1
            continue
        if record.get("status") != "ok":
            written["errors"] += 1
            continue
        extraction = EntityExtraction(
            doc_id=doc_id,
            mentions=[
                EntityMention(
                    clause_id=item["clause_id"],
                    type_key=item["type"],
                    name=item["name"],
                    extractor=item.get("extractor", ""),
                )
                for item in record.get("mentions", [])
            ],
        )
        counts = graph.write_entities(extraction, batch_size=batch_size)
        written["docs"] += 1
        written["entities"] += counts["entities"]
        written["mentions"] += counts["mentions"]
    return written


def _report(graph) -> bool:
    """Print the entity layer's shape and walk the bridges; True when they hold."""
    rec = graph.query_one(_DISTRIBUTION_QUERY)
    if rec:
        print(f"类型分布     : {sorted(rec.get('dist') or [])}")
    single = graph.query_one(_SINGLETON_QUERY)
    total = int(rec["total"]) if rec else 0
    singletons = int(single["c"]) if single else 0
    if total:
        print(f"仅出现一次   : {singletons}/{total} ({singletons / total:.1%})")

    print("\n--- 语义桥校验（结构层接不上的那几跳）---")
    ok = True
    for name, query, minimum in _BRIDGE_CHECKS:
        hit = graph.query_one(query)
        value = int(hit["c"]) if hit else 0
        passed = value >= minimum
        ok = ok and passed
        print(f"[{'PASS' if passed else 'FAIL'}] {name} : {value}")
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--snapshot", default=DEFAULT_SNAPSHOT,
                        help=f"chunk 快照（默认 {DEFAULT_SNAPSHOT}）")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help=f"抽取结果快照，断点续跑的依据（默认 {DEFAULT_OUT}）")
    parser.add_argument("--limit", type=int, default=None,
                        help="只处理前 N 篇文档（冒烟用；仍会真实调用模型）")
    parser.add_argument("--only", default="", help="只处理标题/路径含该子串的文档")
    parser.add_argument("--concurrency", type=int, default=None,
                        help="并发文档数（默认取配置 knowledge.graph.entity.concurrency）")
    parser.add_argument("--batch-size", type=int, default=500, help="每批 UNWIND 写入的行数")
    parser.add_argument("--report-only", action="store_true",
                        help="只抽取并打印统计，不连接 Neo4j（抽取结果仍写入快照）")
    parser.add_argument("--reset-entities", action="store_true",
                        help="写入前删除 Entity 节点与 MENTIONS 边（不动 Clause/结构边/向量库）")
    parser.add_argument("--no-resume", action="store_true",
                        help="忽略已有快照，重新调用模型抽取")
    parser.add_argument("--verify", action="store_true", help="写入后跑语义桥校验")
    args = parser.parse_args(argv)

    for var in ("NEO4J_PASSWORD", "ZHIPU_API_KEY", "DEEPSEEK_API_KEY"):
        _inject_user_env(var)

    from app.core import load_settings
    from app.knowledge.graph.entity_client import EntityExtractor
    from app.knowledge.graph.extract import extract_structure
    from app.llm.factory import create_llm_service

    settings = load_settings(ROOT / "configs" / "application.toml")
    entity = settings.knowledge.graph.entity
    if not entity.enabled:
        print("knowledge.graph.entity.enabled = false，实体层已关闭；"
              "如需运行请先在 configs/application.toml 中开启。")
        return 1

    snapshot = ROOT / args.snapshot
    if not snapshot.exists():
        print(f"快照不存在：{snapshot}")
        print("请先运行 scripts/ingest_knowledge.py 生成 chunks.jsonl。")
        return 1

    started = time.monotonic()
    extraction = extract_structure(_stream_snapshot(snapshot))
    docs = _group_clauses(extraction)
    if args.only:
        docs = {d: c for d, c in docs.items() if _matches(c, args.only)}
    if args.limit:
        docs = dict(list(docs.items())[: args.limit])

    out_path = ROOT / args.out
    results = {} if args.no_resume else _load_results(out_path)
    done = {d for d, r in results.items() if r.get("status") == "ok"}

    print(f"chunk 快照   : {args.snapshot}")
    print(f"抽取快照     : {args.out}")
    print(f"条款         : {sum(len(c) for c in docs.values())} / 文档 {len(docs)}")
    print(f"已有结果     : {len(done)}（本次跳过；--no-resume 可强制重跑）")
    print(f"抽取模型     : {entity.model} (provider={entity.provider})")
    if not entity.fallback_provider:
        print("降级腿       : 未配置（主腿失败即上抛）")
    else:
        print(f"降级腿       : {entity.fallback_provider} / {entity.fallback_model}"
              f"  extra_body={entity.fallback_extra_body or '{}'}")
        if entity.fallback_model not in settings.llm.pricing:
            print(f"  [注意] {entity.fallback_model} 不在 [llm].pricing 中，"
                  "该腿成本将记为 0（只影响成本可见性，不影响抽取）")
    print()

    extractor = EntityExtractor(create_llm_service(settings), entity)
    try:
        extracted, failed = asyncio.run(
            _extract_all(
                extractor, docs, done, out_path,
                concurrency=args.concurrency or entity.concurrency,
            )
        )
    except KeyboardInterrupt:
        print("\n已中断：快照保留已抽取的结果，重跑即可续跑。")
        return 130
    elapsed = time.monotonic() - started

    stats = extractor.stats
    print(f"\n本次抽取     : {extracted} 篇（失败 {failed}）")
    print(f"耗时         : {elapsed:.1f}s")
    print(f"调用         : 合计 {stats.calls}（主腿 {stats.primary_calls} / "
          f"降级腿 {stats.fallback_calls}）")
    if stats.circuit_reason:
        print(f"熔断         : 第 {stats.circuit_docs} 篇起切降级腿 -> {stats.circuit_reason}")
    print(f"截断/解析失败 : {stats.truncated} / {stats.parse_failures}"
          f"（二分重试 {stats.split_retries} 次）")
    print(f"token        : 输入 {stats.prompt_tokens} / 输出 {stats.completion_tokens}")
    print(f"成本         : ${stats.cost_usd:.4f}（未登记单价的模型记为 0）")
    for key in sorted(stats.discards):
        print(f"  {key:<27s}: {stats.discards[key]}")

    if args.report_only:
        print("\n--report-only : 未连接 Neo4j，未写入任何数据（快照已落盘）")
        return 0

    from app.knowledge.graph.neo4j import Neo4jGraphStore

    graph = Neo4jGraphStore(settings.knowledge.graph)
    try:
        graph.ensure_constraints()
        if args.reset_entities:
            removed = graph.reset_entities()
            print(f"\n--reset-entities : 已删除 Entity {removed['entities']} 个")
        # Replaying the whole snapshot, not just this run's documents: the write
        # is idempotent, and it is what makes --report-only / a crashed write
        # recoverable without paying the model twice.
        written = _write_all(
            graph, {d: r for d, r in _load_results(out_path).items() if d in docs},
            docs, batch_size=args.batch_size,
        )
        nodes = graph.counts()
        rels = graph.relation_counts()
    except Exception as exc:
        print(f"neo4j        : 不可用 -> {type(exc).__name__}: {exc}")
        return 1

    print(f"\nneo4j        : {settings.knowledge.graph.uri}")
    print(f"本次写入     : {written}")
    print(f"节点计数     : {nodes}")
    print(f"关系计数     : {rels}")
    if written["mentions"] != rels.get("MENTIONS", 0):
        print(f"  [说明] 图中 MENTIONS 总量 {rels.get('MENTIONS', 0)} 与本轮写入去重后的"
              f" {written['mentions']} 不同：图中还含本次范围之外的实体边"
              "（增量写入的既有结果）；--reset-entities 后重跑可对齐")

    if args.verify and not _report(graph):
        graph.close()
        return 1
    graph.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
