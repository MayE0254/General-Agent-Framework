"""Full / incremental ingestion CLI for the fiscal-tax knowledge base.

Parses every accepted document under the corpus root, embeds the chunks into
Milvus, writes the ``Document`` / chunk skeleton plus the ``NEXT`` ordering
chain into Neo4j, and refreshes ``processed/chunks.jsonl`` — the snapshot the
BM25 leg of hybrid retrieval is rebuilt from.

Run from the repo root::

    .venv\\Scripts\\python scripts\\ingest_knowledge.py                # resume / incremental
    .venv\\Scripts\\python scripts\\ingest_knowledge.py --reset         # clean rebuild
    .venv\\Scripts\\python scripts\\ingest_knowledge.py --no-graph      # vectors only
    .venv\\Scripts\\python scripts\\ingest_knowledge.py --reset-graph   # graph only
    .venv\\Scripts\\python scripts\\ingest_knowledge.py --limit 20      # smoke run

Progress is durable: ``processed/progress.json`` is written atomically every
``--persist-every`` files and again on exit, so an interrupted run resumes
where it stopped instead of re-embedding the corpus.  Vector work and graph
work carry separate markers, so a Neo4j outage never forces a re-embed — and
``--reset-graph`` rebuilds the graph off those markers without touching Milvus.

Cost note: on a CPU-only host embedding bge-base-zh runs at roughly 2 chunks/s,
i.e. a ~2.5e4-chunk corpus takes hours.  Run it detached and watch the periodic
``[progress]`` lines (or ``progress.json``) rather than the terminal.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


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


def _prefer_offline(model_ref: str) -> None:
    """Keep transformers from probing the hub when the model is a local dir."""
    if Path(model_ref).is_dir():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _watch_progress(
    progress_path: Path,
    targets: set[str],
    total: int,
    stop: threading.Event,
    interval: int,
) -> None:
    """Print a periodic ``done/total`` line by polling the progress marker."""
    started = time.monotonic()
    while not stop.wait(interval):
        try:
            data = json.loads(progress_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        done = len(set(data.get("done", [])) & targets)
        graphed = len(set(data.get("graph_done", [])) & targets)
        elapsed = (time.monotonic() - started) / 60
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (total - done) / rate if rate > 0 else 0.0
        print(
            f"[progress] vec={done}/{total} graph={graphed}/{total} "
            f"elapsed={elapsed:.1f}min rate={rate:.1f} files/min eta={eta:.0f}min",
            flush=True,
        )


def _snapshot_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as fh:
        return sum(1 for _ in fh)


def _clear_graph_marker(progress_path: Path) -> int:
    """Drop ``graph_done`` from the progress marker, keeping ``done`` intact.

    ``--reset-graph`` wipes the graph and nothing else, so the marker must lose
    its graph half while keeping the vector half: an intact ``graph_done`` would
    make every file look already-graphed and turn the rebuild into a silent
    no-op that still exits 0.
    """
    if not progress_path.exists():
        return 0
    try:
        data = json.loads(progress_path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    cleared = len(data.get("graph_done", []))
    if not cleared:
        return 0
    data["graph_done"] = []
    tmp = progress_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(progress_path)
    return cleared


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", default=None,
                        help="语料根目录（默认取配置 knowledge.ingest.source_dir）")
    parser.add_argument("--limit", type=int, default=None,
                        help="仅处理前 N 个文件（冒烟用）")
    parser.add_argument("--output-dir", default="data/knowledge/processed",
                        help="progress.json / chunks.jsonl 输出目录")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="嵌入批大小（默认取配置 knowledge.ingest.batch_size）")
    parser.add_argument("--persist-every", type=int, default=25,
                        help="每处理 N 个文件原子落盘一次进度")
    parser.add_argument("--report-every", type=int, default=60,
                        help="进度打印间隔（秒）")
    parser.add_argument("--reset", action="store_true",
                        help="先清空 Milvus collection、Neo4j 骨架与进度标记，做纯净重建")
    parser.add_argument("--no-graph", action="store_true",
                        help="跳过 Neo4j 图谱写入（仅向量 + BM25 快照）")
    parser.add_argument("--reset-graph", action="store_true",
                        help="只清空 Neo4j 骨架并清掉图中的完成标记，不动 Milvus；"
                             "用于向量库已就绪、只需重建图谱的场合")
    args = parser.parse_args(argv)

    if args.reset_graph and args.no_graph:
        parser.error("--reset-graph 与 --no-graph 互斥：前者为重建图谱，后者为跳过图谱")

    for var in ("MILVUS_PASSWORD", "NEO4J_PASSWORD"):
        _inject_user_env(var)

    from app.core import load_settings
    from app.knowledge.embeddings import LocalEmbeddingModel
    from app.knowledge.graph.neo4j import Neo4jGraphStore
    from app.knowledge.ingest.pipeline import IngestionPipeline, discover_files
    from app.knowledge.vectorstore.milvus import MilvusStore

    settings = load_settings(ROOT / "configs" / "application.toml")
    kn = settings.knowledge

    corpus = Path(args.corpus) if args.corpus else ROOT / kn.ingest.source_dir
    output_dir = ROOT / args.output_dir
    progress_path = output_dir / "progress.json"
    batch_size = args.batch_size or kn.ingest.batch_size

    targets = discover_files(str(corpus), limit=args.limit)
    target_rels = {
        str(p.relative_to(corpus)).replace("\\", "/") for p in targets
    }
    print(f"corpus       : {corpus}")
    print(f"files        : {len(targets)} accepted "
          f"(graph={'off' if args.no_graph else 'on'})")

    if args.reset:
        for stale in (progress_path, output_dir / "chunks.jsonl"):
            if stale.exists():
                stale.unlink()
        print("--reset      : 已清除 progress.json / chunks.jsonl")

    _prefer_offline(kn.embedding_model)
    store = MilvusStore(kn.milvus)
    if args.reset:
        store.reset()
        print("--reset      : 已 drop Milvus collection")
    created = store.ensure_collection(dim=kn.embedding_dim)
    print(f"milvus       : collection '{store.collection}' "
          f"{'created' if created else 'exists'} rows={store.count()}")

    graph: Neo4jGraphStore | None = None
    if not args.no_graph:
        graph = Neo4jGraphStore(kn.graph)
        try:
            if args.reset or args.reset_graph:
                flag = "--reset" if args.reset else "--reset-graph"
                print(f"{flag:<13s}: 已删除 Neo4j 骨架节点 {graph.reset()}")
                cleared = _clear_graph_marker(progress_path)
                if cleared:
                    print(f"{flag:<13s}: 已清掉 progress.json 中 {cleared} 条图标记"
                          "（保留向量标记，故不会重新嵌入）")
            graph.ensure_constraints()
        except Exception as exc:
            print(f"neo4j        : 不可用 -> {type(exc).__name__}: {exc}")
            print("              图谱构建要求 Neo4j 可达；用 --no-graph 可只入库向量。")
            return 1
        print(f"neo4j        : {kn.graph.uri} constraints ready nodes={graph.counts()}")

    pipeline = IngestionPipeline(
        kn,
        embedding=LocalEmbeddingModel(kn.embedding_model),
        store=store,
        graph=graph,
    )

    stop = threading.Event()
    watcher = threading.Thread(
        target=_watch_progress,
        args=(progress_path, target_rels, len(targets), stop, args.report_every),
        daemon=True,
    )
    watcher.start()
    started = time.monotonic()
    try:
        report = pipeline.run(
            str(corpus),
            limit=args.limit,
            output_dir=str(output_dir),
            batch_size=batch_size,
            persist_every=args.persist_every,
        )
    except KeyboardInterrupt:
        print("\n中断：进度已落盘，可重跑本命令续传。")
        return 130
    finally:
        stop.set()

    errors = report.pop("errors", [])
    print(f"\n耗时         : {(time.monotonic() - started) / 60:.1f} min")
    print(f"入库报告     : {report}")

    print("\n--- 校验 ---")
    rows = store.count()
    lines = _snapshot_lines(output_dir / "chunks.jsonl")
    print(f"Milvus rows        : {rows}")
    print(f"BM25 snapshot rows : {lines}  (chunks.jsonl)")
    if rows != lines:
        print(f"注意               : 两者相差 {rows - lines}"
              f"（含历史遗留行则正常，--reset 后应相等）")
    if graph is not None:
        print(f"Neo4j nodes        : {graph.counts()}")
        graph.close()
    if errors:
        print(f"\n错误样本（共 {len(errors)}）:")
        for line in errors[:10]:
            print(f"  - {line}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
