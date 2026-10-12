"""RAGAS evaluation CLI for the fiscal-tax knowledge chain.

Runs the real chain end to end — hybrid retrieval (vector + BM25 → RRF →
dedup/quota → cross-encoder rerank → parent expansion), then grounded
generation — and scores each run with RAGAS.

Run from the repo root::

    .venv\\Scripts\\python scripts\\evaluate_knowledge.py                 # full run
    .venv\\Scripts\\python scripts\\evaluate_knowledge.py --limit 3       # smoke run
    .venv\\Scripts\\python scripts\\evaluate_knowledge.py --report out.json

What is measured (see ``app/knowledge/evaluation/runner.py`` for the policy):
three reference-free metrics on every sample, plus Context Recall on the
samples that carry a ``reference``.  A metric that cannot run is reported as
``skipped`` or ``errored`` with a reason and excluded from the mean — never
padded with 0.

Two cost/scope notes, because the report can otherwise be misread:

* ``cost_usd`` in the report is the **generation leg only**.  The judge calls
  are made by RAGAS through its own LangChain client and are not metered by
  ``LLMService``, so judge spend is not in the numbers.
* A sample whose retrieval returned nothing is answered with a refusal; the
  context-based metrics score that refusal low by design.  A cluster of such
  rows points at a retrieval gap, not at a metric defect.

Requires the corpus to be ingested already (``scripts/ingest_knowledge.py``) and
Neo4j/Milvus reachable, plus the judge provider's key (``ZHIPU_API_KEY`` by
default; see ``judge_provider`` in ``[knowledge.evaluation]``).
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
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


def _resolve_model_ref(model_ref: str) -> str:
    """Resolve a configured model ref, and keep transformers off the hub.

    Model refs in the config are relative to the repo root, but this script may
    be launched from anywhere; a bare ``Path()`` check would then miss the
    local copy and send transformers to huggingface.co, which simply times out
    on an offline machine.  Returning the resolved path also makes the load
    independent of the working directory.
    """
    from app.knowledge.factory import resolve_model_ref

    return resolve_model_ref(model_ref)


def _build_graph_expander(kn, graph_arm: str, chunks):
    """The graph recall leg for this arm, or ``None``.

    ``config`` follows ``[knowledge.graph.expand].enabled``; ``off`` / ``cites``
    / ``both`` override it.  The arm is stated on the command line because the
    comparison runs two arms back to back, and editing the config between runs
    is exactly how a report ends up labelled with the wrong arm.
    """
    from app.knowledge.graph.expand import ClauseTextIndex, GraphExpander
    from app.knowledge.graph.neo4j import Neo4jGraphStore

    if graph_arm == "off" or (
        graph_arm == "config" and not kn.graph.expand.enabled
    ):
        # Printed, not silent: a report has to say which arm produced it, and
        # "the leg was switched off" is an arm.
        print("图扩散腿     : 未接入（off）")
        return None
    if not kn.graph.enabled:
        raise RuntimeError(
            "[knowledge.graph].enabled = false，图不可达，--graph 无法开启"
        )

    expand = kn.graph.expand
    if graph_arm == "cites":
        expand = expand.model_copy(
            update={"enabled": True, "use_cites": True, "use_mentions": False}
        )
    elif graph_arm == "both":
        expand = expand.model_copy(
            update={"enabled": True, "use_cites": True, "use_mentions": True}
        )

    expander = GraphExpander(
        expand, Neo4jGraphStore(kn.graph).query_all, ClauseTextIndex(chunks)
    )
    # Reachability is asserted, not assumed: a leg that never ran and a leg with
    # no effect must not look the same in the report.
    print(f"图扩散腿     : {graph_arm} (cites={expand.use_cites}, "
          f"mentions={expand.use_mentions}, seed_top_n={expand.seed_top_n}, "
          f"per_seed={expand.neighbors_per_seed}, max={expand.max_results}) "
          f"图中条款 {expander.verify()['clauses']}")
    return expander


def _build_retriever(kn, embed, *, graph_arm: str = "config"):
    """Hybrid retriever over the ingested corpus (the chain under test)."""
    from app.knowledge.factory import build_single_hop_retriever
    from app.knowledge.ingest.pipeline import load_chunk_snapshot

    snapshot_path = ROOT / kn.chunk_snapshot
    if not snapshot_path.exists():
        raise FileNotFoundError(
            f"chunk 快照不存在：{snapshot_path}；请先运行 scripts/ingest_knowledge.py"
        )
    chunks = load_chunk_snapshot(str(snapshot_path))
    retriever = build_single_hop_retriever(kn, embedding=embed, chunks=chunks)
    # Only the experiment entry point may attach a graph leg.
    retriever.graph_expander = _build_graph_expander(kn, graph_arm, chunks)
    return retriever


def _print_summary(report: dict) -> None:
    print("\n" + "=" * 74)
    print(f"评测汇总（样本 {report['samples']}，生成于 {report['generated_at']}）")
    print("=" * 74)
    print(f"{'metric':<20}{'mean':>9}{'scored':>9}{'skipped':>9}{'errored':>9}")
    for name, row in report["metrics"].items():
        mean = "-" if row["mean"] is None else f"{row['mean']:.4f}"
        print(f"{name:<20}{mean:>9}{row['scored']:>9}{row['skipped']:>9}"
              f"{row['errored']:>9}")
        for reason in row["reasons"][:3]:
            print(f"    · {reason}")
    scored = [item for item in report["items"] if item["status"] == "scored"]
    if scored:
        cost = sum(item.get("cost_usd", 0.0) for item in scored)
        latency = sum(item.get("latency_ms", 0) for item in scored) / len(scored)
        print(f"\n生成腿：平均延迟 {latency:.0f} ms，累计成本 ${cost:.4f}"
              f"（不含评判调用成本）")
    failed = [item for item in report["items"] if item["status"] == "error"]
    if failed:
        print(f"\n生成失败样本 {len(failed)}（未计分）:")
        for item in failed[:5]:
            print(f"  - {item['id'] or item['question'][:30]}: {item['errors']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--questions", default="data/knowledge/eval/questions.jsonl",
                        help="评测问题集（JSONL，每行一个问题）")
    parser.add_argument("--report", default="data/knowledge/eval/report.json",
                        help="评测报告输出路径（JSON，可 diff）")
    parser.add_argument("--limit", type=int, default=None,
                        help="只评测前 N 个样本（冒烟用；仍会真实调用评判模型）")
    parser.add_argument("--concurrency", type=int, default=None,
                        help="并发样本数（默认取配置 knowledge.evaluation.concurrency）")
    parser.add_argument("--graph", choices=["config", "off", "cites", "both"],
                        default="config",
                        help="图扩散腿的取值臂：config=按 [knowledge.graph.expand]；"
                             "off/cites/both=覆盖配置（同轮对比必须用同一评判模型）")
    args = parser.parse_args(argv)

    for var in ("MILVUS_PASSWORD", "NEO4J_PASSWORD", "ZHIPU_API_KEY",
                "DEEPSEEK_API_KEY"):
        _inject_user_env(var)

    from app.core import load_settings
    from app.knowledge.embeddings import LocalEmbeddingModel
    from app.knowledge.evaluation.ragas_scorer import build_ragas_scorer
    from app.knowledge.evaluation.runner import (
        available_metrics,
        load_samples,
        run_evaluation,
        write_report,
    )
    from app.knowledge.rag import KnowledgeAnswerer
    from app.llm.factory import create_llm_service

    settings = load_settings(ROOT / "configs" / "application.toml")
    kn = settings.knowledge
    ev = kn.evaluation
    if not ev.enabled:
        print("knowledge.evaluation.enabled = false，评测已关闭；"
              "如需运行请先在 configs/application.toml 中开启。")
        return 1

    samples = load_samples(ROOT / args.questions)
    if args.limit:
        samples = samples[: args.limit]
    with_reference = sum(1 for s in samples if s.reference.strip())
    print(f"问题集       : {args.questions}")
    print(f"样本         : {len(samples)}（其中 {with_reference} 个带 reference）")
    print(f"计划指标     : {', '.join(available_metrics(samples))}")

    # One embedder instance for both retrieval and the Response-Relevancy
    # similarity computation: loading BGE twice would double the memory for
    # identical vectors.
    embed = LocalEmbeddingModel(_resolve_model_ref(kn.embedding_model))
    retriever = _build_retriever(kn, embed, graph_arm=args.graph)

    answer_provider = ev.answer_provider or None
    answer_model = ev.answer_model or settings.llm.model
    answerer = KnowledgeAnswerer(
        retriever,
        create_llm_service(settings),
        model=answer_model,
        provider=answer_provider,
        top_k=ev.answer_top_k,
        temperature=ev.answer_temperature,
        max_tokens=ev.answer_max_tokens,
        max_context_chars=ev.max_context_chars,
        extra_body=ev.answer_extra_body or None,
    )
    print(f"被测生成腿   : {answer_model} "
          f"(provider={answer_provider or settings.llm.provider})")

    provider_cfg = settings.llm.providers.get(ev.judge_provider)
    if provider_cfg is None:
        print(f"评判 provider '{ev.judge_provider}' 未在 [llm.providers] 中定义。")
        return 1
    scorer = build_ragas_scorer(ev, provider_cfg, embed)
    print(f"评判模型     : {ev.judge_model} "
          f"(provider={ev.judge_provider}, timeout={ev.judge_timeout_seconds}s)")
    print()

    def _progress(done: int, total: int, sample) -> None:
        label = sample.id or sample.question[:24]
        print(f"[{done}/{total}] {label}", flush=True)

    report = asyncio.run(
        run_evaluation(
            samples,
            answerer,
            scorer,
            concurrency=args.concurrency or ev.concurrency,
            on_sample=_progress,
        )
    )
    path = write_report(ROOT / args.report, report)
    _print_summary(report)
    print(f"\n报告已写入   : {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
