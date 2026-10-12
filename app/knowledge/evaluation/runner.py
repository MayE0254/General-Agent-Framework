"""Offline evaluation of the knowledge chain: dataset -> answers -> scores.

The scored unit is one end-to-end run of the real chain
(:class:`app.knowledge.rag.KnowledgeAnswerer`): a question goes in, and the
answer *and* the exact contexts that produced it come out.  Anything
reassembled after the fact (re-retrieving, re-truncating) would score a system
that never ran.

Metric policy -- three reference-free metrics run on every sample, the fourth
runs only when it can be computed honestly:

===================  ==================  ==============================
metric               needs reference?    when it runs
===================  ==================  ==============================
Faithfulness         no                  always
Response Relevancy   no                  always
Context Precision    no (LLM variant)    always
Context Recall       yes, by definition  only if the sample has ``reference``
===================  ==================  ==============================

A metric that cannot run is reported as ``skipped`` (inapplicable to the
sample) or ``errored`` (the judge call itself failed) and is excluded from the
mean.  It is never padded with 0: a fabricated zero would move the average
exactly as if the system had answered badly, which is the opposite of what an
evaluation is for.

The scoring callable is injected, so this module neither imports RAGAS nor
needs it installed for the orchestration itself to be tested.
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.knowledge.rag import KnowledgeAnswerer, RagAnswer

# Metrics that need a ground-truth answer to mean anything.
REFERENCE_METRICS = ("context_recall",)
REFERENCE_FREE_METRICS = ("faithfulness", "answer_relevancy", "context_precision")
ALL_METRICS = REFERENCE_FREE_METRICS + REFERENCE_METRICS


class EvalSample(BaseModel):
    """One evaluation question, optionally paired with a reference answer."""

    model_config = ConfigDict(extra="forbid")

    id: str = ""
    question: str = Field(..., min_length=1)
    # Human-written ground truth.  Its presence is what switches Context Recall
    # on for this sample; leaving it out is a legitimate choice, not an error.
    reference: str = ""
    tags: list[str] = Field(default_factory=list)


class MetricScore(BaseModel):
    """One metric's outcome: a number, or the reason there is no number.

    Carrying the reason on the metric (rather than in a sibling dict) keeps a
    judge that stopped parsing distinct from a metric that never applied.
    """

    model_config = ConfigDict(extra="forbid")

    value: float | None = None
    error: str = ""


class SampleScores(BaseModel):
    """What the judge was asked for on one sample, and what came back.

    ``plan`` is the set of metrics *structurally* applicable to the sample, so
    a metric missing from ``scores`` can be classified as inapplicable
    (not planned) versus failed (planned but absent).
    """

    model_config = ConfigDict(extra="forbid")

    plan: list[str] = Field(default_factory=list)
    scores: dict[str, MetricScore] = Field(default_factory=dict)


ScorerFn = Callable[[EvalSample, RagAnswer], Awaitable[SampleScores]]


def load_samples(path: str | Path) -> list[EvalSample]:
    """Read a JSONL question set (one object per line, blank lines ignored)."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Evaluation question set not found: {file_path}")
    samples: list[EvalSample] = []
    with file_path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(EvalSample.model_validate(json.loads(line)))
            except Exception as exc:
                raise ValueError(
                    f"{file_path}:{number} is not a valid sample: {exc}"
                ) from exc
    if not samples:
        raise ValueError(f"Evaluation question set is empty: {file_path}")
    return samples


def available_metrics(samples: list[EvalSample]) -> list[str]:
    """Metrics this dataset can support, given which samples carry references."""
    metrics = list(REFERENCE_FREE_METRICS)
    if any(sample.reference.strip() for sample in samples):
        metrics += list(REFERENCE_METRICS)
    return metrics


def sample_metric_plan(sample: EvalSample, metrics: list[str]) -> list[str]:
    """Metrics that apply *structurally* to one sample.

    This is where the "Context Recall needs a reference" rule lives, so it can
    be asserted without a judge (and so the judge cannot quietly answer a
    different question than the one the report claims).
    """
    if sample.reference.strip():
        return list(metrics)
    return [metric for metric in metrics if metric not in REFERENCE_METRICS]


def _aggregate(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Mean per metric over the samples that produced a number.

    ``skipped`` and ``errored`` are counted and surfaced with their reasons, so
    a metric that quietly stopped running shows up as such instead of looking
    like a dataset of perfect scores.
    """
    summary: dict[str, dict[str, Any]] = {}
    for metric in ALL_METRICS:
        values: list[float] = []
        skipped = 0
        errored = 0
        reasons: list[str] = []
        for item in items:
            if item["status"] != "scored":
                # No answer means no evidence for any metric; the reason is the
                # generation failure itself.
                skipped += 1
                reasons.append(item["errors"].get("answer", "answer generation failed"))
                continue
            if metric not in item["plan"]:
                skipped += 1
                reasons.append("sample has no reference answer")
                continue
            outcome = item["scores"].get(metric)
            if outcome is None or outcome["value"] is None:
                errored += 1
                reasons.append(
                    (outcome or {}).get("error") or item["errors"].get(metric)
                    or "judge returned no value"
                )
                continue
            values.append(float(outcome["value"]))
        summary[metric] = {
            "mean": round(sum(values) / len(values), 4) if values else None,
            "scored": len(values),
            "skipped": skipped,
            "errored": errored,
            "reasons": sorted({reason for reason in reasons if reason}),
        }
    return summary


async def run_evaluation(
    samples: list[EvalSample],
    answerer: KnowledgeAnswerer,
    scorer: ScorerFn,
    *,
    concurrency: int = 2,
    on_sample: Callable[[int, int, EvalSample], None] | None = None,
) -> dict[str, Any]:
    """Answer every question, score it, and aggregate.

    Concurrency bounds the in-flight samples.  Within one sample the metric
    calls run sequentially, so a provider's rate-limit window is not hit from
    several directions at once.
    """
    if concurrency < 1:
        raise ValueError(f"concurrency must be >= 1, got {concurrency}")
    semaphore = asyncio.Semaphore(concurrency)
    total = len(samples)
    done = 0
    lock = asyncio.Lock()
    items: list[dict[str, Any] | None] = [None] * total

    async def _one(index: int, sample: EvalSample) -> None:
        nonlocal done
        async with semaphore:
            item = await _evaluate_sample(sample, answerer, scorer)
        async with lock:
            items[index] = item
            done += 1
            if on_sample is not None:
                on_sample(done, total, sample)

    await asyncio.gather(*(_one(index, sample) for index, sample in enumerate(samples)))

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "samples": total,
        "metrics": _aggregate([item for item in items if item is not None]),
        "items": [item for item in items if item is not None],
    }


async def _evaluate_sample(
    sample: EvalSample,
    answerer: KnowledgeAnswerer,
    scorer: ScorerFn,
) -> dict[str, Any]:
    """Produce one report row; a failure is recorded, never raised.

    One unavailable sample must not abort a run that may already have cost a
    hundred judge calls, so the error lands in the row and the caller gets a
    partial-but-honest report.
    """
    try:
        answer = await answerer.answer(sample.question)
    except Exception as exc:  # noqa: BLE001 - any generation failure is data
        return {
            "id": sample.id,
            "question": sample.question,
            "has_reference": bool(sample.reference.strip()),
            "status": "error",
            "plan": [],
            "answer": "",
            "contexts": 0,
            "citations": [],
            "supplementary_citations": [],
            "retrieved": 0,
            "scores": {},
            "errors": {"answer": f"{type(exc).__name__}: {exc}"},
        }

    result = SampleScores()
    errors: dict[str, str] = {}
    try:
        result = await scorer(sample, answer)
    except Exception as exc:  # noqa: BLE001 - judge failure is data too
        errors["scorer"] = f"{type(exc).__name__}: {exc}"

    return {
        "id": sample.id,
        "question": sample.question,
        "has_reference": bool(sample.reference.strip()),
        "status": "scored",
        "plan": result.plan,
        "answer": answer.answer,
        "contexts": len(answer.contexts),
        "citations": [item.chunk_id for item in answer.citations],
        # The blocks that came from the supplementary (quota-rejected) section
        # rather than from the ranked results.  Recorded separately because they
        # are how a row explains itself after the fact: an answer that only holds
        # together when the hints are counted is a retrieval gap, not a win.
        "supplementary_citations": [
            item.chunk_id for item in answer.citations if item.supplementary
        ],
        "retrieved": answer.retrieved,
        "model": answer.model,
        "latency_ms": answer.latency_ms,
        "cost_usd": answer.cost_usd,
        "scores": {
            name: outcome.model_dump() for name, outcome in result.scores.items()
        },
        "errors": errors,
    }


def write_report(path: str | Path, report: dict[str, Any]) -> Path:
    """Persist the report so two runs can be diffed (e.g. in a PR)."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return file_path
