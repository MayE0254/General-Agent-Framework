"""Offline evaluation of the knowledge chain (datasets, runners, reports).

The orchestration in :mod:`app.knowledge.evaluation.runner` is deliberately
judge-agnostic: it produces the answers and aggregates the numbers, and the
scoring callable is injected.  :mod:`app.knowledge.evaluation.ragas_scorer` is
the one module that imports RAGAS, and it is intentionally *not* re-exported
here so that importing this package never drags the judge stack in.
"""
from app.knowledge.evaluation.runner import (
    ALL_METRICS,
    REFERENCE_FREE_METRICS,
    REFERENCE_METRICS,
    EvalSample,
    MetricScore,
    SampleScores,
    ScorerFn,
    available_metrics,
    load_samples,
    run_evaluation,
    sample_metric_plan,
    write_report,
)

__all__ = [
    "ALL_METRICS",
    "REFERENCE_FREE_METRICS",
    "REFERENCE_METRICS",
    "EvalSample",
    "MetricScore",
    "SampleScores",
    "ScorerFn",
    "available_metrics",
    "load_samples",
    "run_evaluation",
    "sample_metric_plan",
    "write_report",
]
