"""Unit tests for offline knowledge evaluation: dataset -> aggregation -> judge.

Two halves, matching the module split:

* the orchestration (:mod:`app.knowledge.evaluation.runner`) is judge-agnostic,
  so it is tested with a stub scorer and never touches a model;
* the RAGAS adapter (:mod:`app.knowledge.evaluation.ragas_scorer`) is tested
  for the things it alone decides -- which RAGAS class backs each reported
  metric name, which endpoint/model/key the judge client is built from, and how
  a non-result (NaN, exception) is recorded instead of being scored.

No test here makes a network call: the judge client is constructed but never
invoked, and every metric is a stub except where the class identity matters.
"""

import asyncio
import json
import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    ContextPrecision,
    Faithfulness,
    LLMContextPrecisionWithoutReference,
    LLMContextRecall,
    ResponseRelevancy,
)

from app.core.settings import KnowledgeEvaluationSettings, LLMProviderSettings
from app.knowledge.evaluation import (
    ALL_METRICS,
    EvalSample,
    MetricScore,
    SampleScores,
    available_metrics,
    load_samples,
    run_evaluation,
    sample_metric_plan,
    write_report,
)
from app.knowledge.evaluation.ragas_scorer import (
    _LocalBgeEmbeddings,
    build_judge,
    build_metrics,
    score_sample,
)
from app.knowledge.rag import RagAnswer, RagCitation

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
@pytest.fixture()
def workdir() -> Iterator[Path]:
    """A writable scratch directory.

    pytest's own ``tmp_path`` cannot be used on this machine: its base
    directory (``%TEMP%\\pytest-of-EDY``) exists with an ACL that denies
    writing, so the fixture fails at setup.  ``mkdtemp`` creates a directory
    directly under the temp root, which is writable.
    """
    path = Path(tempfile.mkdtemp(prefix="knowledge-eval-test-"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def test_load_samples_reads_jsonl_and_ignores_blank_lines(workdir) -> None:
    path = workdir / "questions.jsonl"
    path.write_text(
        '{"id": "a", "question": "问题一？", "reference": "参考答案"}\n'
        "\n"
        '{"id": "b", "question": "问题二？", "tags": ["无参考"]}\n',
        encoding="utf-8",
    )

    samples = load_samples(path)

    assert [s.id for s in samples] == ["a", "b"]
    assert samples[0].reference == "参考答案"
    assert samples[1].reference == ""
    assert samples[1].tags == ["无参考"]


def test_load_samples_reports_the_offending_line(workdir) -> None:
    path = workdir / "questions.jsonl"
    path.write_text('{"question": "好的？"}\n{"id": "broken"}\n', encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        load_samples(path)

    assert "questions.jsonl:2" in str(excinfo.value)


def test_load_samples_rejects_a_missing_or_empty_file(workdir) -> None:
    with pytest.raises(FileNotFoundError):
        load_samples(workdir / "absent.jsonl")

    empty = workdir / "empty.jsonl"
    empty.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_samples(empty)


# ---------------------------------------------------------------------------
# Metric policy: three run always, Context Recall only with a ground truth
# ---------------------------------------------------------------------------
def test_available_metrics_adds_recall_only_when_a_reference_exists() -> None:
    free = [EvalSample(question="问题？")]
    mixed = free + [EvalSample(question="问题？", reference="参考答案")]

    assert available_metrics(free) == ["faithfulness", "answer_relevancy",
                                       "context_precision"]
    assert "context_recall" not in available_metrics(free)
    assert "context_recall" in available_metrics(mixed)


def test_sample_metric_plan_drops_recall_for_a_reference_free_sample() -> None:
    # A whitespace-only reference is not a ground truth: it must not switch the
    # metric on, or the judge would be asked to compare against nothing.
    blank = EvalSample(question="问题？", reference="   ")
    grounded = EvalSample(question="问题？", reference="参考答案")
    metrics = list(ALL_METRICS)

    assert sample_metric_plan(blank, metrics) == [
        metric for metric in metrics if metric != "context_recall"
    ]
    assert sample_metric_plan(grounded, metrics) == metrics


# ---------------------------------------------------------------------------
# Orchestration with a stub judge
# ---------------------------------------------------------------------------
class _StubAnswerer:
    """Stands in for KnowledgeAnswerer; can be told to fail on one question."""

    def __init__(self, fail_on: str = "", citations: list | None = None) -> None:
        self.fail_on = fail_on
        self.citations = citations or []
        self.questions: list[str] = []

    async def answer(self, question: str) -> RagAnswer:
        self.questions.append(question)
        if question == self.fail_on:
            raise RuntimeError("generation exploded")
        return RagAnswer(
            question=question,
            answer=f"回答：{question}",
            contexts=[f"片段：{question}"],
            citations=self.citations,
            retrieved=1,
            model="stub-model",
            latency_ms=7,
            cost_usd=0.002,
        )


def _stub_scorer(by_id: dict[str, SampleScores]):
    async def _scorer(sample: EvalSample, answer: RagAnswer) -> SampleScores:
        return by_id[sample.id]

    return _scorer


def _scores(plan: list[str], **values: object) -> SampleScores:
    """Build a SampleScores from metric name -> value or error string."""
    scores: dict[str, MetricScore] = {}
    for name in plan:
        outcome = values.get(name)
        if isinstance(outcome, str):
            scores[name] = MetricScore(error=outcome)
        else:
            scores[name] = MetricScore(value=outcome)  # type: ignore[arg-type]
    return SampleScores(plan=plan, scores=scores)


def test_run_evaluation_scores_every_metric_and_aggregates_the_means() -> None:
    samples = [
        EvalSample(id="a", question="问题一？", reference="参考答案"),
        EvalSample(id="b", question="问题二？"),
    ]
    free = ["faithfulness", "answer_relevancy", "context_precision"]
    scorer = _stub_scorer(
        {
            "a": _scores(free + ["context_recall"], faithfulness=0.8,
                         answer_relevancy=0.9, context_precision=0.5,
                         context_recall=0.4),
            "b": _scores(free, faithfulness=0.6, answer_relevancy=0.7,
                         context_precision=0.3),
        }
    )

    report = asyncio.run(
        run_evaluation(samples, _StubAnswerer(), scorer, concurrency=2)
    )

    assert report["samples"] == 2
    metrics = report["metrics"]
    assert metrics["faithfulness"]["mean"] == pytest.approx(0.7)
    assert metrics["faithfulness"]["scored"] == 2
    # Recall runs on one sample only, and the other is *skipped with a reason*
    # rather than counted as a zero.
    assert metrics["context_recall"]["mean"] == pytest.approx(0.4)
    assert metrics["context_recall"]["scored"] == 1
    assert metrics["context_recall"]["skipped"] == 1
    assert metrics["context_recall"]["errored"] == 0
    assert "sample has no reference answer" in metrics["context_recall"]["reasons"]

    rows = {item["id"]: item for item in report["items"]}
    assert rows["a"]["status"] == "scored"
    assert rows["a"]["has_reference"] is True
    assert rows["b"]["has_reference"] is False
    assert rows["b"]["answer"] == "回答：问题二？"
    assert rows["b"]["contexts"] == 1


def test_run_evaluation_records_supplementary_citations_separately() -> None:
    # A row whose answer only holds together once the quota-rejected hints are
    # counted describes a retrieval gap; keeping the two citation lists apart is
    # what makes that readable after the fact.
    citations = [
        RagCitation(index=1, chunk_id="c1"),
        RagCitation(index=2, chunk_id="c2", supplementary=True),
    ]
    samples = [
        EvalSample(id="a", question="问题一？", reference="参考答案"),
    ]
    scorer = _stub_scorer({"a": _scores(["faithfulness"], faithfulness=0.5)})

    report = asyncio.run(
        run_evaluation(samples, _StubAnswerer(citations=citations), scorer)
    )

    row = report["items"][0]
    assert row["citations"] == ["c1", "c2"]
    assert row["supplementary_citations"] == ["c2"]


def test_run_evaluation_excludes_a_failed_metric_instead_of_zero_filling() -> None:
    samples = [EvalSample(id=str(n), question=f"问题{n}？",
                          reference=f"参考答案{n}") for n in (1, 2)]
    plan = list(ALL_METRICS)
    scorer = _stub_scorer(
        {
            # Sample 1: the judge blew up on Context Recall only.
            "1": _scores(plan, faithfulness=0.9, answer_relevancy=1.0,
                         context_precision=1.0,
                         context_recall="TimeoutError: judge timed out"),
            "2": _scores(plan, faithfulness=0.7, answer_relevancy=0.8,
                         context_precision=0.6, context_recall=0.5),
        }
    )

    report = asyncio.run(run_evaluation(samples, _StubAnswerer(), scorer))

    recall = report["metrics"]["context_recall"]
    # 0.5, not (0.5 + 0)/2: a failed judge call is not a bad score.
    assert recall["mean"] == pytest.approx(0.5)
    assert (recall["scored"], recall["errored"], recall["skipped"]) == (1, 1, 0)
    assert "TimeoutError: judge timed out" in recall["reasons"]
    assert report["metrics"]["faithfulness"]["errored"] == 0


def test_run_evaluation_records_a_generation_failure_without_aborting() -> None:
    samples = [EvalSample(id="ok", question="问题一？"),
               EvalSample(id="bad", question="问题二？")]
    scorer = _stub_scorer(
        {"ok": _scores(["faithfulness"], faithfulness=0.5), "bad": SampleScores()}
    )

    report = asyncio.run(run_evaluation(samples, _StubAnswerer(fail_on="问题二？"), scorer))

    rows = {item["id"]: item for item in report["items"]}
    assert rows["bad"]["status"] == "error"
    assert rows["bad"]["plan"] == []
    assert "generation exploded" in rows["bad"]["errors"]["answer"]
    # The healthy sample is still scored, and the failure is attributed to it.
    assert rows["ok"]["status"] == "scored"
    assert report["metrics"]["faithfulness"]["scored"] == 1
    assert report["metrics"]["faithfulness"]["skipped"] == 1


def test_run_evaluation_rejects_invalid_concurrency() -> None:
    with pytest.raises(ValueError, match="concurrency"):
        asyncio.run(
            run_evaluation([], _StubAnswerer(), _stub_scorer({}), concurrency=0)
        )


def test_write_report_persists_readable_utf8_json(workdir) -> None:
    path = workdir / "nested" / "report.json"

    written = write_report(path, {"metrics": {"faithfulness": {"mean": 0.75}},
                                  "question": "增值税税率？"})

    assert written == path
    raw = path.read_text(encoding="utf-8")
    assert "增值税税率？" in raw, "reports must stay readable, not \\u-escaped"
    assert json.loads(raw)["metrics"]["faithfulness"]["mean"] == 0.75


# ---------------------------------------------------------------------------
# Judge construction: model id, endpoint, key, telemetry opt-out
# ---------------------------------------------------------------------------
def _provider(**overrides) -> LLMProviderSettings:
    fields = {
        "api_base": "https://api.z.ai/api/paas/v4",
        "api_key": "zhipu-secret",
    }
    fields.update(overrides)
    return LLMProviderSettings(**fields)


def test_ragas_telemetry_is_opted_out_at_import() -> None:
    # Module import happened above; RAGAS reads this at call time.
    assert os.environ["RAGAS_DO_NOT_TRACK"] == "true"


def test_build_judge_strips_the_litellm_prefix_and_uses_provider_credentials() -> None:
    evaluation = KnowledgeEvaluationSettings(
        judge_model="zai/glm-5.3", judge_timeout_seconds=120
    )

    wrapper = build_judge(evaluation, _provider())

    client = wrapper.langchain_llm
    # Config names models LiteLLM-style; an OpenAI-compatible endpoint wants the
    # bare id, so the prefix must not leak into the request.
    assert client.model_name == "glm-5.3"
    assert client.openai_api_base == "https://api.z.ai/api/paas/v4"
    assert client.openai_api_key.get_secret_value() == "zhipu-secret"
    assert wrapper.run_config.timeout == 120
    # Regression guard for the reasoning-effort precondition: at the default
    # effort GLM-5.3's NLI verdict call never returned (measured >420s), which
    # silently turned Faithfulness into an all-``errored`` metric.
    assert client.reasoning_effort == "low"


def test_build_judge_accepts_an_unprefixed_model_name() -> None:
    evaluation = KnowledgeEvaluationSettings(judge_model="glm-5.3")

    assert build_judge(evaluation, _provider()).langchain_llm.model_name == "glm-5.3"


def test_build_judge_omits_the_deliberation_knob_when_configured_empty() -> None:
    # Swapping the judge to another vendor must not smuggle GLM's
    # ``reasoning_effort`` onto that vendor's endpoint: an empty value means
    # "send nothing", not "send an empty string".
    evaluation = KnowledgeEvaluationSettings(
        judge_model="deepseek/deepseek-v4-flash",
        judge_reasoning_effort="",
        judge_extra_body={"thinking": {"type": "disabled"}},
    )

    client = build_judge(evaluation, _provider()).langchain_llm

    assert client.model_name == "deepseek-v4-flash"
    assert client.reasoning_effort is None
    assert client.extra_body == {"thinking": {"type": "disabled"}}


def test_build_judge_sends_no_extra_body_by_default() -> None:
    # The default judge (GLM) needs no extra_body; passing {} through would put
    # an empty JSON object on every metric call.
    client = build_judge(KnowledgeEvaluationSettings(), _provider()).langchain_llm

    assert client.extra_body is None


def test_build_judge_requires_resolved_provider_credentials() -> None:
    # An unresolved ${ZHIPU_API_KEY} arrives here as an empty string; failing
    # loudly beats sending an empty bearer token to the judge endpoint.
    for provider in (_provider(api_key=""), _provider(api_base="")):
        with pytest.raises(ValueError, match="ZHIPU_API_KEY"):
            build_judge(KnowledgeEvaluationSettings(), provider)


def _judge_and_embeddings() -> tuple[object, LangchainEmbeddingsWrapper]:
    class _StubModel(Embeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[0.1, 0.2] for _ in texts]

        def embed_query(self, text: str) -> list[float]:
            return [0.1, 0.2]

    return (
        build_judge(KnowledgeEvaluationSettings(), _provider()),
        LangchainEmbeddingsWrapper(_StubModel()),
    )


def test_build_metrics_reports_one_metric_per_name_using_the_free_variants() -> None:
    judge, embeddings = _judge_and_embeddings()

    metrics = build_metrics(judge, embeddings)  # type: ignore[arg-type]

    # The keys are the names the report prints, and they must cover the whole
    # policy list, or a metric would silently never be scored.
    assert set(metrics) == set(ALL_METRICS)
    assert isinstance(metrics["faithfulness"], Faithfulness)
    assert isinstance(metrics["answer_relevancy"], ResponseRelevancy)
    assert isinstance(metrics["context_precision"], LLMContextPrecisionWithoutReference)
    # The reference-requiring variant would make every reference-free sample
    # error out; the two classes are easy to confuse, so the choice is pinned.
    assert not isinstance(metrics["context_precision"], ContextPrecision)
    assert isinstance(metrics["context_recall"], LLMContextRecall)


def test_local_bge_embeddings_forward_to_the_project_model() -> None:
    calls: list[tuple[str, object]] = []

    class _StubModel:
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            calls.append(("documents", list(texts)))
            return [[1.0, 0.0] for _ in texts]

        def embed_query(self, text: str) -> list[float]:
            calls.append(("query", text))
            return [0.0, 1.0]

    adapter = _LocalBgeEmbeddings(_StubModel())  # type: ignore[arg-type]

    assert adapter.embed_documents(["甲", "乙"]) == [[1.0, 0.0], [1.0, 0.0]]
    assert adapter.embed_query("丙") == [0.0, 1.0]
    assert calls == [("documents", ["甲", "乙"]), ("query", "丙")]

    # Response Relevancy reaches the embedder through RAGAS's wrapper (and its
    # async path), so that hop is asserted too.
    wrapper = LangchainEmbeddingsWrapper(adapter)
    assert asyncio.run(wrapper.aembed_query("丁")) == [0.0, 1.0]


# ---------------------------------------------------------------------------
# score_sample: what is handed to the judge, and what a non-result looks like
# ---------------------------------------------------------------------------
class _StubMetric:
    """Records every judged row; returns (or raises) a canned outcome."""

    def __init__(self, outcome: object) -> None:
        self.outcome = outcome
        self.rows: list[object] = []
        self.timeouts: list[int | None] = []

    async def single_turn_ascore(self, row, timeout=None) -> object:
        self.rows.append(row)
        self.timeouts.append(timeout)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def _answer() -> RagAnswer:
    return RagAnswer(
        question="税率是多少？",
        answer="结论：13%。[1]",
        contexts=["第一条 税率为百分之十三。", "第二条 其他。"],
    )


def test_score_sample_judges_the_exact_prompt_contexts_and_rounds_scores() -> None:
    sample = EvalSample(id="a", question="税率是多少？", reference="百分之十三")
    metrics = {
        name: _StubMetric(0.123456)
        for name in ("faithfulness", "answer_relevancy", "context_precision",
                     "context_recall")
    }

    result = asyncio.run(
        score_sample(sample, _answer(), metrics, timeout_seconds=90)
    )

    assert result.plan == list(ALL_METRICS)
    assert result.scores["faithfulness"] == MetricScore(value=0.1235)
    for name, metric in metrics.items():
        assert metric.timeouts == [90], f"{name} must run under the judge timeout"
        row = metric.rows[0]
        assert row.user_input == sample.question
        assert row.response == "结论：13%。[1]"
        # The invariant behind Faithfulness: the judged contexts are verbatim
        # the ones the answerer put in the prompt, in order and unedited.
        assert row.retrieved_contexts == ["第一条 税率为百分之十三。", "第二条 其他。"]
        assert row.reference == "百分之十三"


def test_score_sample_never_judges_recall_without_a_reference() -> None:
    sample = EvalSample(id="a", question="税率是多少？")
    metrics = {
        name: _StubMetric(0.5)
        for name in ("faithfulness", "context_precision", "context_recall")
    }

    result = asyncio.run(
        score_sample(sample, _answer(), metrics, timeout_seconds=30)
    )

    assert result.plan == ["faithfulness", "context_precision"]
    assert "context_recall" not in result.scores
    assert metrics["context_recall"].rows == [], "no ground truth, no judge call"
    # ...and the row must not carry an empty reference that RAGAS could read as
    # a real one.
    assert metrics["faithfulness"].rows[0].reference is None


def test_score_sample_records_a_non_finite_score_as_a_non_result() -> None:
    # RAGAS returns NaN where a metric is undefined (e.g. Faithfulness with no
    # extractable statements) and None when a metric bails out.
    sample = EvalSample(id="a", question="税率是多少？")
    metrics = {
        "faithfulness": _StubMetric(float("nan")),
        "answer_relevancy": _StubMetric(None),
    }

    result = asyncio.run(
        score_sample(sample, _answer(), metrics, timeout_seconds=30)
    )

    assert result.scores["faithfulness"].value is None
    assert "no finite score" in result.scores["faithfulness"].error
    assert result.scores["answer_relevancy"].value is None
    assert "no finite score" in result.scores["answer_relevancy"].error


def test_score_sample_isolates_one_failing_metric_from_the_rest() -> None:
    sample = EvalSample(id="a", question="税率是多少？")
    metrics = {
        "faithfulness": _StubMetric(0.8),
        "answer_relevancy": _StubMetric(TimeoutError("judge timed out")),
        "context_precision": _StubMetric(ValueError("bad json")),
    }

    result = asyncio.run(
        score_sample(sample, _answer(), metrics, timeout_seconds=30)
    )

    # One judge failure costs that metric, not the sample.
    assert result.scores["faithfulness"].value == 0.8
    assert result.scores["answer_relevancy"].error == "TimeoutError: judge timed out"
    assert result.scores["context_precision"].value is None
    assert result.scores["context_precision"].error.startswith("ValueError: bad json")
