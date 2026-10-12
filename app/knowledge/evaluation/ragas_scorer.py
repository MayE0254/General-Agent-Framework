"""RAGAS scoring: turn one run of the chain into four metric numbers.

The only module that imports RAGAS.  Everything else in
``app.knowledge.evaluation`` stays judge-agnostic, so testing the orchestration
never requires a judge, and swapping RAGAS for another scorer touches this file
alone.

Three decisions are worth stating, because each is easy to get wrong:

* **Metrics run one at a time, not through ``ragas.evaluate()``.**  A batch
  call returns one column per metric and hides which individual call failed,
  while the report needs a per-sample, per-metric verdict: a judge timeout on
  Context Recall for sample 7 must be recorded as *that*, not as a missing
  number.  Scoring individually is also the only way Context Recall can run on
  just the samples that carry a ground truth.
* **The judge is a different model family from the answerer.**  Scoring an
  answer with the model that wrote it shares the author's blind spots, so the
  judge is configured separately in ``[knowledge.evaluation]`` and reached
  through the provider entry-point that owns its credentials.
* **The judge is a plain ``ChatOpenAI`` pointed at the provider's
  OpenAI-compatible endpoint.**  RAGAS needs a LangChain ``BaseLLM``, and
  ``ragas.llms.llm_factory`` cannot accept an api_key — it reads
  ``OPENAI_API_KEY`` from the environment, which would mean smuggling a Zhipu
  key through another vendor's variable.  Building the client here keeps the
  key coming from ``ZHIPU_API_KEY`` and nowhere else.

Note on the corpus: when retrieval returns nothing, the chain refuses to
answer, and the context-based metrics score that refusal low (there is nothing
to ground it in).  That is the intended signal — an unanswerable question is a
retrieval failure, not a metric defect — so no special case is carved out for
it here.
"""
from __future__ import annotations

import math
import os

from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithoutReference,
    LLMContextRecall,
    ResponseRelevancy,
)
from ragas.metrics.base import SingleTurnMetric
from ragas.run_config import RunConfig

from app.core.settings import KnowledgeEvaluationSettings, LLMProviderSettings
from app.knowledge.embeddings import LocalEmbeddingModel
from app.knowledge.evaluation.runner import (
    EvalSample,
    MetricScore,
    SampleScores,
    ScorerFn,
    sample_metric_plan,
)
from app.knowledge.rag import RagAnswer

# RAGAS emits anonymous usage telemetry on every metric call.  An enterprise
# deployment must not report anything to a third party, and this module is the
# single boundary to RAGAS, so the opt-out belongs here.  setdefault leaves it
# overridable for a machine that explicitly wants the telemetry.  RAGAS reads
# the variable at call time, so setting it at import is sufficient.
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

# GLM-5.3 always reasons: its ``thinking`` block only accepts "enabled", so
# deliberation cannot be switched off.  At the default effort it does not
# finish the NLI prompt RAGAS sends for statement attribution -- measured on a
# 3-statement sample, the verdict call ran past 420s without returning, and
# ``max_tokens=2048`` did not bound it either (the trace is not the completion).
# The identical prompt at "low" answers in 8.3s with valid JSON and zero
# reasoning tokens.  Without it the Faithfulness metric cannot complete on any
# sample, so it is a precondition for the metric existing at all, not a
# quality/speed trade-off.  The value lives in ``[knowledge.evaluation]``
# (``judge_reasoning_effort``) rather than here because the knob is
# provider-specific: it is GLM's cap, not a property of RAGAS scoring, and a
# judge on another endpoint needs its own switch instead (see
# ``judge_extra_body``).


class _LocalBgeEmbeddings(Embeddings):
    """LangChain adapter over the project's local BGE model.

    Response Relevancy scores an answer by generating questions *from* it and
    measuring cosine similarity against the real question.  That is an internal
    similarity computation, so it reuses the retrieval embedder: no extra
    provider, no extra key, and identical vectors on every rerun.  Subclassing
    ``Embeddings`` (rather than duck-typing) is what supplies the async entry
    points RAGAS calls.
    """

    def __init__(self, model: LocalEmbeddingModel) -> None:
        self._model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._model.embed_query(text)


def build_metrics(
    judge: LangchainLLMWrapper,
    embeddings: LangchainEmbeddingsWrapper,
) -> dict[str, SingleTurnMetric]:
    """One metric instance per reported name.

    Keys are the names the report uses, so the mapping from reported name to
    RAGAS class is stated in exactly one place.  In particular
    ``context_precision`` maps to ``LLMContextPrecisionWithoutReference``:
    RAGAS also ships a ``ContextPrecision`` that *requires* a reference answer,
    which would silently make the metric unavailable on every sample that has
    no ground truth — i.e. on most of a reference-free run.
    """
    return {
        "faithfulness": Faithfulness(llm=judge),
        "answer_relevancy": ResponseRelevancy(llm=judge, embeddings=embeddings),
        "context_precision": LLMContextPrecisionWithoutReference(llm=judge),
        "context_recall": LLMContextRecall(llm=judge),
    }


def build_judge(
    evaluation: KnowledgeEvaluationSettings,
    provider: LLMProviderSettings,
) -> LangchainLLMWrapper:
    """Point a LangChain chat client at the judge provider's endpoint."""
    if not provider.api_base or not provider.api_key:
        raise ValueError(
            f"Judge provider '{evaluation.judge_provider}' is missing api_base "
            "or api_key in [llm.providers]; its key normally resolves from the "
            "ZHIPU_API_KEY environment variable."
        )
    # Config names models the way LiteLLM does ("zai/glm-5.3"), matching
    # [llm].model and [llm.pricing]; a raw OpenAI-compatible endpoint wants the
    # bare id ("glm-5.3").  rsplit tolerates a name that carries no prefix.
    model = evaluation.judge_model.rsplit("/", 1)[-1]
    client = ChatOpenAI(
        model=model,
        api_key=provider.api_key,
        base_url=provider.api_base,
        temperature=evaluation.judge_temperature,
        timeout=evaluation.judge_timeout_seconds,
        # Both knobs are optional and provider-specific: omitted (None) rather
        # than sent empty, so a judge on an endpoint that does not know the
        # field never receives it.
        reasoning_effort=evaluation.judge_reasoning_effort or None,
        extra_body=evaluation.judge_extra_body or None,
    )
    return LangchainLLMWrapper(
        client, RunConfig(timeout=evaluation.judge_timeout_seconds)
    )


def build_ragas_scorer(
    evaluation: KnowledgeEvaluationSettings,
    judge_provider: LLMProviderSettings,
    embedding_model: LocalEmbeddingModel,
) -> ScorerFn:
    """Wire the judge, the metrics and the embeddings into one scorer.

    The embedding model is injected rather than built here so the caller keeps
    a single instance across the run (and across the retriever, when the same
    model is used for both).
    """
    metrics = build_metrics(
        build_judge(evaluation, judge_provider),
        LangchainEmbeddingsWrapper(_LocalBgeEmbeddings(embedding_model)),
    )
    timeout_seconds = evaluation.judge_timeout_seconds

    async def _scorer(sample: EvalSample, answer: RagAnswer) -> SampleScores:
        return await score_sample(
            sample, answer, metrics, timeout_seconds=timeout_seconds
        )

    return _scorer


async def score_sample(
    sample: EvalSample,
    answer: RagAnswer,
    metrics: dict[str, SingleTurnMetric],
    *,
    timeout_seconds: int,
) -> SampleScores:
    """Score one answered sample, isolating each metric's failure.

    Every metric is scored in its own ``try``: one judge call failing (timeout,
    rate limit, unparseable JSON) costs that metric on that sample and nothing
    else, which is what the report's ``errored`` count is built from.

    ``answer.contexts`` is passed through untouched — it is by construction the
    exact text the answerer put in the prompt.
    """
    plan = sample_metric_plan(sample, list(metrics))
    row = SingleTurnSample(
        user_input=sample.question,
        response=answer.answer,
        retrieved_contexts=list(answer.contexts),
        # "" is this project's "no ground truth"; None is RAGAS'.  Passing the
        # empty string through would present an absent reference as a present
        # one, which is the distinction the whole metric policy rests on.
        reference=sample.reference or None,
    )
    scores: dict[str, MetricScore] = {}
    for name in plan:
        metric = metrics[name]
        try:
            raw = await metric.single_turn_ascore(row, timeout=timeout_seconds)
        except Exception as exc:  # noqa: BLE001 - a judge failure is data
            scores[name] = MetricScore(
                error=f"{type(exc).__name__}: {exc}".strip()
            )
            continue
        value = None if raw is None else float(raw)
        if value is None or not math.isfinite(value):
            # RAGAS returns NaN where a metric is undefined for the row (e.g.
            # Faithfulness with no extractable statements).  Letting that reach
            # the mean would turn the whole run's average into NaN, and
            # substituting 0 would read as a perfect-failure score; neither is
            # a measurement, so it is reported as a non-result.
            scores[name] = MetricScore(
                error=f"{name} returned no finite score (raw={raw!r})"
            )
            continue
        scores[name] = MetricScore(value=round(value, 4))
    return SampleScores(plan=plan, scores=scores)
