"""Grounded question answering: retrieve -> assemble context -> generate.

Retrieval (:mod:`app.knowledge.hybrid`) and generation (:class:`app.llm.LLMService`)
are independent today.  This module is the seam that makes the *whole chain*
addressable as one unit, which two callers need:

- product use: a grounded answer whose claims can be traced back to clauses;
- offline evaluation (``app.knowledge.evaluation``): RAGAS scores a
  ``question -> answer + retrieved contexts`` triple, so the triple has to be
  produced by one run rather than reassembled afterwards.

The single most important invariant here is that :attr:`RagAnswer.contexts` is
*exactly* the text handed to the model in the prompt.  Judging an answer against
contexts that differ from what the model saw (e.g. re-extracting them later, or
passing pre-truncation chunks) would make Faithfulness measure the wrong thing.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from app.knowledge.base import QUOTA_OVERFLOW_KEY, BaseKnowledgeRetriever, KnowledgeHit
from app.llm import LLMMessage, LLMRequest, LLMResponse, LLMRole

if TYPE_CHECKING:
    from app.llm import LLMService

# The sentence the system prompt demands when the fragments cannot answer the
# question.  Defined once and interpolated into the prompt below, because it is
# read back programmatically: it doubles as the sufficiency signal that decides
# whether the supplementary clauses are worth a second call (see
# ``KnowledgeAnswerer.answer``).  Two spellings of it would silently disable
# that retry.
INSUFFICIENT_EVIDENCE_MARKER = "检索到的法规片段不足以回答该问题"

# Headings the prompt refers to *by name*: rule 1 confines the answer to them
# and the supplementary rule attaches the low-reliability caveat to one.  Defined
# once because the prompt and the user message must agree on the spelling --
# a rule pointing at a heading the prompt never renders is a rule the model
# cannot follow.
RETRIEVED_HEADING = "【检索到的法规片段】"

# Heading of the supplementary block in the prompt.  Kept next to the retrieved
# heading because the two must agree: the rules above name this heading, and a
# heading the instructions never mention would leave those clauses unusable
# (rule 1 confines the answer to the retrieved fragments).
SUPPLEMENTARY_HEADING = "【可能对你有帮助的内容】"


def build_system_prompt(*, with_supplementary: bool) -> str:
    """The system prompt for one call, describing the sources that call has.

    The rules name their sources, so the prompt has to match the user message it
    is sent with -- and the mismatch is not hypothetical: naming
    ``SUPPLEMENTARY_HEADING`` in a prompt whose user message carries no such
    section measurably cost Faithfulness on the five evaluation samples that
    receive no supplementary clauses, even though their user messages were
    byte-identical (control run, see the RAGAS evaluation doc §7.6).  So the
    supplementary rules are added only for the retry that actually carries the
    section (see :meth:`KnowledgeAnswerer.answer`).
    """
    sources = (
        f"{RETRIEVED_HEADING}和{SUPPLEMENTARY_HEADING}"
        if with_supplementary
        else RETRIEVED_HEADING
    )
    rules = [
        f"1. 只使用{sources}中的信息作答，不得凭记忆补充、不得推测；",
        "2. 每个结论后用 [编号] 标注依据，编号对应片段序号，可同时标注多个；",
        f"3. 若片段不足以回答，直接说明“{INSUFFICIENT_EVIDENCE_MARKER}”，"
        "并指出缺少哪类依据，不要编造条款号、税率或金额；",
        "4. 使用中文，先给结论再给依据，涉及多个条件时分条列举"
        + ("；" if with_supplementary else "。"),
    ]
    if with_supplementary:
        rules.append(
            f"5. {SUPPLEMENTARY_HEADING}是同一来源因数量限制而未被优先采用的补充条款，"
            f"可靠性低于上面已列出的片段：优先依据{RETRIEVED_HEADING}作答，"
            "只有当其中确实含有回答问题所需的条款时才引用它，引用方式同样是 [编号]。"
        )
    return (
        "你是财税法规智能助手，依据检索到的法规片段回答用户问题。\n作答要求：\n"
        + "\n".join(rules)
    )


# What the refusal check actually matches: the marker minus its tail.  Rule 3
# also asks the model to name what is missing, and the observed answer folded
# that into the demanded sentence -- "检索到的法规片段不足以完整回答该问题"
# (vat-001, 2026-09-24) -- so testing the whole marker as a substring missed it
# and turned the retry off for the one sample that needed it.  The subject and
# the insufficiency verdict are what survive that rewriting; the passage after
# them is what the model restates in its own words.
_REFUSAL_PREFIX = "检索到的法规片段不足"


def _is_refusal(text: str) -> bool:
    """Whether the model said the retrieved fragments were not enough.

    This is the system prompt's own third rule read back, so it costs nothing:
    the model already had to decide whether the evidence suffices, and the
    answerer only has to notice the verdict.  Matching stops at
    ``_REFUSAL_PREFIX`` on purpose: a false positive costs one extra call,
    whereas a false negative silently drops the supplementary clauses from
    exactly the questions that cannot do without them.
    """
    return _REFUSAL_PREFIX in text


# Per-hit cap when a single parent clause is unusually long: a background
# notice can carry a whole schedule table, and one such block must not crowd
# every other source out of the prompt.
_MAX_HIT_CHARS = 2000


class RagCitation(BaseModel):
    """One numbered context block, i.e. the provenance of ``[n]`` in an answer."""

    model_config = ConfigDict(extra="forbid")

    index: int = Field(..., ge=1)
    chunk_id: str
    parent_id: str = ""
    source: str = ""
    title: str = ""
    section: str = ""
    score: float = 0.0
    # True when this block came from the quota-rejected supplementary list, i.e.
    # it is the fallback evidence the diversity guard traded away rather than a
    # primary result.  Recorded so a report (or a human) can tell that an answer
    # leaned on clauses the ranking had deprioritised.
    supplementary: bool = False


class RagAnswer(BaseModel):
    """A grounded answer plus the exact evidence it was generated from."""

    model_config = ConfigDict(extra="forbid")

    question: str
    answer: str
    # Verbatim text of every context block included in the prompt, in order.
    contexts: list[str] = Field(default_factory=list)
    citations: list[RagCitation] = Field(default_factory=list)
    # Hits the retriever returned before the prompt budget dropped any.
    retrieved: int = Field(default=0, ge=0)
    model: str = ""
    provider: str | None = None
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    latency_ms: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    # The generation leg's own verdict that the retrieved fragments could not
    # answer the question (see ``_is_refusal``).  Recorded rather than inferred
    # from the text downstream, because it is also the trigger that decides
    # whether the supplementary clauses were worth a second call.
    insufficient_evidence: bool = False

    @property
    def citation_refs(self) -> list[str]:
        return [item.chunk_id for item in self.citations]


def _section_of(hit: KnowledgeHit) -> str:
    return str(hit.metadata_payload.get("section", "") or "")


def _block(index: int, hit: KnowledgeHit) -> str:
    """Render one numbered context block (header + clause text)."""
    header = " ｜ ".join(
        part for part in (hit.source, hit.title, _section_of(hit)) if part
    )
    body = hit.content.strip()[:_MAX_HIT_CHARS]
    return f"[{index}] {header}\n{body}" if header else f"[{index}]\n{body}"


def build_prompt(
    question: str,
    hits: list[KnowledgeHit],
    *,
    max_context_chars: int,
) -> tuple[str, list[KnowledgeHit]]:
    """Assemble the user message and report which hits actually made it in.

    Hits are taken in ranked order while the character budget allows.  The
    returned hit list is authoritative for both the prompt and the citations,
    so a hit that was dropped for budget is never cited in the answer.

    Hits flagged ``QUOTA_OVERFLOW_KEY`` are rendered in their own section after
    the retrieved clauses, with continuous numbering.  They are the evidence the
    per-document quota rejected, so they belong in the prompt precisely because
    the primary set could not cover the question -- but under a heading that
    says they are hints, never as peers of the ranked results.  They still count
    toward ``included``: the invariant that ``contexts`` is exactly what the
    model saw, and may therefore be cited, has to hold for every block in the
    prompt.
    """
    primary: list[KnowledgeHit] = []
    supplementary: list[KnowledgeHit] = []
    for hit in hits:
        if hit.metadata_payload.get(QUOTA_OVERFLOW_KEY):
            supplementary.append(hit)
        else:
            primary.append(hit)

    blocks: list[str] = []
    extra_blocks: list[str] = []
    included: list[KnowledgeHit] = []
    used = 0

    def _take(hit: KnowledgeHit) -> bool:
        """Append the next numbered block unless the budget is exhausted.

        Soft cap: the first block always gets in, otherwise a budget smaller
        than one clause would turn a grounded answer into a refusal.
        """
        nonlocal used
        block = _block(len(included) + 1, hit)
        if included and used + len(block) > max_context_chars:
            return False
        (extra_blocks if hit.metadata_payload.get(QUOTA_OVERFLOW_KEY) else blocks).append(block)
        included.append(hit)
        used += len(block)
        return True

    for hit in primary:
        if not _take(hit):
            break
    for hit in supplementary:
        if not _take(hit):
            break

    if blocks:
        context = f"{RETRIEVED_HEADING}\n" + "\n\n".join(blocks)
    else:
        context = f"{RETRIEVED_HEADING}\n（没有检索到相关片段）"
    if extra_blocks:
        context += f"\n\n{SUPPLEMENTARY_HEADING}\n" + "\n\n".join(extra_blocks)
    return f"{context}\n\n【用户问题】\n{question}", included


class IncompleteAnswerError(RuntimeError):
    """The provider's completion is not a usable answer.

    A reasoning model whose thinking chain eats ``max_tokens`` fails here in two
    shapes, both *after* a successful, billed call, so neither is proof that
    generation was skipped -- only that no complete answer came out:

    - no text at all (:class:`EmptyAnswerError`);
    - a half-sentence cut off at the cap (:class:`TruncatedAnswerError`, i.e.
      ``finish_reason="length"``).

    Raised rather than returned because the two consumers treat an incomplete
    answer differently, and the wrong one is silent:

    - the evaluation runner records it as a failed sample (excluded from the
      means), whereas scoring an empty answer gives Faithfulness ``NaN``
      (reported as errored) and Answer Relevancy ``0.0`` -- a fabricated "the
      system answered badly" data point -- and scoring a truncated one grades a
      sentence the model never had room to finish;
    - a product caller has no complete answer to show and must be able to tell
      that apart from a refusal.
    """


class EmptyAnswerError(IncompleteAnswerError):
    """The completion contained no text at all."""


class TruncatedAnswerError(IncompleteAnswerError):
    """The completion stopped at ``max_tokens`` mid-answer."""


class KnowledgeAnswerer:
    """Answer questions from the knowledge base with citations.

    Retrieval and generation are both injected so the same object serves the
    CLI, the console and the RAGAS evaluation run; ``model`` / ``provider`` are
    per-request overrides resolved by :class:`LLMService` (a named provider
    contributes endpoint + credentials, never a model).
    """

    def __init__(
        self,
        retriever: BaseKnowledgeRetriever,
        llm_service: LLMService,
        *,
        model: str,
        provider: str | None = None,
        top_k: int = 5,
        temperature: float = 0.0,
        # Matches [knowledge.evaluation].answer_max_tokens: the default has to be
        # large enough for a reasoning model's thinking chain, otherwise a caller
        # that omits the argument reproduces the empty-answer failure.
        max_tokens: int = 3000,
        timeout_seconds: int = 120,
        max_context_chars: int = 12000,
        extra_body: dict | None = None,
        trace_id: str | None = None,
        request_id: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm_service = llm_service
        self.model = model
        self.provider = provider
        self.top_k = top_k
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.max_context_chars = max_context_chars
        self.extra_body = extra_body
        self.trace_id = trace_id
        self.request_id = request_id
        self.session_id = session_id
        self.user_id = user_id

    async def _complete(
        self, user_message: str, *, with_supplementary: bool
    ) -> LLMResponse:
        """One generation call, with the responses that prove it unusable rejected."""
        request = LLMRequest(
            model=self.model,
            provider=self.provider,
            messages=[
                LLMMessage(
                    role=LLMRole.SYSTEM,
                    content=build_system_prompt(with_supplementary=with_supplementary),
                ),
                LLMMessage(role=LLMRole.USER, content=user_message),
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            extra_body=self.extra_body,
            timeout_seconds=self.timeout_seconds,
            trace_id=self.trace_id,
            request_id=self.request_id,
            session_id=self.session_id,
            user_id=self.user_id,
            agent_id="knowledge_qa",
        )
        response = await self.llm_service.complete(request)
        answer_text = response.content.strip()
        if not answer_text:
            raise EmptyAnswerError(
                f"{response.model} returned no answer text after a billed call "
                f"({response.usage.completion_tokens} completion tokens, "
                f"prompt={response.usage.prompt_tokens})"
            )
        # Non-empty is not the same as complete: when the thinking chain leaves
        # only a few dozen tokens for the answer, the call succeeds, is billed
        # and returns a sentence cut mid-clause.  Observed on vat-005 (101
        # chars, the round's slowest and costliest call, finish_reason=length).
        if response.finish_reason == "length":
            raise TruncatedAnswerError(
                f"{response.model} stopped at max_tokens={self.max_tokens} with "
                f"{len(answer_text)} chars of answer "
                f"({response.usage.completion_tokens} completion tokens billed)"
            )
        return response

    async def answer(
        self, question: str, *, retrieval_query: str | None = None
    ) -> RagAnswer:
        """Retrieve, then generate a grounded answer for one question.

        The supplementary clauses are held back from the first call and added
        only if that call cannot answer: they exist for the question shape the
        per-document quota structurally cannot serve (the answer spread across
        more clauses of one statute than the quota admits), and offering them
        unconditionally would put every answer on clauses the ranking had
        already deprioritised -- 10 of 15 evaluation samples received them while
        only 2 needed them, which cost precision and 55% more latency.

        The trigger is the model's own verdict, read back from the refusal the
        system prompt's third rule requires.  No threshold is introduced: the
        question "was the evidence enough?" is one the generation leg has
        already had to answer, and a rerank-score cutoff would be a new magic
        number with no calibration.

        Each call is sent the system prompt describing the sources it carries
        (:func:`build_system_prompt`): the first pass has no supplementary
        section, so it is told about none.
        """
        hits = await self.retriever.retrieve(retrieval_query or question, top_k=self.top_k)
        # Preserve the latest wording and supply only the resolved referent,
        # never replay prior assistant answers as authoritative evidence.
        prompt_question = question
        if retrieval_query and retrieval_query != question:
            prompt_question = f"{question}\n（根据对话补全的检索问题：{retrieval_query}）"
        primary = [h for h in hits if not h.metadata_payload.get(QUOTA_OVERFLOW_KEY)]
        supplementary = [
            h for h in hits if h.metadata_payload.get(QUOTA_OVERFLOW_KEY)
        ]
        base = RagAnswer(
            question=question,
            answer="",
            retrieved=len(hits),
            model=self.model,
            provider=self.provider,
        )

        user_message, included = build_prompt(
            prompt_question, primary, max_context_chars=self.max_context_chars
        )
        if not included:
            # Nothing retrieved: answering would be pure hallucination, and the
            # judge would score the refusal against an empty context anyway.
            return base.model_copy(
                update={
                    "answer": f"{INSUFFICIENT_EVIDENCE_MARKER}。",
                    "insufficient_evidence": True,
                }
            )

        response = await self._complete(user_message, with_supplementary=False)
        cost_usd = response.cost_usd
        latency_ms = response.latency_ms
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        if supplementary and _is_refusal(response.content):
            # Both passes are billed and both took time, so the answer reports
            # their sum: attributing only the last call would understate what a
            # question the retrieval could not serve actually costs.
            user_message, included = build_prompt(
                prompt_question,
                primary + supplementary,
                max_context_chars=self.max_context_chars,
            )
            response = await self._complete(user_message, with_supplementary=True)
            cost_usd += response.cost_usd
            latency_ms += response.latency_ms
            prompt_tokens += response.usage.prompt_tokens
            completion_tokens += response.usage.completion_tokens

        answer_text = response.content.strip()
        citations = [
            RagCitation(
                index=index,
                chunk_id=hit.id,
                parent_id=hit.parent_id,
                source=hit.source,
                title=hit.title,
                section=_section_of(hit),
                score=hit.score,
                supplementary=bool(hit.metadata_payload.get(QUOTA_OVERFLOW_KEY)),
            )
            for index, hit in enumerate(included, start=1)
        ]
        return base.model_copy(
            update={
                "answer": answer_text,
                "contexts": [hit.content.strip()[:_MAX_HIT_CHARS] for hit in included],
                "citations": citations,
                "model": response.model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "latency_ms": latency_ms,
                "cost_usd": cost_usd,
                "insufficient_evidence": _is_refusal(answer_text),
            }
        )
