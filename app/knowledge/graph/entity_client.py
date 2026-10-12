"""LLM orchestration for the semantic entity layer, including its fallback leg.

``entities.py`` owns the vocabulary and the parsing; this module owns the calls.
It exists separately because the two have different failure modes: a parsing
bug is deterministic and unit-testable, while a provider running out of quota
is an operational event that must degrade *visibly* instead of aborting a run
that has already spent hours of API budget.

The degradation contract (explicit, never silent):

* the primary leg is ``[knowledge.graph.entity]``'s provider/model;
* when it fails, the same request is retried once on the configured fallback
  leg -- which is where the DeepSeek "thinking" switch belongs, since that flag
  is provider-specific and must never be sent to the primary;
* a *quota* failure (or N consecutive failures of any kind) additionally opens
  a circuit for the rest of the run: an exhausted quota is not a per-call
  accident, and retrying it once per document would waste thousands of calls.
  The switch is recorded with its reason and the document count at which it
  happened;
* if both legs fail the document raises, so the build script records it and
  re-runs it later.  Nothing is dropped quietly, and every mention carries the
  model id that produced it into the graph.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum

from app.core.settings import GraphEntitySettings
from app.knowledge.graph.entities import (
    ClauseInput,
    EntityExtraction,
    EntityParseError,
    assign_refs,
    build_messages,
    is_truncated,
    parse_extraction,
    split_batch,
)
from app.llm.schemas import LLMRequest, LLMResponse
from app.llm.service import LLMService

# Textual hints, checked case-insensitively against the provider's message.
# Deliberately no numeric provider codes: a wrong code list is worse than none,
# because a misclassified quota error would keep the primary leg for thousands
# of documents.  The generic consecutive-failure circuit below covers that case.
_QUOTA_HINTS = (
    "quota", "insufficient", "balance", "arrears", "billing",
    "余额", "额度", "欠费", "账户",
)
_RATE_LIMIT_HINTS = (
    "rate limit", "too many requests", "429", "concurrent", "并发", "限流",
)


class FailureKind(StrEnum):
    QUOTA = "quota"
    RATE_LIMIT = "rate_limit"
    OTHER = "other"


def classify_failure(exc: BaseException) -> FailureKind:
    """Bucket a provider failure by what it means for the rest of the run."""
    message = str(exc).lower()
    if any(hint in message for hint in _QUOTA_HINTS):
        return FailureKind.QUOTA
    if any(hint in message for hint in _RATE_LIMIT_HINTS):
        return FailureKind.RATE_LIMIT
    return FailureKind.OTHER


class EntityExtractionError(RuntimeError):
    """A call, batch or document could not be extracted on any configured leg."""


@dataclass
class ExtractStats:
    """Run-level counters, printed by the build script and compared per run."""

    docs: int = 0
    batches: int = 0
    calls: int = 0
    primary_calls: int = 0
    fallback_calls: int = 0
    truncated: int = 0
    parse_failures: int = 0
    split_retries: int = 0
    failed_docs: int = 0
    mentions: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    circuit_reason: str = ""
    circuit_docs: int = 0
    discards: dict[str, int] = field(default_factory=dict)

    def merge_discards(self, counters: dict[str, int]) -> None:
        """Accumulate per-document validation counters into the run totals."""
        for key, value in counters.items():
            self.discards[key] = self.discards.get(key, 0) + value


@dataclass(frozen=True)
class _Leg:
    """One provider leg: where a request goes and which model answers."""

    provider: str
    model: str
    extra_body: dict | None = None


class EntityExtractor:
    """Extract clause->entity mentions for whole documents."""

    def __init__(
        self,
        service: LLMService,
        settings: GraphEntitySettings,
        *,
        circuit_after_consecutive_failures: int = 3,
    ) -> None:
        self._service = service
        self._settings = settings
        self._circuit_threshold = max(1, circuit_after_consecutive_failures)
        self._consecutive_failures = 0
        self._circuit_open = False
        self._primary = _Leg(
            provider=settings.provider,
            model=settings.model,
            extra_body=settings.extra_body or None,
        )
        self._fallback = (
            _Leg(
                provider=settings.fallback_provider,
                model=settings.fallback_model,
                extra_body=settings.fallback_extra_body or None,
            )
            if settings.fallback_provider and settings.fallback_model
            else None
        )
        self.stats = ExtractStats()

    @property
    def circuit_open(self) -> bool:
        """True once the primary leg was abandoned for the rest of the run."""
        return self._circuit_open

    @property
    def has_fallback(self) -> bool:
        return self._fallback is not None

    async def extract_document(
        self, doc_id: str, title: str, clauses: list[ClauseInput]
    ) -> EntityExtraction:
        """Extract one document, one call per batch of clauses.

        A document that cannot be extracted raises ``EntityExtractionError``;
        the batch loop never swallows a failure into an empty result.
        """
        batches = split_batch(
            clauses,
            max_clauses=self._settings.max_clauses_per_call,
            max_chars=self._settings.max_chars_per_call,
        )
        merged = EntityExtraction(doc_id=doc_id)
        self.stats.docs += 1
        for batch in batches:
            part = await self._extract_batch(doc_id, title, batch, depth=0)
            merged.mentions.extend(part.mentions)
            for key, value in part.stats.items():
                merged.stats[key] = merged.stats.get(key, 0) + value
        merged.stats["batches"] = len(batches)
        self.stats.mentions += len(merged.mentions)
        return merged

    async def _extract_batch(
        self, doc_id: str, title: str, batch: list[ClauseInput], *, depth: int
    ) -> EntityExtraction:
        ref_to_clause, rows = assign_refs(batch)
        messages = build_messages(title, rows)
        response, leg = await self._call(messages)
        truncated = is_truncated(response.finish_reason, response.content)
        if truncated:
            self.stats.truncated += 1
        try:
            if truncated:
                # An unclosed JSON object cannot be parsed, so route it through
                # the same split-and-retry recovery as a parse failure.
                raise EntityParseError("response truncated by max_tokens")
            extraction = parse_extraction(response.content, ref_to_clause, doc_id)
        except EntityParseError as exc:
            self.stats.parse_failures += 1
            if len(batch) > 1 and depth < self._settings.max_split_depth:
                return await self._split_and_retry(doc_id, title, batch, depth=depth)
            raise EntityExtractionError(
                f"document {doc_id}: {len(batch)} clause(s) unparseable "
                f"after {depth} split(s): {exc}"
            ) from exc
        # Provenance: which model id produced these mentions, so the graph can
        # be audited afterwards when a fallback leg was in use.
        extraction.mentions = [
            replace(m, extractor=leg.model) for m in extraction.mentions
        ]
        self.stats.batches += 1
        self.stats.merge_discards(extraction.stats)
        return extraction

    async def _split_and_retry(
        self, doc_id: str, title: str, batch: list[ClauseInput], *, depth: int
    ) -> EntityExtraction:
        """Re-send a failed batch as two halves; the halves are independent."""
        self.stats.split_retries += 1
        middle = len(batch) // 2
        merged = EntityExtraction(doc_id=doc_id)
        for half in (batch[:middle], batch[middle:]):
            part = await self._extract_batch(doc_id, title, half, depth=depth + 1)
            merged.mentions.extend(part.mentions)
            for key, value in part.stats.items():
                merged.stats[key] = merged.stats.get(key, 0) + value
        return merged

    async def _call(self, messages) -> tuple[LLMResponse, _Leg]:
        """Run one call, degrading to the fallback leg per the contract above."""
        if self._circuit_open:
            return await self._run(self._fallback, messages), self._fallback
        try:
            response = await self._run(self._primary, messages)
        except EntityExtractionError as exc:
            self._consecutive_failures += 1
            if self._fallback is None:
                raise
            self._note_primary_failure(exc)
            return await self._run(self._fallback, messages), self._fallback
        self._consecutive_failures = 0
        return response, self._primary

    def _note_primary_failure(self, exc: EntityExtractionError) -> None:
        if classify_failure(exc) is FailureKind.QUOTA:
            self._open_circuit(f"quota exhausted: {exc}")
        elif self._consecutive_failures >= self._circuit_threshold:
            self._open_circuit(
                f"{self._consecutive_failures} consecutive primary failures: {exc}"
            )

    def _open_circuit(self, reason: str) -> None:
        if self._circuit_open:
            return
        self._circuit_open = True
        self.stats.circuit_reason = reason
        self.stats.circuit_docs = self.stats.docs

    async def _run(self, leg: _Leg | None, messages) -> LLMResponse:
        """One LLM call on one leg; failures surface as EntityExtractionError."""
        if leg is None:
            raise EntityExtractionError("no fallback leg is configured")
        request = LLMRequest(
            model=leg.model,
            messages=messages,
            temperature=self._settings.temperature,
            max_tokens=self._settings.max_tokens,
            timeout_seconds=self._settings.timeout_seconds,
            provider=leg.provider or None,
            extra_body=leg.extra_body,
        )
        self.stats.calls += 1
        if leg is self._fallback:
            self.stats.fallback_calls += 1
        else:
            self.stats.primary_calls += 1
        try:
            response = await self._service.complete(request)
        except Exception as exc:
            leg_name = leg.provider or "default"
            raise EntityExtractionError(f"{leg_name} failed: {exc}") from exc
        self.stats.prompt_tokens += response.usage.prompt_tokens
        self.stats.completion_tokens += response.usage.completion_tokens
        self.stats.cost_usd += response.cost_usd
        return response
