from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from app.core.settings import LLMSettings, ModelPricingSettings
from app.llm.pricing import estimate_cost_usd
from app.llm.schemas import (
    LLMCallAudit,
    LLMError,
    LLMRequest,
    LLMResponse,
    LLMUsage,
)

# LiteLLM fetches a remote model cost map on first import, which blocks
# startup when offline. Local fallback keeps behavior deterministic.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "true")

CompletionFn = Callable[[dict[str, Any]], Awaitable[Any]]
# Returns the async iterator produced by a streaming completion call; kept
# separate from CompletionFn so tests can fake one without the other.
StreamFn = Callable[[dict[str, Any]], Awaitable[AsyncIterator[Any]]]
AuditRecorderFn = Callable[[LLMCallAudit], Any]
# Resolves per-model pricing that lives outside the static settings table
# (the console model pool stores its own unit prices per entry).
PricingResolverFn = Callable[[str], ModelPricingSettings | None]


def _is_retryable(exc: BaseException) -> bool:
    """Transient failures are retried; everything else propagates."""
    if isinstance(exc, LLMError):
        return False

    try:
        import httpx

        if isinstance(exc, httpx.TransportError):
            return True
    except ModuleNotFoundError:  # pragma: no cover - optional dependency
        pass

    try:
        from litellm.exceptions import (
            APIConnectionError,
            InternalServerError,
            RateLimitError,
            Timeout,
        )

        if isinstance(exc, (APIConnectionError, RateLimitError, Timeout, InternalServerError)):
            return True
    except ModuleNotFoundError:  # pragma: no cover - optional dependency
        pass

    return False


class LLMService:
    """Unified async LLM entry point backed by LiteLLM.

    The service owns request execution, timeout, retry/backoff, error
    mapping, and (optionally) trace emission. Model/provider policy lives
    in the caller, so agents and orchestrators can pass their own
    ``LLMRequest`` without coupling to LiteLLM directly.
    """

    def __init__(
        self,
        settings: LLMSettings,
        *,
        completion_fn: CompletionFn | None = None,
        stream_fn: StreamFn | None = None,
        tracer: Any | None = None,
        audit_recorder: AuditRecorderFn | None = None,
        pricing_resolver: PricingResolverFn | None = None,
    ) -> None:
        self._settings = settings
        # Injectable for tests; defaults to litellm.acompletion.
        self._completion_fn = completion_fn
        self._stream_fn = stream_fn
        self._tracer = tracer
        # Optional sink for per-call audit events (usage + outcome).
        self._audit_recorder = audit_recorder
        # Optional external pricing source (console model pool). When it has
        # no entry for a model, [llm].pricing stays authoritative.
        self._pricing_resolver = pricing_resolver

    @property
    def is_configured(self) -> bool:
        """True when an API key is present and is not the placeholder."""
        key = self._settings.api_key
        return bool(key) and key != "replace_me"

    @property
    def model(self) -> str:
        return self._settings.model

    @property
    def settings(self) -> LLMSettings:
        return self._settings

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Execute a single completion request with retry/backoff.

        ``request.provider`` selects a named provider from ``[llm].providers``
        (api_base/api_key/model override); when unset the default ``[llm]``
        block is used. A provider whose api_key is empty fails fast rather
        than silently falling back to another backend.
        """
        started = time.monotonic()
        span = None
        if self._tracer is not None:
            span = self._tracer.start_llm_span(request)
        # Falls back to the requested model when provider resolution itself
        # fails, so even a misconfigured call is auditable.
        resolved_model = request.model
        try:
            api_base, api_key, resolved_model = self._resolve_provider(request)
            kwargs = self._build_kwargs(
                request,
                api_base=api_base,
                api_key=api_key,
                model=resolved_model,
            )
            raw = await self._call_with_retry(kwargs)
        except LLMError as exc:
            self._record_audit(
                request,
                model=resolved_model,
                status="failed",
                error=str(exc),
                latency_ms=int((time.monotonic() - started) * 1000),
            )
            raise
        except Exception as exc:  # pragma: no cover - defensive boundary
            wrapped = LLMError(f"LLM call failed: {exc}")
            self._record_audit(
                request,
                model=resolved_model,
                status="failed",
                error=str(wrapped),
                latency_ms=int((time.monotonic() - started) * 1000),
            )
            raise wrapped from exc
        finally:
            if span is not None:
                span.end()

        latency_ms = int((time.monotonic() - started) * 1000)
        response = self._parse_response(raw, resolved_model, latency_ms)
        # Audit the resolved model, not response.model: providers echo their
        # own aliases (deepseek answers "deepseek-flash" for a request for
        # "deepseek/deepseek-flash"), which would split one model across two
        # names in the per-model usage report. The streaming path has no echo
        # at all, so both paths must agree on the resolved string.
        self._record_audit(
            request,
            model=resolved_model,
            status="success",
            response=response,
            latency_ms=latency_ms,
        )
        if self._tracer is not None:
            self._tracer.finish_llm_span(span, response)
        return response

    async def stream_complete(
        self,
        request: LLMRequest,
        *,
        on_complete: Callable[[LLMResponse], None] | None = None,
    ) -> AsyncIterator[str]:
        """Stream a completion as incremental text deltas.

        Audited exactly like :meth:`complete` (one row per call, carrying the
        aggregated usage and the resolved model), but deliberately *not*
        retried: the caller has already received partial output by the time a
        failure surfaces, so replaying the call would duplicate it.

        ``on_complete`` fires once with the aggregated response, giving the
        caller the token usage that only the stream tail knows about.
        """
        started = time.monotonic()
        resolved_model = request.model
        pieces: list[str] = []
        usage = LLMUsage()
        finish_reason = ""
        try:
            api_base, api_key, resolved_model = self._resolve_provider(request)
            kwargs = self._build_kwargs(
                request,
                api_base=api_base,
                api_key=api_key,
                model=resolved_model,
            )
            kwargs["stream"] = True
            # Ask providers that support it to append a usage-only trailer,
            # so streaming calls are billed like non-streaming ones.
            kwargs["stream_options"] = {"include_usage": True}
            stream = await self._call_stream(kwargs)
            async for chunk in stream:
                delta = self._parse_stream_chunk(chunk)
                if delta:
                    pieces.append(delta)
                    yield delta
                chunk_usage = self._parse_stream_usage(chunk)
                if chunk_usage is not None:
                    usage = chunk_usage
                chunk_finish = self._parse_stream_finish_reason(chunk)
                if chunk_finish:
                    finish_reason = chunk_finish
        except Exception as exc:
            wrapped = (
                exc
                if isinstance(exc, LLMError)
                else LLMError(f"LLM stream failed: {exc}")
            )
            self._record_audit(
                request,
                model=resolved_model,
                status="failed",
                error=str(wrapped),
                latency_ms=int((time.monotonic() - started) * 1000),
            )
            raise wrapped from exc

        latency_ms = int((time.monotonic() - started) * 1000)
        response = self._parse_stream_result(
            requested_model=resolved_model,
            content="".join(pieces),
            finish_reason=finish_reason,
            usage=usage,
            latency_ms=latency_ms,
        )
        self._record_audit(
            request,
            model=response.model,
            status="success",
            response=response,
            latency_ms=latency_ms,
        )
        if on_complete is not None:
            on_complete(response)

    def _build_kwargs(
        self,
        request: LLMRequest,
        *,
        api_base: str,
        api_key: str,
        model: str,
    ) -> dict[str, Any]:
        """Assemble the LiteLLM payload for an already-resolved provider."""
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                item.model_dump(exclude_none=True) for item in request.messages
            ],
            "temperature": request.temperature,
            "timeout": request.timeout_seconds,
            "api_base": api_base,
            "api_key": api_key,
        }
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        if request.extra_body:
            kwargs["extra_body"] = request.extra_body
        return kwargs

    def _resolve_provider(
        self,
        request: LLMRequest,
    ) -> tuple[str, str, str]:
        """Resolve (api_base, api_key, model) for the request's provider.

        Raises LLMError when a named provider is missing, has no api_key, or
        is not configured at all (strict mode: no silent fallback).
        """
        model = request.model
        if not request.provider:
            if not self.is_configured:
                raise LLMError(
                    "LLMService is not configured: provide a real api_key in "
                    "configs/application.toml ([llm]) before calling complete()."
                )
            return self._settings.api_base, self._settings.api_key, model

        provider = self._settings.providers.get(request.provider)
        if provider is None:
            raise LLMError(
                f"LLM provider '{request.provider}' is not configured in "
                "[llm].providers."
            )
        if not provider.api_key or provider.api_key == "replace_me":
            raise LLMError(
                f"LLM provider '{request.provider}' has no api_key configured."
            )
        api_base = provider.api_base or self._settings.api_base
        model = provider.model or model
        return api_base, provider.api_key, model

    def _record_audit(
        self,
        request: LLMRequest,
        *,
        model: str,
        status: str,
        response: LLMResponse | None = None,
        error: str | None = None,
        latency_ms: int = 0,
    ) -> None:
        """Emit one audit event per call when a recorder is wired in.

        ``model`` is the *resolved* model (provider override applied), not the
        requested one, so per-model usage stats reflect what actually ran.
        """
        if self._audit_recorder is None:
            return
        usage = response.usage if response is not None else LLMUsage()
        cost_usd = response.cost_usd if response is not None else 0.0
        audit = LLMCallAudit(
            request_id=request.request_id,
            session_id=request.session_id,
            workflow_id=request.workflow_id,
            trace_id=request.trace_id,
            agent_id=request.agent_id,
            user_id=request.user_id,
            model=model,
            status=status,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            messages=[
                item.model_dump(exclude_none=True) for item in request.messages
            ],
            error=error,
        )
        self._audit_recorder(audit)

    async def _call_with_retry(self, kwargs: dict[str, Any]) -> Any:
        attempts = self._settings.max_retries + 1
        if attempts <= 1:
            return await self._call_litellm(kwargs)

        retrying = AsyncRetrying(
            stop=stop_after_attempt(attempts),
            wait=wait_exponential(
                multiplier=self._settings.retry_backoff_seconds,
                max=self._settings.retry_max_seconds,
            ),
            retry=retry_if_exception(_is_retryable),
            reraise=True,
        )
        async for attempt in retrying:
            with attempt:
                return await self._call_litellm(kwargs)
        raise LLMError("LLM call exhausted retries.")  # pragma: no cover

    async def _call_litellm(self, kwargs: dict[str, Any]) -> Any:
        if self._completion_fn is not None:
            return await self._completion_fn(kwargs)

        try:
            import litellm
        except ModuleNotFoundError as exc:  # pragma: no cover - env-dependent
            raise LLMError(
                "LiteLLM is not installed in the current environment."
            ) from exc
        return await litellm.acompletion(**kwargs)

    async def _call_stream(self, kwargs: dict[str, Any]) -> AsyncIterator[Any]:
        """Open a streaming completion, preferring the injected test double."""
        if self._stream_fn is not None:
            return await self._stream_fn(kwargs)

        try:
            import litellm
        except ModuleNotFoundError as exc:  # pragma: no cover - env-dependent
            raise LLMError(
                "LiteLLM is not installed in the current environment."
            ) from exc
        return await litellm.acompletion(**kwargs)

    @staticmethod
    def _parse_stream_chunk(chunk: Any) -> str:
        """Extract incremental text from one stream chunk ("" when none)."""
        choices = getattr(chunk, "choices", None)
        if not choices:
            # Usage-only trailer chunk, or a keep-alive with no choices.
            return ""
        delta = getattr(choices[0], "delta", None)
        if delta is None:
            return ""
        return getattr(delta, "content", None) or ""

    @staticmethod
    def _parse_stream_finish_reason(chunk: Any) -> str:
        """Extract the stop reason when a chunk carries one.

        Only the last choice-bearing chunk reports it, so callers accumulate
        rather than overwrite with the empty string of a keep-alive chunk.
        """
        choices = getattr(chunk, "choices", None)
        if not choices:
            return ""
        return getattr(choices[0], "finish_reason", None) or ""

    @staticmethod
    def _parse_stream_usage(chunk: Any) -> LLMUsage | None:
        """Extract usage when the provider attaches it to a chunk."""
        usage_raw = getattr(chunk, "usage", None)
        if usage_raw is None:
            return None
        try:
            return LLMUsage(
                prompt_tokens=int(usage_raw.prompt_tokens or 0),
                completion_tokens=int(usage_raw.completion_tokens or 0),
                total_tokens=int(usage_raw.total_tokens or 0),
            )
        except (AttributeError, TypeError, ValueError):
            return None

    def _parse_stream_result(
        self,
        *,
        requested_model: str,
        content: str,
        finish_reason: str,
        usage: LLMUsage,
        latency_ms: int,
    ) -> LLMResponse:
        """Aggregate streamed deltas into the same shape `complete` returns."""
        return self._finalize_response(
            model=requested_model,
            requested_model=requested_model,
            content=content,
            finish_reason=finish_reason,
            usage=usage,
            latency_ms=latency_ms,
        )

    def _parse_response(
        self,
        raw: Any,
        requested_model: str,
        latency_ms: int,
    ) -> LLMResponse:
        try:
            choices = raw.choices
            content = choices[0].message.content or ""
            # "length" is the only proof that an answer was cut off by
            # max_tokens rather than finished; without it a half-sentence and a
            # complete one are the same string.
            finish_reason = getattr(choices[0], "finish_reason", None) or ""
            usage_raw = raw.usage
            usage = LLMUsage(
                prompt_tokens=int(usage_raw.prompt_tokens or 0),
                completion_tokens=int(usage_raw.completion_tokens or 0),
                total_tokens=int(usage_raw.total_tokens or 0),
            )
            model = getattr(raw, "model", None) or requested_model
        except (AttributeError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected LLM response shape: {exc}") from exc

        return self._finalize_response(
            model=str(model),
            requested_model=requested_model,
            content=content,
            finish_reason=finish_reason,
            usage=usage,
            latency_ms=latency_ms,
        )

    def _resolve_pricing(self, model: str) -> ModelPricingSettings | None:
        """Per-model unit prices: console model pool first, then config.

        The model pool (managed in the console) is allowed to override the
        static ``[llm].pricing`` table so operators can price newly added
        models without a redeploy.
        """
        if self._pricing_resolver is not None:
            override = self._pricing_resolver(model)
            if override is not None:
                return override
        return self._settings.pricing.get(model)

    def _finalize_response(
        self,
        *,
        model: str,
        requested_model: str,
        content: str,
        finish_reason: str = "",
        usage: LLMUsage,
        latency_ms: int,
    ) -> LLMResponse:
        # Cost lookup: providers may echo a concrete model alias (e.g.
        # "deepseek-v4-flash") that has no pricing entry; fall back to the
        # requested model key so the pricing tables match.
        cost_model = (
            model
            if self._resolve_pricing(model) is not None
            else requested_model
        )
        price = self._resolve_pricing(cost_model)
        cost_usd = (
            0.0
            if price is None
            else estimate_cost_usd(cost_model, usage, {cost_model: price})
        )
        return LLMResponse(
            model=model,
            content=content,
            finish_reason=finish_reason,
            usage=usage,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            raw={},
        )
