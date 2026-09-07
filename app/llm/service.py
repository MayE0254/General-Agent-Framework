from __future__ import annotations

import os
import time
from collections.abc import Awaitable, Callable
from typing import Any

from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from app.core.settings import LLMSettings
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
AuditRecorderFn = Callable[[LLMCallAudit], Any]


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
        tracer: Any | None = None,
        audit_recorder: AuditRecorderFn | None = None,
    ) -> None:
        self._settings = settings
        # Injectable for tests; defaults to litellm.acompletion.
        self._completion_fn = completion_fn
        self._tracer = tracer
        # Optional sink for per-call audit events (usage + outcome).
        self._audit_recorder = audit_recorder

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
        try:
            api_base, api_key, model = self._resolve_provider(request)

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

            raw = await self._call_with_retry(kwargs)
        except LLMError as exc:
            self._record_audit(
                request,
                status="failed",
                error=str(exc),
                latency_ms=int((time.monotonic() - started) * 1000),
            )
            raise
        except Exception as exc:  # pragma: no cover - defensive boundary
            wrapped = LLMError(f"LLM call failed: {exc}")
            self._record_audit(
                request,
                status="failed",
                error=str(wrapped),
                latency_ms=int((time.monotonic() - started) * 1000),
            )
            raise wrapped from exc
        finally:
            if span is not None:
                span.end()

        latency_ms = int((time.monotonic() - started) * 1000)
        response = self._parse_response(raw, model, latency_ms)
        self._record_audit(
            request,
            status="success",
            response=response,
            latency_ms=latency_ms,
        )
        if self._tracer is not None:
            self._tracer.finish_llm_span(span, response)
        return response

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
        status: str,
        response: LLMResponse | None = None,
        error: str | None = None,
        latency_ms: int = 0,
    ) -> None:
        """Emit one audit event per call when a recorder is wired in."""
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
            model=request.model,
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

    def _parse_response(
        self,
        raw: Any,
        requested_model: str,
        latency_ms: int,
    ) -> LLMResponse:
        try:
            choices = raw.choices
            content = choices[0].message.content or ""
            usage_raw = raw.usage
            usage = LLMUsage(
                prompt_tokens=int(usage_raw.prompt_tokens or 0),
                completion_tokens=int(usage_raw.completion_tokens or 0),
                total_tokens=int(usage_raw.total_tokens or 0),
            )
            model = getattr(raw, "model", None) or requested_model
        except (AttributeError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected LLM response shape: {exc}") from exc

        # Cost lookup: providers may echo a concrete model alias (e.g.
        # "deepseek-v4-flash") that has no pricing entry; fall back to the
        # requested model key so the [llm].pricing table matches.
        cost_model = model if model in self._settings.pricing else requested_model
        return LLMResponse(
            model=str(model),
            content=content,
            usage=usage,
            latency_ms=latency_ms,
            cost_usd=estimate_cost_usd(
                str(cost_model), usage, self._settings.pricing
            ),
            raw={},
        )
