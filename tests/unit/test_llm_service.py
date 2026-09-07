import asyncio
import httpx
import pytest

from app.core.settings import LLMSettings
from app.llm import LLMError, LLMMessage, LLMRequest, LLMRole, LLMResponse, LLMService, LLMUsage


def _settings(api_key: str = "replace_me", **overrides) -> LLMSettings:
    kwargs = dict(
        provider="openai",
        model="gpt-test",
        api_base="https://api.example.com/v1",
        api_key=api_key,
        timeout_seconds=5,
    )
    kwargs.update(overrides)
    return LLMSettings(**kwargs)


async def _fake_completion(kwargs: dict) -> object:
    """Minimal fake standing in for litellm.acompletion."""
    class _Message:
        content = '{"next_agent": "knowledge_agent"}'

    class _Choice:
        message = _Message()

    class _Usage:
        prompt_tokens = 10
        completion_tokens = 5
        total_tokens = 15

    class _Raw:
        model = kwargs["model"]
        choices = [_Choice()]
        usage = _Usage()

    return _Raw()


def test_service_not_configured_with_placeholder_key() -> None:
    service = LLMService(_settings())
    assert service.is_configured is False

    with pytest.raises(LLMError, match="not configured"):
        asyncio.run(
            service.complete(
                LLMRequest(
                    model="gpt-test",
                    messages=[LLMMessage(role=LLMRole.USER, content="hi")],
                )
            )
        )


def test_service_complete_returns_parsed_response() -> None:
    service = LLMService(
        _settings(api_key="sk-real"),
        completion_fn=_fake_completion,
    )
    assert service.is_configured is True

    response = asyncio.run(
        service.complete(
            LLMRequest(
                model="gpt-test",
                messages=[LLMMessage(role=LLMRole.USER, content="route this")],
            )
        )
    )

    assert isinstance(response, LLMResponse)
    assert response.model == "gpt-test"
    assert response.content == '{"next_agent": "knowledge_agent"}'
    assert response.usage == LLMUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    assert response.latency_ms >= 0


def test_service_maps_unexpected_response_to_llm_error() -> None:
    async def _broken(kwargs: dict) -> object:
        class _Empty:
            pass

        return _Empty()

    service = LLMService(
        _settings(api_key="sk-real"),
        completion_fn=_broken,
    )

    with pytest.raises(LLMError, match="Unexpected LLM response shape"):
        asyncio.run(
            service.complete(
                LLMRequest(
                    model="gpt-test",
                    messages=[LLMMessage(role=LLMRole.USER, content="hi")],
                )
            )
        )


def test_service_retries_transient_failure_then_succeeds() -> None:
    calls = {"count": 0}

    async def _flaky(kwargs: dict) -> object:
        calls["count"] += 1
        if calls["count"] < 3:
            raise httpx.ConnectError("temporary network issue", request=object())
        return await _fake_completion(kwargs)

    service = LLMService(
        _settings(api_key="sk-real", max_retries=3, retry_backoff_seconds=0.01),
        completion_fn=_flaky,
    )

    response = asyncio.run(
        service.complete(
            LLMRequest(
                model="gpt-test",
                messages=[LLMMessage(role=LLMRole.USER, content="retry me")],
            )
        )
    )

    assert calls["count"] == 3
    assert response.content == '{"next_agent": "knowledge_agent"}'


def test_service_raises_after_exhausting_retries() -> None:
    calls = {"count": 0}

    async def _always_fail(kwargs: dict) -> object:
        calls["count"] += 1
        raise httpx.ConnectError("persistent network issue", request=object())

    service = LLMService(
        _settings(api_key="sk-real", max_retries=2, retry_backoff_seconds=0.01),
        completion_fn=_always_fail,
    )

    with pytest.raises(LLMError, match="LLM call failed"):
        asyncio.run(
            service.complete(
                LLMRequest(
                    model="gpt-test",
                    messages=[LLMMessage(role=LLMRole.USER, content="boom")],
                )
            )
        )

    assert calls["count"] == 3  # initial attempt + 2 retries


def test_service_does_not_retry_non_transient_error() -> None:
    calls = {"count": 0}

    async def _always_bad_request(kwargs: dict) -> object:
        calls["count"] += 1
        raise ValueError("bad request, not retryable")

    service = LLMService(
        _settings(api_key="sk-real", max_retries=5, retry_backoff_seconds=0.01),
        completion_fn=_always_bad_request,
    )

    with pytest.raises(LLMError, match="LLM call failed"):
        asyncio.run(
            service.complete(
                LLMRequest(
                    model="gpt-test",
                    messages=[LLMMessage(role=LLMRole.USER, content="boom")],
                )
            )
        )

    assert calls["count"] == 1  # non-transient errors are not retried


def test_service_routes_to_named_provider_and_overrides_model() -> None:
    captured: dict = {}

    async def _capturing_completion(kwargs: dict) -> object:
        captured.update(kwargs)
        return await _fake_completion(kwargs)

    service = LLMService(
        _settings(
            api_key="sk-default",
            providers={
                "deepseek": {
                    "api_base": "https://api.deepseek.com",
                    "api_key": "sk-deepseek",
                    "model": "deepseek/deepseek-chat",
                }
            },
        ),
        completion_fn=_capturing_completion,
    )

    response = asyncio.run(
        service.complete(
            LLMRequest(
                model="unused-fallback",
                provider="deepseek",
                messages=[LLMMessage(role=LLMRole.USER, content="route me")],
            )
        )
    )

    assert response.model == "deepseek/deepseek-chat"
    assert captured["model"] == "deepseek/deepseek-chat"
    assert captured["api_key"] == "sk-deepseek"
    assert captured["api_base"] == "https://api.deepseek.com"


def test_service_provider_missing_key_fails_fast() -> None:
    service = LLMService(
        _settings(
            api_key="sk-default",
            providers={
                "placeholder": {
                    "api_base": "https://api.example.com",
                    "api_key": "replace_me",
                }
            },
        ),
        completion_fn=_fake_completion,
    )

    with pytest.raises(LLMError, match="has no api_key"):
        asyncio.run(
            service.complete(
                LLMRequest(
                    model="gpt-test",
                    provider="placeholder",
                    messages=[LLMMessage(role=LLMRole.USER, content="hi")],
                )
            )
        )


def test_service_unknown_provider_fails_fast() -> None:
    service = LLMService(
        _settings(api_key="sk-real"),
        completion_fn=_fake_completion,
    )

    with pytest.raises(LLMError, match="not configured"):
        asyncio.run(
            service.complete(
                LLMRequest(
                    model="gpt-test",
                    provider="nonexistent",
                    messages=[LLMMessage(role=LLMRole.USER, content="hi")],
                )
            )
        )


def test_service_estimates_cost_from_pricing_table() -> None:
    service = LLMService(
        _settings(
            api_key="sk-real",
            pricing={
                "gpt-test": {
                    "input_per_million": 2.0,
                    "output_per_million": 8.0,
                }
            },
        ),
        completion_fn=_fake_completion,
    )

    response = asyncio.run(
        service.complete(
            LLMRequest(
                model="gpt-test",
                messages=[LLMMessage(role=LLMRole.USER, content="cost me")],
            )
        )
    )

    # 10 input tokens * $2/1M + 5 output tokens * $8/1M
    assert response.cost_usd == pytest.approx(0.00002 + 0.00004)


def test_service_unpriced_model_reports_zero_cost() -> None:
    service = LLMService(
        _settings(api_key="sk-real"),
        completion_fn=_fake_completion,
    )

    response = asyncio.run(
        service.complete(
            LLMRequest(
                model="gpt-test",
                messages=[LLMMessage(role=LLMRole.USER, content="free")],
            )
        )
    )

    assert response.cost_usd == 0.0


def test_service_falls_back_to_requested_model_for_cost_lookup() -> None:
    # Providers often echo a concrete alias (e.g. "deepseek-v4-flash") that
    # is not in the pricing table; cost must fall back to the requested key.
    async def _alias_completion(kwargs: dict) -> object:
        class _Message:
            content = "ok"

        class _Choice:
            message = _Message()

        class _Usage:
            prompt_tokens = 1000
            completion_tokens = 0
            total_tokens = 1000

        class _Raw:
            model = "deepseek-v4-flash"
            choices = [_Choice()]
            usage = _Usage()

        return _Raw()

    service = LLMService(
        _settings(
            api_key="sk-real",
            pricing={
                "deepseek/deepseek-chat": {
                    "input_per_million": 0.27,
                    "output_per_million": 1.10,
                }
            },
        ),
        completion_fn=_alias_completion,
    )

    response = asyncio.run(
        service.complete(
            LLMRequest(
                model="deepseek/deepseek-chat",
                messages=[LLMMessage(role=LLMRole.USER, content="cost me")],
            )
        )
    )

    assert response.model == "deepseek-v4-flash"
    # 1000 input tokens * $0.27/1M = $0.00027; alias has no price so the
    # requested model's pricing is used.
    assert response.cost_usd == pytest.approx(0.00027)
