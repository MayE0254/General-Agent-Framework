from __future__ import annotations

from collections.abc import Generator
import os

import pytest
from fastapi.testclient import TestClient

import app.runtime_stack as runtime_stack_module
from app.llm import LLMService

os.environ.setdefault("REDIS_USERNAME", "multiagent")
os.environ.setdefault("REDIS_PASSWORD", "redis-test-password")
os.environ.setdefault("DEEPSEEK_API_KEY", "sk-test-env-key")

from app.main import app


@pytest.fixture
def api_client(monkeypatch) -> Generator[TestClient, None, None]:
    """Provide a TestClient whose lifespan has actually run.

    Entering the ``with`` block is required for FastAPI's lifespan handler
    to execute; without it ``app.state`` stays uninitialized and every
    request fails with 503 Service Unavailable.

    The LLM service is replaced with a fake completion_fn so unit tests stay
    deterministic and never hit the real provider (even when a real api_key
    is configured in the environment). The fake returns non-JSON content,
    which makes planner/router/reviewer fall back to their deterministic
    rule paths.
    """
    async def _fake_completion(kwargs: dict) -> object:
        class _Message:
            content: str = "not-json-fallback-content"

        class _Choice:
            message: object = None

        class _Usage:
            prompt_tokens = 3
            completion_tokens = 2
            total_tokens = 5

        class _Raw:
            model: str = ""
            choices: list = []
            usage: object = None

        raw = _Raw()
        raw.model = kwargs["model"]
        message = _Message()
        raw.choices = [_Choice()]
        raw.choices[0].message = message
        raw.usage = _Usage()
        return raw

    def _fake_create_llm_service(settings, *, tracer=None, audit_recorder=None):
        return LLMService(
            settings,
            completion_fn=_fake_completion,
            tracer=tracer,
            audit_recorder=audit_recorder,
        )

    monkeypatch.setattr(
        runtime_stack_module,
        "create_llm_service",
        _fake_create_llm_service,
    )

    with TestClient(app) as client:
        yield client
