import logging

import pytest

from app.core.settings import ObservabilitySettings
from app.observability import (
    LangfuseTracer,
    NullTracer,
    configure_logging,
    create_tracer,
    get_logger,
)


def test_create_tracer_returns_null_when_not_configured() -> None:
    tracer = create_tracer(ObservabilitySettings())
    assert isinstance(tracer, NullTracer)


def test_create_tracer_returns_langfuse_when_configured() -> None:
    tracer = create_tracer(
        ObservabilitySettings(
            service_name="multiagent-service",
            langfuse_host="https://cloud.langfuse.com",
            langfuse_public_key="pk-xxx",
            langfuse_secret_key="sk-xxx",
        )
    )
    assert isinstance(tracer, LangfuseTracer)


def test_null_tracer_is_noop() -> None:
    tracer = NullTracer()
    span = tracer.start_llm_span(object())
    assert span is None
    tracer.finish_llm_span(None, object())
    tracer.capture_error(RuntimeError("boom"))


def test_configure_logging_keeps_get_logger_usable() -> None:
    configure_logging(debug=True)
    logger = get_logger("tests.observability")
    assert logger is not None
    logger.info("observability smoke", key="value")

    # Standard logging remains usable after configure_logging.
    std_logger = logging.getLogger("tests.stdlib")
    std_logger.info("stdlib smoke")


def test_langfuse_tracer_disabled_when_incomplete() -> None:
    tracer = LangfuseTracer(
        ObservabilitySettings(
            langfuse_host="https://cloud.langfuse.com",
            langfuse_public_key="pk-xxx",
            langfuse_secret_key="",  # missing secret -> disabled
        )
    )
    assert tracer.enabled is False
