from __future__ import annotations

from app.core import Settings, get_settings
from app.llm.service import LLMService


def create_llm_service(
    settings: Settings | None = None,
    *,
    completion_fn=None,
    tracer=None,
    audit_recorder=None,
) -> LLMService:
    current_settings = settings or get_settings()
    return LLMService(
        current_settings.llm,
        completion_fn=completion_fn,
        tracer=tracer,
        audit_recorder=audit_recorder,
    )
