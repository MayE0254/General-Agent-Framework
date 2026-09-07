from __future__ import annotations

from app.core import Settings, get_settings


def get_litellm_config(settings: Settings | None = None) -> dict[str, object]:
    current_settings = settings or get_settings()
    llm = current_settings.llm

    return {
        "provider": llm.provider,
        "model": llm.model,
        "api_base": llm.api_base,
        "api_key": llm.api_key,
        "timeout_seconds": llm.timeout_seconds,
    }
