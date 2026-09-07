from __future__ import annotations

from app.core import Settings, get_settings


def get_vectorstore_configs(settings: Settings | None = None) -> dict[str, dict[str, object]]:
    current_settings = settings or get_settings()

    return {
        "pgvector": current_settings.knowledge.pgvector.model_dump(),
        "milvus": current_settings.knowledge.milvus.model_dump(),
    }
