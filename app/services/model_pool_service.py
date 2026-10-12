from __future__ import annotations

from typing import Any
from uuid import uuid4

from app.agents.base import LLMProfile
from app.core import NotFoundError
from app.core.settings import ModelPricingSettings
from app.models import ModelPoolEntry
from app.repositories import ModelPoolRepository


def entry_to_llm_profile(entry: ModelPoolEntry) -> LLMProfile:
    """Map a pool entry onto the request-level model routing contract.

    Chat (and any future vertical) hands this profile to the orchestrator via
    ``WorkflowRunRequest.llm_profile``, which propagates to every agent in the
    run -- so one selection drives the whole workflow.
    """
    return LLMProfile(
        provider=entry.provider_key,
        model=entry.model,
        temperature=entry.temperature,
        max_tokens=entry.max_tokens,
    )


def entry_to_pricing(entry: ModelPoolEntry) -> ModelPricingSettings:
    return ModelPricingSettings(
        input_per_million=entry.input_per_million,
        output_per_million=entry.output_per_million,
    )


class ModelPoolService:
    """Console-facing model pool: admin CRUD, user-facing selection."""

    # The only column a PATCH may set back to NULL. Every other field is
    # non-nullable, so an explicit ``null`` there is a client mistake and is
    # ignored rather than surfacing as a database error.
    _NULLABLE_FIELDS = frozenset({"max_tokens"})

    def __init__(self, repository: ModelPoolRepository) -> None:
        self._repository = repository

    def list_entries(
        self,
        *,
        enabled_only: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        items, total = self._repository.list_entries(
            enabled_only=enabled_only,
            limit=limit,
            offset=offset,
        )
        return {
            "items": items,
            "total": total,
            "limit": limit,
            "offset": offset,
            "filters": {"enabled_only": enabled_only},
        }

    def get_entry(self, entry_id: str) -> ModelPoolEntry:
        entry = self._repository.get_entry(entry_id)
        if entry is None:
            raise NotFoundError(
                "Model pool entry was not found.",
                details={"entry_id": entry_id},
            )
        return entry

    def create_entry(
        self,
        *,
        name: str,
        provider_key: str,
        model: str,
        description: str = "",
        temperature: float = 0.2,
        max_tokens: int | None = None,
        input_per_million: float = 0.0,
        output_per_million: float = 0.0,
        enabled: bool = True,
        is_default: bool = False,
        sort_order: int = 0,
    ) -> ModelPoolEntry:
        entry = ModelPoolEntry(
            id=str(uuid4()),
            name=name,
            provider_key=provider_key,
            model=model,
            description=description,
            temperature=temperature,
            max_tokens=max_tokens,
            input_per_million=input_per_million,
            output_per_million=output_per_million,
            enabled=enabled,
            is_default=is_default,
            sort_order=sort_order,
        )
        self._repository.add_entry(entry)
        if is_default:
            self._repository.clear_default(except_id=entry.id)
        self._repository.commit()
        return entry

    def update_entry(
        self,
        entry_id: str,
        **changes: Any,
    ) -> ModelPoolEntry:
        """Apply a partial update; callers pass only the fields they mean.

        ``None`` is honoured for nullable columns (clearing a token cap) and
        ignored for the rest, so a stray ``null`` cannot blank a required
        field.
        """
        entry = self.get_entry(entry_id)
        for field, value in changes.items():
            if value is None and field not in self._NULLABLE_FIELDS:
                continue
            setattr(entry, field, value)
        if changes.get("is_default") is True:
            self._repository.clear_default(except_id=entry.id)
        self._repository.commit()
        return entry

    def delete_entry(self, entry_id: str) -> None:
        self.get_entry(entry_id)
        self._repository.delete_entry(entry_id)
        self._repository.commit()

    def resolve_entry(
        self,
        *,
        entry_id: str | None = None,
        name: str | None = None,
    ) -> ModelPoolEntry:
        """Pick the entry a chat request should run with.

        Explicit id/name selection is honoured first; otherwise the pool's
        default entry is used. Missing/disabled selections raise 404 rather
        than silently falling back, so a user never gets a different model
        from the one they picked.
        """
        if entry_id is not None:
            entry = self._repository.get_entry(entry_id)
        elif name is not None:
            entry = self._repository.get_entry_by_name(name)
        else:
            entry = self._repository.get_default_entry()
        if entry is None or not entry.enabled:
            raise NotFoundError(
                "Selected model is not available in the model pool.",
                details={"entry_id": entry_id, "name": name},
            )
        return entry

    def close(self) -> None:
        self._repository.close()
