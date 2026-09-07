from __future__ import annotations

import json
import logging
from enum import StrEnum
from typing import TYPE_CHECKING
from uuid import uuid4

from app.core.settings import MemorySettings
from app.llm import LLMMessage, LLMRequest, LLMRole
from app.models import MemoryEntryRecord, SessionMessageRecord
from app.repositories import MemoryRepository

if TYPE_CHECKING:
    from app.llm import LLMService

logger = logging.getLogger(__name__)


class MemoryScope(StrEnum):
    """Durable memory scopes, ordered by recall priority.

    Recall always walks this order: a user-scoped memory wins over the
    same-importance project memory, which wins over global defaults.
    """

    USER = "user"
    PROJECT = "project"
    GLOBAL = "global"


class MemorySource(StrEnum):
    """Where a memory entry came from."""

    USER = "user"
    LLM = "llm"
    MANUAL = "manual"


# Global memories are tenant-wide; they all share this synthetic scope key.
GLOBAL_SCOPE_KEY = "global"

# Upper bound of candidates scanned when looking for a duplicate statement
# inside one (memory_type, scope, scope_key) partition.
MERGE_CANDIDATE_LIMIT = 200


class MemoryService:
    """Session context and durable long-term memory (user habits, project facts)."""

    def __init__(
        self,
        repository: MemoryRepository,
        *,
        llm_service: LLMService | None = None,
        settings: MemorySettings | None = None,
    ) -> None:
        self._repository = repository
        self._llm_service = llm_service
        self._settings = settings or MemorySettings()

    # ---- session context ----

    def load_session_context(
        self,
        *,
        session_id: str,
        user_id: str | None,
        project_key: str,
    ) -> dict[str, object]:
        """Build the context injected into the next workflow turn."""
        history = self._repository.list_session_messages(
            session_id, limit=self._settings.session_history_limit
        )
        recalled = self._recall_for(user_id=user_id, project_key=project_key)
        return {
            "session_id": session_id,
            "session_history": [
                {
                    "role": item.role,
                    "content": item.content,
                }
                for item in history
            ],
            "recalled_memories": [
                {
                    "memory_type": item.memory_type,
                    "scope": item.scope,
                    "scope_key": item.scope_key,
                    "content": item.content,
                }
                for item in recalled
            ],
        }

    def append_session_messages(
        self,
        *,
        session_id: str,
        request_id: str,
        turns: list[dict[str, object]],
    ) -> None:
        records = [
            SessionMessageRecord(
                id=str(uuid4()),
                session_id=session_id,
                request_id=request_id,
                role=str(turn["role"]),
                content=str(turn["content"]),
                step_index=int(turn.get("step_index", 0)),
                metadata_payload=dict(turn.get("metadata_payload") or {}),
            )
            for turn in turns
        ]
        if records:
            self._repository.add_session_messages(records)
            self._repository.commit()

    def list_session_messages(
        self, session_id: str, *, limit: int = 50
    ) -> list[SessionMessageRecord]:
        return self._repository.list_session_messages(session_id, limit=limit)

    def count_session_messages(self, session_id: str) -> int:
        return self._repository.count_session_messages(session_id)

    # ---- durable memory ----

    def add_memory(
        self,
        *,
        memory_type: str,
        content: str,
        scope: str,
        scope_key: str,
        source: str = "manual",
        importance: int = 3,
        tags: list[str] | None = None,
        metadata_payload: dict[str, object] | None = None,
    ) -> MemoryEntryRecord:
        try:
            scope = MemoryScope(scope).value
        except ValueError as exc:
            raise ValueError(
                f"Invalid memory scope '{scope}'; "
                f"expected one of {[item.value for item in MemoryScope]}."
            ) from exc
        try:
            source = MemorySource(source).value
        except ValueError as exc:
            raise ValueError(
                f"Invalid memory source '{source}'; "
                f"expected one of {[item.value for item in MemorySource]}."
            ) from exc
        if scope == MemoryScope.GLOBAL:
            # Global memories are tenant-wide: the scope key is synthetic.
            scope_key = GLOBAL_SCOPE_KEY
        elif not scope_key:
            raise ValueError("scope_key is required for user/project memories.")
        new_tags = tags or []
        new_meta = metadata_payload or {}
        if self._settings.merge_duplicates:
            duplicate = self._find_duplicate(
                memory_type=memory_type,
                scope=scope,
                scope_key=scope_key,
                content=content,
            )
            if duplicate is not None:
                existing, similarity = duplicate
                return self._merge_into(
                    existing,
                    content=content,
                    importance=importance,
                    tags=new_tags,
                    metadata_payload=new_meta,
                    similarity=similarity,
                )
        record = MemoryEntryRecord(
            id=str(uuid4()),
            memory_type=memory_type,
            scope=scope,
            scope_key=scope_key,
            content=content,
            source=source,
            importance=importance,
            tags=new_tags,
            metadata_payload=new_meta,
        )
        self._repository.add_memory(record)
        self._repository.commit()
        return record

    def _find_duplicate(
        self,
        *,
        memory_type: str,
        scope: str,
        scope_key: str,
        content: str,
    ) -> tuple[MemoryEntryRecord, float] | None:
        """Find an existing entry matching the same normalized statement.

        Returns (record, similarity) where similarity is 1.0 for an exact
        normalized match, or >= merge_similarity_threshold for near-duplicate
        wording (0 disables fuzzy matching).
        """
        target_key = self._normalize_content(content)
        if not target_key:
            return None
        threshold = self._settings.merge_similarity_threshold
        candidates = self._repository.list_memories(
            scope=scope,
            scope_key=scope_key,
            memory_type=memory_type,
            limit=MERGE_CANDIDATE_LIMIT,
            offset=0,
        )[0]
        for candidate in candidates:
            candidate_key = self._normalize_content(candidate.content)
            if not candidate_key:
                continue
            if candidate_key == target_key:
                return candidate, 1.0
            if threshold > 0.0:
                from difflib import SequenceMatcher

                similarity = SequenceMatcher(
                    None, candidate_key, target_key
                ).ratio()
                if similarity >= threshold:
                    return candidate, similarity
        return None

    @staticmethod
    def _normalize_content(content: str) -> str:
        """Normalize text so near-identical statements merge.

        Strips case, Unicode compat-decomposed marks, and punctuation/space
        separators; an empty result means nothing matches.
        """
        import re
        import unicodedata

        text = unicodedata.normalize("NFKC", content or "").lower()
        return re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE)

    def _merge_into(
        self,
        existing: MemoryEntryRecord,
        *,
        content: str,
        importance: int,
        tags: list[str],
        metadata_payload: dict[str, object],
        similarity: float,
    ) -> MemoryEntryRecord:
        """Merge a duplicate statement into an existing memory entry."""
        if len(content) > len(existing.content):
            existing.content = content
        existing.importance = max(existing.importance, importance)
        existing.tags = list(dict.fromkeys([*existing.tags, *tags]))
        merged_meta = dict(existing.metadata_payload)
        # Migrate legacy singular provenance keys into their list form so a
        # merged entry keeps the full history of requests/sessions.
        for legacy_key in ("request_id", "session_id"):
            legacy_value = merged_meta.pop(legacy_key, None)
            if legacy_value is not None:
                merged_meta.setdefault(f"{legacy_key}s", [])
                if legacy_value not in merged_meta[f"{legacy_key}s"]:
                    merged_meta[f"{legacy_key}s"].insert(0, legacy_value)
        for key, value in metadata_payload.items():
            if key in {"request_id", "session_id"}:
                merged_meta.setdefault(f"{key}s", [])
                seen = list(merged_meta[f"{key}s"])
                if value not in seen:
                    seen.append(value)
                    merged_meta[f"{key}s"] = seen
            else:
                merged_meta[key] = value
        merged_meta["merge_count"] = int(merged_meta.get("merge_count", 0)) + 1
        merged_meta["last_merge_similarity"] = similarity
        existing.metadata_payload = merged_meta
        self._repository.update_memory(
            existing.id,
            content=existing.content,
            importance=existing.importance,
            tags=existing.tags,
            metadata_payload=existing.metadata_payload,
        )
        self._repository.commit()
        return existing

    def list_memories(
        self,
        *,
        scope: str | None = None,
        scope_key: str | None = None,
        memory_type: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, object]:
        items, total = self._repository.list_memories(
            scope=scope,
            scope_key=scope_key,
            memory_type=memory_type,
            limit=limit,
            offset=offset,
        )
        return {
            "items": items,
            "total": total,
            "limit": limit,
            "offset": offset,
            "filters": {
                "scope": scope,
                "scope_key": scope_key,
                "memory_type": memory_type,
            },
        }

    def update_memory(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        importance: int | None = None,
        tags: list[str] | None = None,
        metadata_payload: dict[str, object] | None = None,
    ) -> MemoryEntryRecord | None:
        record = self._repository.update_memory(
            memory_id,
            content=content,
            importance=importance,
            tags=tags,
            metadata_payload=metadata_payload,
        )
        if record is not None:
            self._repository.commit()
        return record

    def delete_memory(self, memory_id: str) -> bool:
        deleted = self._repository.delete_memory(memory_id)
        if deleted:
            self._repository.commit()
        return deleted

    def _recall_for(
        self,
        *,
        user_id: str | None,
        project_key: str,
    ) -> list[MemoryEntryRecord]:
        """Recall durable memories in fixed scope priority.

        user -> project -> global; within each scope entries come back
        ordered by importance desc, created_at desc. Every scope contributes
        at most ``recall_limit`` entries and global always participates so
        tenant-wide defaults are never shadowed by write-only scopes.
        """
        scopes: list[tuple[MemoryScope, str]] = []
        if user_id:
            scopes.append((MemoryScope.USER, user_id))
        scopes.append((MemoryScope.PROJECT, project_key))
        scopes.append((MemoryScope.GLOBAL, GLOBAL_SCOPE_KEY))
        recalled: list[MemoryEntryRecord] = []
        for scope, scope_key in scopes:
            recalled.extend(
                self._repository.recall_memories(
                    scope=scope.value,
                    scope_key=scope_key,
                    limit=self._settings.recall_limit,
                )
            )
        return recalled

    # ---- automatic extraction (LLM) ----

    async def extract_and_store_memories(
        self,
        *,
        session_id: str,
        user_id: str | None,
        request_id: str,
        user_input: str,
        assistant_output: str,
        project_key: str,
        model: str = "deepseek/deepseek-chat",
    ) -> int:
        """Ask the LLM what is worth remembering, then persist durable entries.

        Extraction is best-effort by design: a failure (or an LLM response
        with empty payloads) is logged and skipped without breaking the
        workflow, and no placeholder memory is written.
        """
        llm_service = self._llm_service
        if llm_service is None or not getattr(llm_service, "is_configured", False):
            logger.info("memory extraction skipped: llm_service not configured")
            return 0
        system_prompt = (
            "You are the memory curator for a multi-agent platform. "
            "From the following user-assistant exchange, extract durable "
            "facts worth remembering long-term for future sessions. "
            "Categories:\n"
            "- user preference: the user's habits, style or preferences\n"
            "- project fact: durable facts about the user's project or domain\n"
            "Answer with ONLY a JSON object:\n"
            '{"preferences": [{"content": "...", "importance": 1-5}], '
            '"facts": [{"content": "...", "importance": 1-5}]}\n'
            'Use empty arrays when nothing is worth remembering.'
        )
        user_content = (
            f"User input:\n{user_input or '(empty)'}\n\n"
            f"Assistant output:\n{assistant_output or '(empty)'}"
        )
        request = LLMRequest(
            model=model,
            messages=[
                LLMMessage(role=LLMRole.SYSTEM, content=system_prompt),
                LLMMessage(role=LLMRole.USER, content=user_content),
            ],
            temperature=0.0,
            max_tokens=512,
            request_id=request_id,
            session_id=session_id,
            agent_id="memory_curator",
        )
        try:
            response = await llm_service.complete(request)
            payload = json.loads(response.content)
            preferences = payload.get("preferences") or []
            facts = payload.get("facts") or []
        except (json.JSONDecodeError, AttributeError, TypeError, ValueError) as exc:
            logger.warning(
                "memory extraction failed to parse LLM payload: %s", exc
            )
            return 0
        except Exception as exc:  # pragma: no cover - external boundary
            logger.warning("memory extraction failed: %s", exc)
            return 0

        stored = 0
        scope = "user" if user_id else "project"
        scope_key = user_id or project_key
        for category, entries in (("preference", preferences), ("fact", facts)):
            for entry in entries:
                content = str(entry.get("content") or "").strip()
                if not content:
                    continue
                self.add_memory(
                    memory_type=category,
                    content=content,
                    scope=scope,
                    scope_key=scope_key,
                    source="llm",
                    importance=int(entry.get("importance") or 3),
                    metadata_payload={
                        "request_id": request_id,
                        "session_id": session_id,
                    },
                )
                stored += 1
        return stored

    def close(self) -> None:
        self._repository.close()
