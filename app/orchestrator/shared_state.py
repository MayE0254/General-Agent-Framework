from __future__ import annotations

from copy import deepcopy
from typing import Any

STANDARD_SHARED_STATE_KEYS = (
    "agent_outputs",
    "tool_results",
    "last_tool_results",
    "review_context",
)


def normalize_shared_state(
    shared_state: dict[str, Any] | None,
) -> dict[str, Any]:
    normalized = deepcopy(shared_state or {})

    agent_outputs = normalized.get("agent_outputs")
    normalized["agent_outputs"] = agent_outputs if isinstance(agent_outputs, dict) else {}

    tool_results = normalized.get("tool_results")
    normalized["tool_results"] = tool_results if isinstance(tool_results, dict) else {}

    last_tool_results = normalized.get("last_tool_results")
    normalized["last_tool_results"] = (
        list(last_tool_results) if isinstance(last_tool_results, list) else []
    )

    review_context = normalized.get("review_context")
    normalized["review_context"] = (
        review_context if isinstance(review_context, dict) else {}
    )

    return normalized


def record_agent_output(
    shared_state: dict[str, Any],
    *,
    agent_id: str,
    output: dict[str, Any],
) -> None:
    shared_state["agent_outputs"][agent_id] = deepcopy(output)


def append_tool_result(
    shared_state: dict[str, Any],
    *,
    agent_id: str,
    result_payload: dict[str, Any],
) -> None:
    shared_state["tool_results"].setdefault(agent_id, []).append(
        deepcopy(result_payload)
    )


def set_last_tool_results(
    shared_state: dict[str, Any],
    results: list[dict[str, Any]],
) -> None:
    shared_state["last_tool_results"] = [deepcopy(item) for item in results]


def set_review_context(
    shared_state: dict[str, Any],
    review_context: dict[str, Any],
) -> None:
    shared_state["review_context"] = deepcopy(review_context)
