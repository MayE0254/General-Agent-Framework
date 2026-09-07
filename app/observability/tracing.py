from __future__ import annotations

from typing import Any, Protocol

from app.core.settings import ObservabilitySettings


class Tracer(Protocol):
    """Minimal tracer surface used across the framework.

    Implementations may be no-ops (unconfigured) or wrap Langfuse.
    """

    def start_workflow_span(self, state: Any) -> Any:
        """Start a trace-level span for a whole workflow run."""

    def finish_workflow_span(self, span: Any, state: Any) -> None:
        """Attach the final workflow state to the span and end it."""

    def start_agent_span(self, agent_id: str, step_index: int, context: Any) -> Any:
        """Start a span around a single agent step."""

    def finish_agent_span(self, span: Any, result: Any) -> None:
        """Attach the agent result to a span and end it."""

    def start_tool_span(self, tool_name: str, agent_id: str, request: Any) -> Any:
        """Start a span around a single tool invocation."""

    def finish_tool_span(self, span: Any, result: Any) -> None:
        """Attach the tool result to a span and end it."""

    def start_llm_span(self, request: Any) -> Any:
        """Start a span around an LLM call; returns an opaque span handle."""

    def finish_llm_span(self, span: Any, response: Any) -> None:
        """Attach the parsed LLM response to a span and end it."""

    def capture_error(self, error: BaseException, **context: Any) -> None:
        """Record an exception with optional structured context."""


class NullTracer:
    """No-op tracer used when observability is not configured."""

    def start_workflow_span(self, state: Any) -> Any:
        return None

    def finish_workflow_span(self, span: Any, state: Any) -> None:
        return None

    def start_agent_span(self, agent_id: str, step_index: int, context: Any) -> Any:
        return None

    def finish_agent_span(self, span: Any, result: Any) -> None:
        return None

    def start_tool_span(self, tool_name: str, agent_id: str, request: Any) -> Any:
        return None

    def finish_tool_span(self, span: Any, result: Any) -> None:
        return None

    def start_llm_span(self, request: Any) -> Any:
        return None

    def finish_llm_span(self, span: Any, response: Any) -> None:
        return None

    def capture_error(self, error: BaseException, **context: Any) -> None:
        return None


class LangfuseTracer:
    """Langfuse-backed tracer.

    Only constructs the Langfuse client when host and keys are present,
    so the application stays bootable without a configured backend.
    """

    def __init__(self, settings: ObservabilitySettings) -> None:
        self._enabled = bool(
            settings.langfuse_host
            and settings.langfuse_public_key
            and settings.langfuse_secret_key
        )
        self._client = None
        if self._enabled:
            try:
                from langfuse import Langfuse

                self._client = Langfuse(
                    host=settings.langfuse_host,
                    public_key=settings.langfuse_public_key,
                    secret_key=settings.langfuse_secret_key,
                )
            except Exception as exc:  # pragma: no cover - external boundary
                self._enabled = False
                self._client = None
                self._init_error = str(exc)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def start_workflow_span(self, state: Any) -> Any:
        if not self._enabled or self._client is None:
            return None
        return self._client.trace(
            name=f"workflow:{state.workflow_id or state.request_id}",
            input={
                "request_id": state.request_id,
                "trace_id": state.trace_id,
                "initial_agent_id": state.initial_agent_id,
                "input_text": state.input_text,
                "structured_input": state.structured_input,
            },
        )

    def finish_workflow_span(self, span: Any, state: Any) -> None:
        if not self._enabled or span is None:
            return
        span.update(
            output={
                "status": getattr(state, "status", None),
                "execution_path": getattr(state, "execution_path", []),
                "final_result": getattr(state, "final_result", None),
                "errors": getattr(state, "errors", []),
            }
        )
        span.end()

    def start_agent_span(self, agent_id: str, step_index: int, context: Any) -> Any:
        if not self._enabled or self._client is None:
            return None
        return self._client.span(
            name=f"agent:{agent_id}",
            input={
                "step_index": step_index,
                "input_text": context.input_text,
                "structured_input": context.structured_input,
            },
        )

    def finish_agent_span(self, span: Any, result: Any) -> None:
        if not self._enabled or span is None:
            return
        span.end(
            output={
                "status": result.status,
                "summary": result.summary,
                "next_agent_id": result.next_agent_id,
                "output": result.output,
            }
        )

    def start_tool_span(self, tool_name: str, agent_id: str, request: Any) -> Any:
        if not self._enabled or self._client is None:
            return None
        return self._client.span(
            name=f"tool:{tool_name}",
            input={
                "agent_id": agent_id,
                "arguments": request.arguments,
            },
        )

    def finish_tool_span(self, span: Any, result: Any) -> None:
        if not self._enabled or span is None:
            return
        span.end(
            output={
                "status": result.status,
                "message": result.message,
                "content": result.content,
            }
        )

    def start_llm_span(self, request: Any) -> Any:
        if not self._enabled or self._client is None:
            return None
        return self._client.generation(
            name=f"llm:{request.model}",
            model=request.model,
            input={
                "messages": [
                    message.model_dump(exclude_none=True)
                    for message in request.messages
                ],
                "temperature": request.temperature,
            },
        )

    def finish_llm_span(self, span: Any, response: Any) -> None:
        if not self._enabled or span is None:
            return
        span.end(
            output={"content": response.content},
            usage={
                "input": response.usage.prompt_tokens,
                "output": response.usage.completion_tokens,
                "total": response.usage.total_tokens,
            },
            metadata={"model": response.model, "latency_ms": response.latency_ms},
        )

    def capture_error(self, error: BaseException, **context: Any) -> None:
        if not self._enabled or self._client is None:
            return
        self._client.capture_exception(error, **context)


def create_tracer(
    settings: ObservabilitySettings | None = None,
) -> Tracer:
    """Build the appropriate tracer based on observability configuration."""
    if settings is not None and settings.langfuse_host:
        return LangfuseTracer(settings)
    return NullTracer()
