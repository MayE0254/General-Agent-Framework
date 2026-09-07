from __future__ import annotations

import logging
import sys

try:
    import structlog
    from structlog.contextvars import (
        bind_contextvars,
        clear_contextvars,
    )
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    structlog = None  # type: ignore[assignment]
    bind_contextvars = None  # type: ignore[assignment]
    clear_contextvars = None  # type: ignore[assignment]

_configured = False


def _configure_structlog(*, debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def configure_logging(*, debug: bool = False) -> None:
    global _configured
    if structlog is None:  # pragma: no cover - optional dependency
        root_logger = logging.getLogger()
        if not root_logger.handlers:
            logging.basicConfig(
                level=logging.DEBUG if debug else logging.INFO,
                format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            )
        _configured = True
        return

    if _configured:
        return
    _configure_structlog(debug=debug)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    if structlog is None:
        return logging.getLogger(name)
    return structlog.get_logger(name)  # type: ignore[no-any-return]


def bind_request_context(*, request_id: str, trace_id: str) -> None:
    """Bind request-scoped identifiers into structlog contextvars.

    Every log line emitted inside the request scope (agents, tools,
    orchestrator, API layer) automatically carries request_id/trace_id
    because ``merge_contextvars`` is in the processor chain.
    """
    if bind_contextvars is None:
        return
    bind_contextvars(request_id=request_id, trace_id=trace_id)


def clear_request_context() -> None:
    """Clear structlog contextvars after the request scope ends.

    Prevents request identifiers leaking into later requests or async
    tasks that share the event loop.
    """
    if clear_contextvars is None:
        return
    clear_contextvars()
