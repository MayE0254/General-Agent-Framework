from __future__ import annotations

from time import perf_counter
from uuid import uuid4

from fastapi import Request, Response

from app.observability import bind_request_context, clear_request_context, get_logger


logger = get_logger("app.api.middleware")


async def request_context_middleware(request: Request, call_next) -> Response:
    request_id = request.headers.get("X-Request-ID") or uuid4().hex
    trace_id = request.headers.get("X-Trace-ID") or request_id

    request.state.request_id = request_id
    request.state.trace_id = trace_id
    # Make request_id/trace_id available to every structlog line emitted
    # within this request scope (agents, tools, orchestrator, API layer).
    bind_request_context(request_id=request_id, trace_id=trace_id)

    start_time = perf_counter()
    try:
        response = await call_next(request)
    finally:
        clear_request_context()

    duration_ms = round((perf_counter() - start_time) * 1000, 2)

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Trace-ID"] = trace_id
    response.headers["X-Process-Time-MS"] = str(duration_ms)
    return response


async def access_log_middleware(request: Request, call_next) -> Response:
    start_time = perf_counter()
    response = await call_next(request)
    duration_ms = round((perf_counter() - start_time) * 1000, 2)

    logger.info(
        "request_completed",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=duration_ms,
    )
    return response
