from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core import AppError


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "")
        trace_id = getattr(request.state, "trace_id", "")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.error_code,
                    "message": exc.message,
                    "details": exc.details,
                },
                "path": str(request.url.path),
                "request_id": request_id,
                "trace_id": trace_id,
            },
            headers={"X-Request-ID": request_id, "X-Trace-ID": trace_id},
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "")
        trace_id = getattr(request.state, "trace_id", "")
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "internal_server_error",
                    "message": "An unexpected error occurred.",
                    "details": {},
                },
                "path": str(request.url.path),
                "request_id": request_id,
                "trace_id": trace_id,
            },
            headers={"X-Request-ID": request_id, "X-Trace-ID": trace_id},
        )
