from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from app.api.error_handlers import register_exception_handlers
from app.api.lifespan import application_lifespan
from app.api.middleware import access_log_middleware, request_context_middleware
from app.api.routes.agents import router as agents_router
from app.api.routes.health import router as health_router
from app.api.routes.reviews import router as reviews_router
from app.api.routes.runtime import router as runtime_router
from app.api.routes.system import router as system_router
from app.api.routes.workflows import router as workflows_router
from app.core import get_settings
from app.observability import configure_logging


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(debug=settings.app.debug)
    app = FastAPI(
        title=settings.app.name,
        debug=settings.app.debug,
        version="0.1.0",
        lifespan=application_lifespan,
    )

    register_exception_handlers(app)
    app.middleware("http")(access_log_middleware)
    app.middleware("http")(request_context_middleware)
    app.include_router(health_router)
    app.include_router(agents_router, prefix=settings.api.prefix)
    app.include_router(runtime_router, prefix=settings.api.prefix)
    app.include_router(reviews_router, prefix=settings.api.prefix)
    app.include_router(system_router, prefix=settings.api.prefix)
    app.include_router(workflows_router, prefix=settings.api.prefix)

    @app.get("/", include_in_schema=False)
    async def dashboard_root() -> RedirectResponse:
        # 根路径直接跳转运行控制台，避免访问者看到 404。
        return RedirectResponse(url=f"{settings.api.prefix}/system/dashboard")

    return app


app = create_app()
