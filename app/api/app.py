from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api.error_handlers import register_exception_handlers
from app.api.lifespan import application_lifespan
from app.api.middleware import access_log_middleware, request_context_middleware
from app.api.routes.agents import router as agents_router
from app.api.routes.automation import router as automation_router
from app.api.routes.auth import router as auth_router
from app.api.routes.chat import router as chat_router
from app.api.routes.health import router as health_router
from app.api.routes.models import router as models_router
from app.api.routes.reviews import router as reviews_router
from app.api.routes.runtime import router as runtime_router
from app.api.routes.system import router as system_router
from app.api.routes.users import router as users_router
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
    app.include_router(auth_router, prefix=settings.api.prefix)
    app.include_router(users_router, prefix=settings.api.prefix)
    app.include_router(agents_router, prefix=settings.api.prefix)
    app.include_router(automation_router, prefix=settings.api.prefix)
    app.include_router(chat_router, prefix=settings.api.prefix)
    app.include_router(models_router, prefix=settings.api.prefix)
    app.include_router(runtime_router, prefix=settings.api.prefix)
    app.include_router(reviews_router, prefix=settings.api.prefix)
    app.include_router(system_router, prefix=settings.api.prefix)
    app.include_router(workflows_router, prefix=settings.api.prefix)

    @app.get("/", include_in_schema=False)
    async def dashboard_root() -> RedirectResponse:
        # 根路径直接跳转运行控制台，避免访问者看到 404。
        return RedirectResponse(url="/console/")

    # 控制台静态资源（Vue 3 + Element Plus，本地 vendor，无构建步骤）。
    # 必须在所有 API 路由之后挂载，避免覆盖 /api、/docs 等前缀。
    console_dir = Path(__file__).resolve().parent.parent / "static" / "console"
    app.mount("/console", StaticFiles(directory=console_dir, html=True), name="console")

    return app


app = create_app()
