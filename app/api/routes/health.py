from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.api.dependencies import get_app_resources, get_app_settings
from app.core import Settings
from app.infra import AppResources

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check(
    request: Request,
    settings: Settings = Depends(get_app_settings),
    resources: AppResources = Depends(get_app_resources),
) -> dict[str, str | bool | dict[str, str]]:
    return {
        "status": "ok",
        "service": settings.observability.service_name,
        "environment": settings.app.env,
        "debug": settings.app.debug,
        "resources": resources.health_summary(),
        "request_id": getattr(request.state, "request_id", ""),
        "trace_id": getattr(request.state, "trace_id", ""),
    }
