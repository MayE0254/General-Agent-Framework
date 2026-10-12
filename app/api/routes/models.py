from __future__ import annotations

from fastapi import APIRouter, Depends, Query, status

from app.api.dependencies import (
    AuthenticatedUser,
    admin_required,
    get_current_user,
    get_model_pool_service,
)
from app.schemas import (
    ModelPoolDeleteResponse,
    ModelPoolEntryCreateRequest,
    ModelPoolEntryResponse,
    ModelPoolEntryUpdateRequest,
    ModelPoolListResponse,
    PaginationMetaResponse,
)
from app.services import ModelPoolService

router = APIRouter(tags=["models"])


def _to_model_pool_response(record) -> ModelPoolEntryResponse:
    """Expose pool metadata only.

    Credentials are never part of a pool row -- ``provider_key`` merely names
    an ``[llm.providers]`` block held in the server-side configuration -- so
    this response is safe for regular members as well as administrators.
    """
    return ModelPoolEntryResponse(
        id=record.id,
        name=record.name,
        provider_key=record.provider_key,
        model=record.model,
        description=record.description,
        temperature=record.temperature,
        max_tokens=record.max_tokens,
        input_per_million=record.input_per_million,
        output_per_million=record.output_per_million,
        enabled=record.enabled,
        is_default=record.is_default,
        sort_order=record.sort_order,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


@router.get("/models", response_model=ModelPoolListResponse)
def list_models(
    enabled_only: bool = Query(default=False),
    limit: int = Query(default=100, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    service: ModelPoolService = Depends(get_model_pool_service),
    user: AuthenticatedUser = Depends(get_current_user),
) -> ModelPoolListResponse:
    # Members only ever see enabled entries: a disabled model is an
    # administrative draft and must not appear in their picker.
    effective_enabled_only = enabled_only or not user.is_admin
    result = service.list_entries(
        enabled_only=effective_enabled_only,
        limit=limit,
        offset=offset,
    )
    return ModelPoolListResponse(
        items=[_to_model_pool_response(item) for item in result["items"]],
        pagination=PaginationMetaResponse(
            total=result["total"],
            limit=result["limit"],
            offset=result["offset"],
            filters={
                **result["filters"],
                "enabled_only": effective_enabled_only,
            },
        ),
    )


@router.post(
    "/models",
    response_model=ModelPoolEntryResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_model(
    payload: ModelPoolEntryCreateRequest,
    service: ModelPoolService = Depends(get_model_pool_service),
    _: AuthenticatedUser = Depends(admin_required),
) -> ModelPoolEntryResponse:
    record = service.create_entry(**payload.model_dump())
    return _to_model_pool_response(record)


@router.patch("/models/{entry_id}", response_model=ModelPoolEntryResponse)
def update_model(
    entry_id: str,
    payload: ModelPoolEntryUpdateRequest,
    service: ModelPoolService = Depends(get_model_pool_service),
    _: AuthenticatedUser = Depends(admin_required),
) -> ModelPoolEntryResponse:
    changes = payload.model_dump(exclude_unset=True)
    record = service.update_entry(entry_id, **changes)
    return _to_model_pool_response(record)


@router.delete("/models/{entry_id}", response_model=ModelPoolDeleteResponse)
def delete_model(
    entry_id: str,
    service: ModelPoolService = Depends(get_model_pool_service),
    _: AuthenticatedUser = Depends(admin_required),
) -> ModelPoolDeleteResponse:
    service.delete_entry(entry_id)
    return ModelPoolDeleteResponse(entry_id=entry_id, deleted=True)
