from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.api.dependencies import (
    AuthenticatedUser,
    get_app_settings,
    get_current_user,
    get_database_session_factory,
    require_admin,
)
from app.core import Settings
from app.core.security import hash_password, verify_password
from app.models.user import UserRecord
from app.repositories import UserRepository

router = APIRouter(prefix="/users", tags=["users"])


class CreateUserRequest(BaseModel):
    username: str = Field(min_length=2, max_length=64)
    password: str = Field(min_length=6, max_length=128)
    role: str = Field(default="user", pattern="^(admin|user)$")
    display_name: str | None = Field(default=None, max_length=64)


class UpdateUserRequest(BaseModel):
    password: str | None = Field(default=None, min_length=6, max_length=128)
    role: str | None = Field(default=None, pattern="^(admin|user)$")
    display_name: str | None = Field(default=None, max_length=64)
    is_active: bool | None = None


def admin_user(user: AuthenticatedUser = Depends(get_current_user)) -> AuthenticatedUser:
    return require_admin(user)


def _serialize(record: UserRecord) -> dict[str, object]:
    return {
        "id": record.id,
        "username": record.username,
        "role": record.role,
        "display_name": record.display_name,
        "is_active": record.is_active,
        "created_at": record.created_at.isoformat() if record.created_at else None,
    }


def _open_session(request: Request, settings: Settings):
    return get_database_session_factory(request)()


@router.get("")
def list_users(
    request: Request,
    settings: Settings = Depends(get_app_settings),
    _: AuthenticatedUser = Depends(admin_user),
) -> dict[str, object]:
    session = _open_session(request, settings)
    try:
        records = UserRepository(session).list()
        return {"items": [_serialize(r) for r in records], "total": len(records)}
    finally:
        session.close()


@router.post("", status_code=status.HTTP_201_CREATED)
def create_user(
    body: CreateUserRequest,
    request: Request,
    settings: Settings = Depends(get_app_settings),
    _: AuthenticatedUser = Depends(admin_user),
) -> dict[str, object]:
    username = body.username.strip()
    session = _open_session(request, settings)
    try:
        repository = UserRepository(session)
        if repository.get_by_username(username) is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Username already exists.",
            )
        record = UserRecord(
            id=str(uuid4()),
            username=username,
            password_hash=hash_password(body.password),
            role=body.role,
            display_name=body.display_name,
            is_active=True,
        )
        repository.add(record)
        repository.commit()
        return _serialize(record)
    finally:
        session.close()


@router.patch("/{user_id}")
def update_user(
    user_id: str,
    body: UpdateUserRequest,
    request: Request,
    settings: Settings = Depends(get_app_settings),
    actor: AuthenticatedUser = Depends(admin_user),
) -> dict[str, object]:
    session = _open_session(request, settings)
    try:
        repository = UserRepository(session)
        record = repository.get(user_id)
        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User was not found.",
            )
        if body.password is not None:
            record.password_hash = hash_password(body.password)
        if body.role is not None:
            if record.id == actor.id and body.role != "admin":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Cannot demote your own admin account.",
                )
            record.role = body.role
        if body.display_name is not None:
            record.display_name = body.display_name
        if body.is_active is not None:
            if record.id == actor.id and not body.is_active:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Cannot disable your own account.",
                )
            record.is_active = body.is_active
        repository.commit()
        return _serialize(record)
    finally:
        session.close()


@router.delete("/{user_id}")
def delete_user(
    user_id: str,
    request: Request,
    settings: Settings = Depends(get_app_settings),
    actor: AuthenticatedUser = Depends(admin_user),
) -> dict[str, bool]:
    if user_id == actor.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own account.",
        )
    session = _open_session(request, settings)
    try:
        repository = UserRepository(session)
        record = repository.get(user_id)
        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User was not found.",
            )
        repository.delete(record)
        repository.commit()
        return {"ok": True}
    finally:
        session.close()
