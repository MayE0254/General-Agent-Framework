from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field

from app.api.dependencies import (
    AuthenticatedUser,
    get_app_settings,
    get_current_user,
    get_database_session_factory,
)
from app.core import Settings
from app.core.security import create_session_token, verify_password
from app.repositories import UserRepository

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)


class UserPayload(BaseModel):
    id: str
    username: str
    role: str
    display_name: str | None


def _to_payload(user: AuthenticatedUser) -> dict[str, str | None]:
    return {
        "id": user.id,
        "username": user.username,
        "role": user.role,
        "display_name": user.display_name,
    }


@router.post("/login")
def login(
    body: LoginRequest,
    request: Request,
    response: Response,
    settings: Settings = Depends(get_app_settings),
) -> dict[str, str | None]:
    session_factory = get_database_session_factory(request)
    session = session_factory()
    try:
        record = UserRepository(session).get_by_username(body.username.strip())
    finally:
        session.close()
    if record is None or not verify_password(body.password, record.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
        )
    if not record.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled.",
        )
    token = create_session_token(
        record.id,
        settings.auth.session_secret,
        ttl_seconds=settings.auth.session_ttl_seconds,
    )
    response.set_cookie(
        key=settings.auth.session_cookie_name,
        value=token,
        max_age=settings.auth.session_ttl_seconds,
        httponly=True,
        samesite="lax",
        path="/",
    )
    return {
        "id": record.id,
        "username": record.username,
        "role": record.role,
        "display_name": record.display_name,
    }


@router.post("/logout")
def logout(
    response: Response,
    settings: Settings = Depends(get_app_settings),
) -> dict[str, bool]:
    response.delete_cookie(
        key=settings.auth.session_cookie_name,
        path="/",
    )
    return {"ok": True}


@router.get("/me")
def me(user: AuthenticatedUser = Depends(get_current_user)) -> dict[str, str | None]:
    return _to_payload(user)
