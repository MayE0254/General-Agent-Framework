"""Password hashing and signed-cookie session helpers (stdlib only)."""
from __future__ import annotations

import hashlib
import hmac
import secrets
import time

_PBKDF2_ITERATIONS = 120_000
_SALT_BYTES = 16
_SESSION_TTL_SECONDS = 7 * 24 * 3600


def hash_password(password: str) -> str:
    salt = secrets.token_hex(_SALT_BYTES)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), _PBKDF2_ITERATIONS
    ).hex()
    return f"pbkdf2_sha256${_PBKDF2_ITERATIONS}${salt}${digest}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algorithm, iterations, salt, digest = stored.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        candidate = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt), int(iterations)
        ).hex()
        return hmac.compare_digest(candidate, digest)
    except (ValueError, TypeError):
        return False


def _sign(payload: str, secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()


def create_session_token(user_id: str, secret: str, ttl_seconds: int = _SESSION_TTL_SECONDS) -> str:
    expires = int(time.time()) + ttl_seconds
    payload = f"{user_id}:{expires}"
    return f"{payload}:{_sign(payload, secret)}"


def verify_session_token(token: str, secret: str) -> str | None:
    """Return user_id when the token is valid and unexpired, else None."""
    try:
        user_id, expires_raw, signature = token.split(":", 2)
        payload = f"{user_id}:{expires_raw}"
        if not hmac.compare_digest(signature, _sign(payload, secret)):
            return None
        if int(expires_raw) < time.time():
            return None
        return user_id or None
    except (ValueError, TypeError):
        return None
