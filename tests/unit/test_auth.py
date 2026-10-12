"""Permission-matrix tests for the console login system.

The rules these tests pin down come from
``docs/2026-09-16_1319_用户登录与权限系统_update.md``:

* every console endpoint sits behind the signed session cookie,
* ``/system/*`` and the broad runtime listings are administrator-only,
* members only ever see the requests, sessions and reviews they submitted,
* decisions (approve/reject) belong to the submitting member, never to an
  administrator and never to a bystander.
"""
from __future__ import annotations

import contextlib
import uuid

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from app.core.security import create_session_token, hash_password
from app.models import UserRecord
from app.repositories import UserRepository


# (method, path, json-body) for endpoints that must never answer anonymously.
_PROTECTED_ENDPOINTS: list[tuple[str, str, dict | None]] = [
    ("GET", "/api/v1/auth/me", None),
    ("GET", "/api/v1/agents", None),
    ("GET", "/api/v1/automation/agents", None),
    ("GET", "/api/v1/users", None),
    ("GET", "/api/v1/workflows/submissions", None),
    ("GET", "/api/v1/workflows/tasks/not-a-real-task", None),
    ("GET", "/api/v1/runtime/tasks", None),
    ("GET", "/api/v1/runtime/workflow-runs", None),
    ("GET", "/api/v1/runtime/sessions", None),
    ("GET", "/api/v1/runtime/memories", None),
    ("GET", "/api/v1/runtime/reviews", None),
    ("GET", "/api/v1/runtime/usage/llm", None),
    ("GET", "/api/v1/models", None),
    ("GET", "/api/v1/chat/sessions", None),
    ("GET", "/api/v1/chat/sessions/not-a-real-session/messages", None),
    ("GET", "/api/v1/system/resources", None),
    ("GET", "/api/v1/system/workers", None),
    ("POST", "/api/v1/workflows/run", {}),
    ("POST", "/api/v1/runtime/memories", {}),
    ("POST", "/api/v1/models", {}),
    ("POST", "/api/v1/chat/sessions", {}),
    ("POST", "/api/v1/chat/turns", {}),
    ("PATCH", "/api/v1/models/not-a-real-entry", {"name": "x"}),
    ("DELETE", "/api/v1/models/not-a-real-entry", None),
    ("DELETE", "/api/v1/chat/sessions/not-a-real-session", None),
]

# Endpoints a regular member must not reach even when logged in.
_ADMIN_ONLY_ENDPOINTS: list[tuple[str, str, dict | None]] = [
    ("GET", "/api/v1/users", None),
    ("GET", "/api/v1/runtime/tasks", None),
    ("GET", "/api/v1/runtime/sessions", None),
    ("GET", "/api/v1/runtime/memories", None),
    ("GET", "/api/v1/system/resources", None),
    ("GET", "/api/v1/system/workers", None),
    ("GET", "/api/v1/system/acceptance", None),
    ("POST", "/api/v1/system/workers/recover", None),
    ("POST", "/api/v1/runtime/memories", {}),
    # Model pool curation is administrators' work; members only ever read the
    # enabled subset through GET /models.
    ("POST", "/api/v1/models", {}),
    ("PATCH", "/api/v1/models/not-a-real-entry", {"name": "x"}),
    ("DELETE", "/api/v1/models/not-a-real-entry", None),
]


def _call(client: TestClient, method: str, path: str, body: dict | None):
    if method == "POST":
        return client.post(path, json=body if body is not None else {})
    if method == "PATCH":
        return client.patch(path, json=body if body is not None else {})
    if method == "DELETE":
        return client.delete(path)
    return client.get(path)


def test_every_console_endpoint_requires_login(anonymous_client: TestClient) -> None:
    for method, path, body in _PROTECTED_ENDPOINTS:
        response = _call(anonymous_client, method, path, body)
        assert response.status_code == 401, f"{method} {path} -> {response.status_code}"


def test_admin_only_endpoints_reject_members(api_client: TestClient, as_user) -> None:
    as_user(user_id="member-plain", username="member")
    for method, path, body in _ADMIN_ONLY_ENDPOINTS:
        response = _call(api_client, method, path, body)
        assert response.status_code == 403, f"{method} {path} -> {response.status_code}"


def test_member_sees_only_own_requests_and_sessions(
    api_client: TestClient, as_user
) -> None:
    owner = as_user(user_id="member-owner", username="owner")
    request_id = f"auth-req-{uuid.uuid4().hex[:8]}"
    session_id = f"auth-sess-{uuid.uuid4().hex[:8]}"

    run = api_client.post(
        "/api/v1/workflows/run",
        json={
            "request_id": request_id,
            "initial_agent_id": "planner_agent",
            "session_id": session_id,
            "input_text": "build a reusable enterprise agent platform",
            "structured_input": {"requested_agent": "executor_agent"},
        },
    )
    assert run.status_code == 200
    assert run.json()["status"] == "completed"

    listing = api_client.get("/api/v1/runtime/workflow-runs")
    assert listing.status_code == 200
    assert request_id in {
        item["request_id"] for item in listing.json()["items"]
    }

    # A different member must not even learn that the request exists.
    as_user(user_id="member-bystander", username="bystander")
    assert api_client.get(f"/api/v1/runtime/tasks/{request_id}").status_code == 404
    assert (
        api_client.get(f"/api/v1/runtime/workflow-runs/{request_id}").status_code
        == 404
    )
    assert (
        api_client.get(f"/api/v1/runtime/sessions/{session_id}").status_code == 404
    )
    assert (
        api_client.get(
            f"/api/v1/runtime/sessions/{session_id}/messages"
        ).status_code
        == 404
    )
    assert api_client.get(
        "/api/v1/runtime/workflow-runs", params={"request_id": request_id}
    ).json()["items"] == []

    # Administrators review read-only, so they still see the run.
    as_user(user_id="member-admin", username="admin2", role="admin")
    assert api_client.get(f"/api/v1/runtime/tasks/{request_id}").status_code == 200
    assert api_client.get(f"/api/v1/runtime/sessions/{session_id}").status_code == 200

    assert owner.id == "member-owner"


def test_review_decision_belongs_to_submitter_only(
    api_client: TestClient, as_user
) -> None:
    as_user(user_id="member-submitter", username="submitter")
    request_id = f"auth-review-{uuid.uuid4().hex[:8]}"
    run = api_client.post(
        "/api/v1/workflows/run",
        json={
            "request_id": request_id,
            "initial_agent_id": "planner_agent",
            "session_id": "session-auth-review",
            "input_text": "this task needs human approval",
            "structured_input": {"requires_human": True},
        },
    )
    assert run.status_code == 200
    review_id = run.json()["review_id"]
    assert review_id

    # Another member cannot see or decide it.
    as_user(user_id="member-other", username="other")
    assert api_client.get(f"/api/v1/runtime/reviews/{review_id}").status_code == 404
    assert (
        api_client.post(
            f"/api/v1/runtime/reviews/{review_id}/approve", json={"note": "nope"}
        ).status_code
        == 404
    )
    assert (
        api_client.post(
            f"/api/v1/runtime/reviews/{review_id}/reject", json={"note": "nope"}
        ).status_code
        == 404
    )
    assert api_client.get("/api/v1/runtime/reviews").json()["items"] == []

    # Administrators are read-only on decisions: 403, not 404.
    as_user(user_id="member-admin3", username="admin3", role="admin")
    assert api_client.get(f"/api/v1/runtime/reviews/{review_id}").status_code == 200
    for decision in ("approve", "reject"):
        response = api_client.post(
            f"/api/v1/runtime/reviews/{review_id}/{decision}", json={"note": "nope"}
        )
        assert response.status_code == 403

    # The decider cannot be forged through the request body either.
    as_user(user_id="member-submitter", username="submitter")
    forged = api_client.post(
        f"/api/v1/runtime/reviews/{review_id}/approve",
        json={"decided_by": "somebody-else", "note": "stolen"},
    )
    assert forged.status_code == 422

    approved = api_client.post(
        f"/api/v1/runtime/reviews/{review_id}/approve", json={"note": "ok"}
    )
    assert approved.status_code == 200
    assert approved.json()["review"]["decided_by"] == "submitter"


def test_tampered_session_cookie_is_rejected(
    api_client: TestClient, anonymous_client: TestClient
) -> None:
    secret = api_client.app.state.settings.auth.session_secret
    cookie_name = api_client.app.state.settings.auth.session_cookie_name

    anonymous_client.cookies.set(cookie_name, "not-a-token")
    assert anonymous_client.get("/api/v1/auth/me").status_code == 401

    anonymous_client.cookies.set(
        cookie_name, create_session_token("test-admin-id", "wrong-secret")
    )
    assert anonymous_client.get("/api/v1/auth/me").status_code == 401

    anonymous_client.cookies.set(
        cookie_name, create_session_token("test-admin-id", secret, ttl_seconds=-1)
    )
    assert anonymous_client.get("/api/v1/auth/me").status_code == 401

    anonymous_client.cookies.set(
        cookie_name, create_session_token("no-such-user-id", secret)
    )
    assert anonymous_client.get("/api/v1/auth/me").status_code == 401

    anonymous_client.cookies.clear()


@contextlib.contextmanager
def _temporary_account(client: TestClient, *, password: str, role: str = "user"):
    """Seed a real account row so the login endpoint can be exercised end to end."""
    session_factory = getattr(
        client.app.state.resources, "database_session_factory", None
    )
    if session_factory is None:
        pytest.skip("database session factory is not available")

    session = session_factory()
    repository = UserRepository(session)
    record = UserRecord(
        id=f"auth-it-{uuid.uuid4().hex[:12]}",
        username=f"auth-it-{uuid.uuid4().hex[:8]}",
        password_hash=hash_password(password),
        role=role,
        display_name="Auth Integration",
        is_active=True,
    )
    try:
        repository.add(record)
        repository.commit()
        yield record
    finally:
        try:
            stored = repository.get(record.id)
            if stored is not None:
                repository.delete(stored)
                repository.commit()
        finally:
            repository.close()


def test_login_logout_round_trip(anonymous_client: TestClient) -> None:
    cookie_name = anonymous_client.app.state.settings.auth.session_cookie_name

    with _temporary_account(anonymous_client, password="secret-pass") as account:
        bad = anonymous_client.post(
            "/api/v1/auth/login",
            json={"username": account.username, "password": "wrong-pass"},
        )
        assert bad.status_code == 401

        login = anonymous_client.post(
            "/api/v1/auth/login",
            json={"username": account.username, "password": "secret-pass"},
        )
        assert login.status_code == 200
        assert login.json()["id"] == account.id
        assert login.json()["role"] == "user"
        assert anonymous_client.cookies.get(cookie_name)

        # The freshly minted cookie is accepted by the real dependency.
        me = anonymous_client.get("/api/v1/auth/me")
        assert me.status_code == 200
        assert me.json()["username"] == account.username

        logout = anonymous_client.post("/api/v1/auth/logout")
        assert logout.status_code == 200
        assert not anonymous_client.cookies.get(cookie_name)
        assert anonymous_client.get("/api/v1/auth/me").status_code == 401
