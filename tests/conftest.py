from __future__ import annotations

from collections.abc import Callable, Generator
from pathlib import Path
import os

import pytest
from fastapi.testclient import TestClient

import app.runtime_stack as runtime_stack_module
from app.api.dependencies import AuthenticatedUser, get_current_user
from app.core.settings import get_settings
from app.llm import LLMService

os.environ.setdefault("REDIS_USERNAME", "multiagent")
os.environ.setdefault("REDIS_PASSWORD", "redis-test-password")
os.environ.setdefault("DEEPSEEK_API_KEY", "sk-test-env-key")

# Isolate tests from development data: every PostgreSQL engine this process
# builds (including the app runtime's) resolves unqualified tables in the
# dedicated test schema instead of public. The session fixture below recreates
# that schema, migrates it with alembic, and drops it afterwards, so tests can
# neither read nor mutate development rows (e.g. the seeded model pool).
TEST_DB_SCHEMA = "multiagent_test"
get_settings().database.table_schema = TEST_DB_SCHEMA

from app.main import app  # noqa: E402

# Every console endpoint is behind a session cookie (see
# app/api/dependencies.py). Unit tests must not depend on a seeded account row
# in the real database, so the identity is injected through the same
# dependency the router uses — the signed-cookie parsing itself is covered by
# tests/unit/test_auth.py.
TEST_ADMIN = AuthenticatedUser(
    id="test-admin-id",
    username="test-admin",
    role="admin",
    display_name="Test Admin",
)


@pytest.fixture(scope="session", autouse=True)
def _isolated_test_schema() -> Generator[None, None, None]:
    """Run the whole suite against the dedicated ``multiagent_test`` schema.

    The schema is dropped, recreated and migrated with alembic before the
    first test, then dropped again at teardown — tests start from a clean,
    fully-migrated state and can never touch development data.
    """
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import create_engine, text

    url = get_settings().database.build_url()
    engine = create_engine(url)
    with engine.begin() as connection:
        connection.execute(text(f'DROP SCHEMA IF EXISTS "{TEST_DB_SCHEMA}" CASCADE'))
        connection.execute(text(f'CREATE SCHEMA "{TEST_DB_SCHEMA}"'))
    engine.dispose()

    project_root = Path(__file__).resolve().parents[1]
    alembic_config = Config(str(project_root / "alembic.ini"))
    alembic_config.set_main_option("script_location", str(project_root / "migrations"))
    # migrations/env.py prefers this override, and the URL already carries the
    # search_path option, so alembic_version lands in the test schema too.
    previous_migrate_url = os.environ.get("MIGRATE_DATABASE_URL")
    os.environ["MIGRATE_DATABASE_URL"] = url
    try:
        command.upgrade(alembic_config, "head")
        yield
    finally:
        if previous_migrate_url is None:
            os.environ.pop("MIGRATE_DATABASE_URL", None)
        else:
            os.environ["MIGRATE_DATABASE_URL"] = previous_migrate_url
        teardown_engine = create_engine(url)
        with teardown_engine.begin() as connection:
            connection.execute(text(f'DROP SCHEMA IF EXISTS "{TEST_DB_SCHEMA}" CASCADE'))
        teardown_engine.dispose()


@pytest.fixture
def api_client(monkeypatch) -> Generator[TestClient, None, None]:
    """Provide a TestClient whose lifespan has actually run.

    Entering the ``with`` block is required for FastAPI's lifespan handler
    to execute; without it ``app.state`` stays uninitialized and every
    request fails with 503 Service Unavailable.

    The LLM service is replaced with a fake completion_fn so unit tests stay
    deterministic and never hit the real provider (even when a real api_key
    is configured in the environment). The fake returns non-JSON content,
    which makes planner/router/reviewer fall back to their deterministic
    rule paths.

    The client acts as :data:`TEST_ADMIN` by default; use the ``as_user``
    fixture to act as a regular member instead.
    """
    # API unit tests must not load local ML models or contact Milvus. RAG
    # integration tests explicitly inject a retriever and enable the policy.
    monkeypatch.setattr(get_settings().knowledge, "retriever", "inmemory")
    monkeypatch.setattr(get_settings().knowledge.chat, "enabled", False)

    async def _fake_completion(kwargs: dict) -> object:
        class _Message:
            content: str = "not-json-fallback-content"

        class _Choice:
            message: object = None

        class _Usage:
            prompt_tokens = 3
            completion_tokens = 2
            total_tokens = 5

        class _Raw:
            model: str = ""
            choices: list = []
            usage: object = None

        raw = _Raw()
        raw.model = kwargs["model"]
        message = _Message()
        raw.choices = [_Choice()]
        raw.choices[0].message = message
        raw.usage = _Usage()
        return raw

    def _fake_create_llm_service(
        settings, *, tracer=None, audit_recorder=None, pricing_resolver=None
    ):
        # Mirrors app.llm.factory: the factory receives the whole Settings and
        # hands LLMService its [llm] block, so the fake must do the same.
        return LLMService(
            settings.llm,
            completion_fn=_fake_completion,
            tracer=tracer,
            audit_recorder=audit_recorder,
            pricing_resolver=pricing_resolver,
        )

    monkeypatch.setattr(
        runtime_stack_module,
        "create_llm_service",
        _fake_create_llm_service,
    )
    monkeypatch.setitem(app.dependency_overrides, get_current_user, lambda: TEST_ADMIN)

    with TestClient(app) as client:
        yield client


@pytest.fixture
def as_user(api_client: TestClient, monkeypatch) -> Callable[..., AuthenticatedUser]:
    """Switch which member the ``api_client`` session belongs to.

    Returns a callable so a single test can run a workflow as one member and
    then verify that another member cannot read or decide it.
    """

    def _switch(
        *,
        user_id: str,
        username: str | None = None,
        role: str = "user",
    ) -> AuthenticatedUser:
        user = AuthenticatedUser(
            id=user_id,
            username=username or user_id,
            role=role,
            display_name=username or user_id,
        )
        monkeypatch.setitem(app.dependency_overrides, get_current_user, lambda: user)
        return user

    return _switch


@pytest.fixture
def anonymous_client(api_client: TestClient, monkeypatch) -> TestClient:
    """The same client without a session, to exercise the real 401 path."""
    monkeypatch.delitem(app.dependency_overrides, get_current_user, raising=False)
    return api_client
