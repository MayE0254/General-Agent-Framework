from __future__ import annotations

from collections.abc import Generator
from typing import Any

from app.core import Settings, get_settings


def get_database_connection_config(settings: Settings | None = None) -> dict[str, object]:
    current_settings = settings or get_settings()
    database = current_settings.database

    return {
        "url": database.build_url(),
        "masked_url": database.build_url(mask_password=True),
        "echo": database.echo,
        "pool_size": database.pool_size,
        "max_overflow": database.max_overflow,
        "connect_args": database.build_connect_args(),
    }


def create_database_engine(settings: Settings | None = None) -> Any:
    current_settings = settings or get_settings()
    config = get_database_connection_config(current_settings)
    from sqlalchemy import create_engine

    return create_engine(
        config["url"],
        echo=bool(config["echo"]),
        pool_size=int(config["pool_size"]),
        max_overflow=int(config["max_overflow"]),
        connect_args=config["connect_args"],
        pool_pre_ping=True,
        future=True,
    )


def create_session_factory(engine: Any) -> Any:
    from sqlalchemy.orm import Session, sessionmaker

    return sessionmaker(
        bind=engine,
        class_=Session,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )


def get_db_session(
    session_factory: Any,
) -> Generator[Any, None, None]:
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


def probe_database_connection(engine: Any) -> tuple[bool, str]:
    from sqlalchemy import text

    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True, "Database connection probe succeeded."
    except Exception as exc:  # pragma: no cover - runtime connection boundary
        return False, f"Database connection probe failed: {exc}"


def build_database_runtime(settings: Settings | None = None) -> dict[str, Any]:
    try:
        engine = create_database_engine(settings)
        session_factory = create_session_factory(engine)
        probe_ok, probe_message = probe_database_connection(engine)
        return {
            "sqlalchemy_available": True,
            "engine": engine,
            "session_factory": session_factory,
            "probe_ok": probe_ok,
            "probe_message": probe_message,
        }
    except ModuleNotFoundError:
        return {
            "sqlalchemy_available": False,
            "engine": None,
            "session_factory": None,
            "probe_ok": False,
            "probe_message": "SQLAlchemy is not installed in the current environment.",
        }
