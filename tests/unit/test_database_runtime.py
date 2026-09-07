from pathlib import Path

from app.core import load_settings
from app.infra.database import build_database_runtime


CONFIG_PATH = Path("configs/application.toml")


def test_build_database_runtime_returns_runtime_shape(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-env-key")

    settings = load_settings(CONFIG_PATH)

    runtime = build_database_runtime(settings)

    assert "sqlalchemy_available" in runtime
    assert "engine" in runtime
    assert "session_factory" in runtime
    assert "probe_ok" in runtime
    assert "probe_message" in runtime

    if runtime["sqlalchemy_available"]:
        assert runtime["engine"] is not None
        assert runtime["session_factory"] is not None
    else:
        assert runtime["engine"] is None
        assert runtime["session_factory"] is None
