from pathlib import Path

from app.core import load_settings
from app.infra import ResourceStatus, close_app_resources, initialize_app_resources


CONFIG_PATH = Path("configs/application.toml")


def test_initialize_app_resources_prepares_runtime_handles(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-env-key")

    settings = load_settings(CONFIG_PATH)

    resources = initialize_app_resources(settings)

    assert resources.database.status in {
        ResourceStatus.READY,
        ResourceStatus.DEGRADED,
    }
    assert resources.redis.status == ResourceStatus.READY
    assert resources.llm.status == ResourceStatus.READY
    assert resources.vectorstores["pgvector"].status == ResourceStatus.READY
    assert resources.vectorstores["milvus"].status == ResourceStatus.READY


def test_close_app_resources_marks_handles_closed(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-env-key")

    settings = load_settings(CONFIG_PATH)
    resources = initialize_app_resources(settings)

    closed_resources = close_app_resources(resources)

    assert closed_resources.database.status == ResourceStatus.CLOSED
    assert closed_resources.redis.status == ResourceStatus.CLOSED
    assert closed_resources.llm.status == ResourceStatus.CLOSED
    assert closed_resources.vectorstores["pgvector"].status == ResourceStatus.CLOSED
    assert closed_resources.stopped_at is not None
