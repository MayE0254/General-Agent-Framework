from pathlib import Path

from app.core import Settings, load_settings
from app.infra import (
    get_celery_redis_urls,
    get_database_connection_config,
    get_litellm_config,
    get_redis_connection_config,
    get_vectorstore_configs,
)


CONFIG_PATH = Path("configs/application.toml")


def test_load_settings_from_configs_directory(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-env-key")
    monkeypatch.setenv("REDIS_USERNAME", "multiagent")
    monkeypatch.setenv("REDIS_PASSWORD", "redis-test-password")

    settings = load_settings(CONFIG_PATH)

    assert settings.app.name == "multiagent"
    assert settings.database.host == "127.0.0.1"
    assert settings.knowledge.milvus.enabled is True


def test_database_and_redis_config_builders(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-env-key")
    monkeypatch.setenv("REDIS_USERNAME", "multiagent")
    monkeypatch.setenv("REDIS_PASSWORD", "redis-test-password")

    settings = load_settings(CONFIG_PATH)

    database_config = get_database_connection_config(settings)
    redis_config = get_redis_connection_config(settings)
    celery_config = get_celery_redis_urls(settings)

    assert database_config["url"] == (
        f"postgresql+psycopg://multiagent_user:{settings.database.password}@127.0.0.1:5432/multiagent_db"
    )
    assert database_config["masked_url"] == (
        "postgresql+psycopg://multiagent_user:***@127.0.0.1:5432/multiagent_db"
    )
    assert (
        redis_config["url"]
        == f"redis://{settings.redis.username}:{settings.redis.password}@"
        f"{settings.redis.host}:{settings.redis.port}/{settings.redis.db}"
    )
    assert celery_config["broker_url"] == settings.redis.build_url(
        database_index=settings.celery.broker_db
    )
    assert celery_config["result_backend"] == settings.redis.build_url(
        database_index=settings.celery.result_db
    )
    assert database_config["masked_url"].endswith("@127.0.0.1:5432/multiagent_db")


def test_llm_and_vectorstore_configs(monkeypatch) -> None:
    # The [llm] api_key references ${DEEPSEEK_API_KEY}; make the reference
    # resolvable and assert it lands in the built config.
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-env-key")
    monkeypatch.setenv("REDIS_USERNAME", "multiagent")
    monkeypatch.setenv("REDIS_PASSWORD", "redis-test-password")

    settings = load_settings(CONFIG_PATH)

    llm_config = get_litellm_config(settings)
    vectorstore_configs = get_vectorstore_configs(settings)

    assert llm_config["provider"] == "deepseek"
    assert llm_config["model"] == "deepseek/deepseek-chat"
    assert llm_config["api_base"] == "https://api.deepseek.com"
    assert llm_config["api_key"] == "sk-test-env-key"
    assert vectorstore_configs["pgvector"]["collection"] == "knowledge_chunks"
    assert vectorstore_configs["milvus"]["port"] == 19530


def test_llm_providers_and_pricing_parse_from_config(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-env-key")
    monkeypatch.setenv("REDIS_USERNAME", "multiagent")
    monkeypatch.setenv("REDIS_PASSWORD", "redis-test-password")

    settings = load_settings(CONFIG_PATH)

    # Named provider routing table: name -> api_base/api_key/model.
    assert settings.llm.providers["deepseek"].model == "deepseek/deepseek-chat"
    assert settings.llm.providers["deepseek"].api_base == "https://api.deepseek.com"
    assert settings.llm.providers["deepseek"].api_key == "sk-test-env-key"

    # Per-model pricing table used by estimate_cost_usd().
    price = settings.llm.pricing["deepseek/deepseek-chat"]
    assert price.input_per_million > 0
    assert price.output_per_million > 0


def test_env_reference_fails_fast_when_missing(monkeypatch) -> None:
    # Unresolvable ${VAR} references must fail loudly instead of silently
    # degrading (strict mode: no mock/downgrade fallbacks).
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("REDIS_USERNAME", "multiagent")
    monkeypatch.setenv("REDIS_PASSWORD", "redis-test-password")

    import pytest

    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        load_settings(CONFIG_PATH)


def test_redis_password_only_mode_still_works() -> None:
    settings = Settings()
    settings.redis.host = "10.0.0.5"
    settings.redis.port = 7000
    settings.redis.password = "secret"

    redis_config = get_redis_connection_config(settings)
    celery_config = get_celery_redis_urls(settings)

    assert redis_config["url"] == "redis://:secret@10.0.0.5:7000/0"
    assert celery_config["broker_url"] == "redis://:secret@10.0.0.5:7000/1"
    assert celery_config["result_backend"] == "redis://:secret@10.0.0.5:7000/2"
