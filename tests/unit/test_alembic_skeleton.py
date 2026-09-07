from pathlib import Path


def test_alembic_skeleton_files_exist() -> None:
    assert Path("alembic.ini").exists()
    assert Path("migrations/env.py").exists()
    assert Path("migrations/script.py.mako").exists()
    assert Path("migrations/versions/README.md").exists()


def test_alembic_env_uses_project_settings() -> None:
    env_content = Path("migrations/env.py").read_text(encoding="utf-8")

    assert "get_settings()" in env_content
    assert "settings.database.build_url()" in env_content
