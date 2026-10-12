"""Connectivity check for the remote Milvus (vector) + Neo4j (graph) backends.

Reads host/port/collection from ``configs/application.toml`` and injects the
User-scope password env vars (MILVUS_PASSWORD / NEO4J_PASSWORD) into the
process so the ``${VAR}`` references in the config resolve (the current shell
often does not inherit User-scope vars set via ``setx``).

Run from the repo root::

    .venv\\Scripts\\python scripts\\check_knowledge_conn.py

Exit 0 only when both backends are reachable and authenticated.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _inject_user_env(name: str) -> None:
    """Promote a User-scope env var into this process when not already set."""
    if os.environ.get(name):
        return
    try:
        import subprocess
        probe = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f"[Environment]::GetEnvironmentVariable('{name}','User')"],
            capture_output=True, text=True, timeout=15,
        )
        if probe.returncode == 0 and probe.stdout.strip():
            os.environ[name] = probe.stdout.strip()
    except Exception:
        pass


def _load_settings():
    from app.core import load_settings

    for var in ("MILVUS_PASSWORD", "NEO4J_PASSWORD"):
        _inject_user_env(var)
    return load_settings(ROOT / "configs" / "application.toml")


def check_milvus(settings) -> tuple[bool, str]:
    host = settings.knowledge.milvus.host
    port = settings.knowledge.milvus.port
    password = settings.knowledge.milvus.password
    user = settings.knowledge.milvus.user
    collection = settings.knowledge.milvus.collection
    try:
        from pymilvus import MilvusClient  # type: ignore
    except Exception as exc:  # pragma: no cover
        return False, f"pymilvus import failed: {exc}"

    kwargs: dict = {"uri": f"http://{host}:{port}"}
    if user:
        kwargs["user"], kwargs["password"] = user, password
    elif password:
        kwargs["token"] = f"root:{password}"  # default Milvus root account
    try:
        client = MilvusClient(**kwargs)
        # MilvusClient is lazy; force a round-trip by listing collections/tables.
        tables = client.list_collections()
        exists = tables if isinstance(tables, list) else list(tables)
        if collection in exists:
            return True, f"OK host={host}:{port} collections={len(exists)} (contains '{collection}')"
        return True, f"OK host={host}:{port} collections={len(exists)} (missing '{collection}')"
    except Exception as exc:
        return False, f"FAIL host={host}:{port} -> {type(exc).__name__}: {exc}"


def check_neo4j(settings) -> tuple[bool, str]:
    uri = settings.knowledge.graph.uri
    user = settings.knowledge.graph.user
    password = settings.knowledge.graph.password
    database = settings.knowledge.graph.database
    try:
        from neo4j import GraphDatabase  # type: ignore
    except Exception as exc:  # pragma: no cover
        return False, f"neo4j import failed: {exc}"

    driver = None
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password), connection_timeout=10)
        driver.verify_connectivity()
        with driver.session(database=database) as session:
            record = session.run("RETURN 1 AS ok").single()
            ok = bool(record and record["ok"] == 1)
        return (True, f"OK uri={uri} user={user} db={database}") if ok else (
            False, f"WRONG-RESULT uri={uri}")
    except Exception as exc:
        return False, f"FAIL uri={uri} -> {type(exc).__name__}: {exc}"
    finally:
        if driver is not None:
            driver.close()


def main() -> int:
    settings = _load_settings()
    problems: list[str] = []

    milvus_ok, milvus_msg = check_milvus(settings)
    print(f"[milvus ] {milvus_msg}")
    if not milvus_ok:
        problems.append(milvus_msg)

    graph_ok, graph_msg = check_neo4j(settings)
    print(f"[neo4j  ] {graph_msg}")
    if not graph_ok:
        problems.append(graph_msg)

    if problems:
        print("\nConnectivity problems:")
        for p in problems:
            print(f"  - {p}")
        print("knowledge-connectivity: FAIL")
        return 1
    print("knowledge-connectivity: OK (both backends reachable)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())