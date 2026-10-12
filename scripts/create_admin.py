"""Create the initial administrator account for the console.

Usage (from project root):
    python scripts/create_admin.py --username admin --password <secret> [--display-name Admin]

Skips silently when the username already exists.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app.core.security import hash_password  # noqa: E402
from app.core.settings import load_settings  # noqa: E402
from app.models import UserRecord  # noqa: E402
from app.repositories import UserRepository  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True, help="Minimum 6 characters.")
    parser.add_argument("--display-name", default=None)
    args = parser.parse_args()
    if len(args.password) < 6:
        print("Password must be at least 6 characters.")
        return 1

    settings = load_settings()
    engine = create_engine(settings.database.build_url())
    session = sessionmaker(bind=engine)()
    try:
        repository = UserRepository(session)
        if repository.get_by_username(args.username) is not None:
            print(f"User '{args.username}' already exists; nothing to do.")
            return 0
        repository.add(
            UserRecord(
                id=str(uuid4()),
                username=args.username,
                password_hash=hash_password(args.password),
                role="admin",
                display_name=args.display_name,
                is_active=True,
            )
        )
        repository.commit()
        print(f"Admin '{args.username}' created.")
        return 0
    finally:
        session.close()
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
