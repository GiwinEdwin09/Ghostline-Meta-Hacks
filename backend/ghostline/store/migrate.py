"""Apply SQL migrations in order (build plan Part 27: run from CI or by hand).

    uv run python -m ghostline.store.migrate            # uses DATABASE_URL
    uv run python -m ghostline.store.migrate --dir backend/migrations

Each file runs in its own transaction and is recorded in schema_migrations. A transaction-scoped
advisory lock keeps two runners from applying the same file, and works through pgbouncer.
"""

import argparse
import os
import sys
from pathlib import Path

import psycopg

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
_LOCK_KEY = "ghostline:migrate"


def migration_files(directory: Path = MIGRATIONS_DIR) -> list[Path]:
    return sorted(directory.glob("*.sql"))


def apply_migrations(dsn: str, directory: Path = MIGRATIONS_DIR) -> list[str]:
    """Apply every file not yet recorded. Returns the versions applied by this call."""
    applied: list[str] = []
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(
            "create table if not exists schema_migrations ("
            "version text primary key, applied_at timestamptz not null default now())"
        )
        for path in migration_files(directory):
            version = path.stem
            with conn.transaction():
                conn.execute("select pg_advisory_xact_lock(hashtextextended(%s, 0))", (_LOCK_KEY,))
                done = conn.execute(
                    "select 1 from schema_migrations where version = %s", (version,)
                ).fetchone()
                if done:
                    continue
                conn.execute(path.read_text())
                conn.execute("insert into schema_migrations (version) values (%s)", (version,))
                applied.append(version)
    return applied


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, default=MIGRATIONS_DIR)
    args = parser.parse_args(argv)
    dsn = os.environ.get("DATABASE_URL", "")
    if not dsn:
        print("DATABASE_URL is not set", file=sys.stderr)
        return 1
    applied = apply_migrations(dsn, args.dir)
    print("applied: " + ", ".join(applied) if applied else "already up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
