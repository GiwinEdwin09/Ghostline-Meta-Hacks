"""Postgres connection pool and JSON adaptation (build plan Part 22)."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import orjson
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool
from pydantic import BaseModel


def _default(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


def _dumps(obj: Any) -> str:
    # orjson writes NaN and infinity as null, which Postgres jsonb requires, and handles numpy.
    return orjson.dumps(obj, default=_default, option=orjson.OPT_SERIALIZE_NUMPY).decode()


def J(obj: Any) -> Jsonb:
    """Wrap a Python value so psycopg stores it as jsonb."""
    return Jsonb(obj, dumps=_dumps)


class DatabaseNotConfigured(RuntimeError):
    pass


class Database:
    """A small sync pool. Supabase's pooled endpoint runs pgbouncer in transaction mode, so
    prepared statements are off and nothing relies on session state (locks are xact-scoped)."""

    def __init__(self, dsn: str, *, min_size: int = 1, max_size: int = 5):
        if not dsn:
            raise DatabaseNotConfigured("DATABASE_URL is not set")
        self.pool = ConnectionPool(
            dsn,
            min_size=min_size,
            max_size=max_size,
            kwargs={"row_factory": dict_row, "prepare_threshold": None, "autocommit": False},
            open=True,
        )

    @contextmanager
    def connection(self) -> Iterator[psycopg.Connection]:
        """A pooled connection. Commits on success and rolls back on an exception."""
        with self.pool.connection() as conn:
            yield conn

    def execute(self, sql: str, params: Any = None) -> None:
        with self.connection() as conn:
            conn.execute(sql, params)

    def fetch_one(self, sql: str, params: Any = None) -> dict[str, Any] | None:
        with self.connection() as conn:
            return conn.execute(sql, params).fetchone()

    def fetch_all(self, sql: str, params: Any = None) -> list[dict[str, Any]]:
        with self.connection() as conn:
            return conn.execute(sql, params).fetchall()

    def ping(self) -> bool:
        try:
            self.fetch_one("select 1 as ok")
            return True
        except Exception:
            return False

    def close(self) -> None:
        self.pool.close()


class BoundDatabase:
    """The Database interface over one open connection, so several repo calls share a
    transaction. The caller that opened the connection commits or rolls back."""

    def __init__(self, conn: psycopg.Connection):
        self._conn = conn

    @contextmanager
    def connection(self) -> Iterator[psycopg.Connection]:
        yield self._conn

    def execute(self, sql: str, params: Any = None) -> None:
        self._conn.execute(sql, params)

    def fetch_one(self, sql: str, params: Any = None) -> dict[str, Any] | None:
        return self._conn.execute(sql, params).fetchone()

    def fetch_all(self, sql: str, params: Any = None) -> list[dict[str, Any]]:
        return self._conn.execute(sql, params).fetchall()
