import psycopg
import pytest

from ghostline.store.migrate import apply_migrations, migration_files

EXPECTED_TABLES = {
    "experiments",
    "trials",
    "tracks",
    "trajectories",
    "models",
    "fits",
    "reviews",
    "ghost_renders",
    "events",
    "llm_calls",
    "jobs",
    "benchmark",
    "uploads",
    "spend_ledger",
    "schema_migrations",
}


def test_migration_files_are_numbered_in_order():
    names = [p.stem for p in migration_files()]

    assert names, "no migrations found"
    assert names == sorted(names)
    assert all(name[:4].isdigit() for name in names)


def test_migrations_create_every_table(pg_dsn):
    with psycopg.connect(pg_dsn) as conn:
        rows = conn.execute(
            "select tablename from pg_tables where schemaname = 'public'"
        ).fetchall()

    assert EXPECTED_TABLES <= {r[0] for r in rows}


def test_migrations_are_idempotent(pg_dsn):
    assert apply_migrations(pg_dsn) == []


def test_row_level_security_is_on_for_every_app_table(pg_dsn):
    with psycopg.connect(pg_dsn) as conn:
        rows = conn.execute(
            "select relname from pg_class where relkind = 'r'"
            " and relnamespace = 'public'::regnamespace and not relrowsecurity"
        ).fetchall()

    assert {r[0] for r in rows} <= {"schema_migrations"}


def _experiment(conn) -> str:
    return conn.execute("insert into experiments (title) values ('t') returning id").fetchone()[0]


def test_events_cannot_be_updated(pg_dsn):
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        exp = _experiment(conn)
        conn.execute(
            "insert into events (experiment_id, seq, actor, kind, title)"
            " values (%s, 1, 'system', 'observation', 'first')",
            (exp,),
        )

        with pytest.raises(psycopg.errors.RaiseException, match="append-only"):
            conn.execute("update events set title = 'edited' where experiment_id = %s", (exp,))


def test_event_seq_is_unique_per_experiment(pg_dsn):
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        exp = _experiment(conn)
        insert = (
            "insert into events (experiment_id, seq, actor, kind, title)"
            " values (%s, 1, 'system', 'observation', 'x')"
        )
        conn.execute(insert, (exp,))

        with pytest.raises(psycopg.errors.UniqueViolation):
            conn.execute(insert, (exp,))


def test_only_one_active_job_per_experiment(pg_dsn):
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        exp = _experiment(conn)
        conn.execute(
            "insert into jobs (experiment_id, kind) values (%s, 'analyze_experiment')", (exp,)
        )

        with pytest.raises(psycopg.errors.UniqueViolation):
            conn.execute("insert into jobs (experiment_id, kind) values (%s, 'revise')", (exp,))

        conn.execute("update jobs set status = 'failed' where experiment_id = %s", (exp,))
        conn.execute("insert into jobs (experiment_id, kind) values (%s, 'revise')", (exp,))


def test_deleting_an_experiment_cascades_but_keeps_the_spend_ledger(pg_dsn):
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        exp = _experiment(conn)
        conn.execute(
            "insert into events (experiment_id, seq, actor, kind, title)"
            " values (%s, 1, 'system', 'observation', 'x')",
            (exp,),
        )
        conn.execute(
            "insert into spend_ledger (experiment_id, provider, amount_usd)"
            " values (%s, 'meta', 1.5)",
            (exp,),
        )

        conn.execute("delete from experiments where id = %s", (exp,))

        assert conn.execute("select count(*) from events").fetchone()[0] == 0
        assert conn.execute("select sum(amount_usd) from spend_ledger").fetchone()[0] == 1.5
