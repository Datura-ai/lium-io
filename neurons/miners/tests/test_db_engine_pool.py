"""The miner's engine replaces a pooled connection that died while idle, and retires one after 30 min.

Regression (DAH-3632): ``core/db.py`` built the engine with ``create_engine(URI)`` and nothing else, so a connection
Postgres or a NAT closed while it sat idle in the pool came back as-is and the next query failed with
``OperationalError``. The backend, the portal and the validator all carry ``pool_pre_ping`` and ``pool_recycle``;
the miner did not. Each test drives the real pool through ``make_engine`` on a SQLite file: with ``make_engine`` reduced
to ``create_engine(uri)`` the first raises ``ProgrammingError: Cannot operate on a closed database`` and the second hands
back the same connection.
"""

from core.db import make_engine


def test_a_connection_that_died_while_idle_in_the_pool_is_replaced_not_handed_out(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path / 'pool.db'}")
    with engine.connect() as conn:
        assert conn.exec_driver_sql("SELECT 1").scalar() == 1
        raw = conn.connection.dbapi_connection
    raw.close()  # the server side goes away while the connection sits idle in the pool

    with engine.connect() as conn:  # without pool_pre_ping: sqlalchemy.exc.ProgrammingError — the dead connection was handed out
        assert conn.exec_driver_sql("SELECT 1").scalar() == 1
        assert conn.connection.dbapi_connection is not raw


def test_a_connection_older_than_thirty_minutes_is_retired_on_checkout(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path / 'pool.db'}")
    with engine.connect() as conn:
        first = conn.connection.dbapi_connection
        record = conn.connection._connection_record
    record.starttime -= 30 * 60 + 1  # opened just over the window ago

    with engine.connect() as conn:  # without pool_recycle (-1, "never") the same connection comes back
        assert conn.connection.dbapi_connection is not first


def test_a_connection_younger_than_thirty_minutes_is_reused(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path / 'pool.db'}")
    with engine.connect() as conn:
        first = conn.connection.dbapi_connection
        record = conn.connection._connection_record
    record.starttime -= 29 * 60

    with engine.connect() as conn:
        assert conn.connection.dbapi_connection is first
