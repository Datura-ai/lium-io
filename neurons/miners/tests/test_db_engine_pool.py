"""The miner's engine checks a pooled connection before it hands it out and retires it after 30 min.

Regression (DAH-3632): ``core/db.py`` built the engine with ``create_engine(URI)`` and nothing else, so a
connection Postgres or a NAT closed while it sat idle in the pool came back as-is and the next query failed
with ``OperationalError``. The backend, the portal and the validator all carry ``pool_pre_ping`` and
``pool_recycle``; the miner did not. On ``main`` before the fix both assertions below fail
(``_pre_ping`` is ``False`` and ``_recycle`` is ``-1``, SQLAlchemy's "never").
"""

from core import db as db_module


def test_engine_pings_a_pooled_connection_before_handing_it_out():
    assert db_module.engine.pool._pre_ping is True


def test_engine_retires_a_pooled_connection_after_thirty_minutes():
    assert db_module.POOL_RECYCLE_SECONDS == 1800
    assert db_module.engine.pool._recycle == 1800
