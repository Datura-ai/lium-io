"""Tests for the SSH connect phase-timing instrumentation (DAH-2272)."""

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

import services.ssh_connect_timing as sct
from services.ssh_connect_timing import connect_with_phase_timing

MODULE_LOGGER = "services.ssh_connect_timing"


def _fake_conn():
    conn = MagicMock(name="ssh_conn")
    conn.close = MagicMock()
    conn.wait_closed = AsyncMock()
    return conn


class _FakeConnectCM:
    """Stand-in for asyncssh.connect()'s return (an async context manager).

    On enter it optionally drives the phase callbacks on the injected client so
    the wrapper sees realistic marks; on exit it mirrors asyncssh cleanup.
    """

    def __init__(self, conn, client, fire_callbacks=True):
        self._conn = conn
        self._client = client
        self._fire = fire_callbacks

    async def __aenter__(self):
        if self._fire:
            self._client.connection_made(object())  # TCP done
            self._client.auth_completed()            # banner+kex+auth done
        return self._conn

    async def __aexit__(self, *_):
        self._conn.close()
        await self._conn.wait_closed()
        return False


def _connect_returning(conn, *, fire_callbacks=True, capture=None):
    """Build a sync fake for asyncssh.connect that returns a _FakeConnectCM."""

    def fake_connect(*, client_factory, **kwargs):
        if capture is not None:
            capture.update(kwargs)
        client = client_factory()
        return _FakeConnectCM(conn, client, fire_callbacks=fire_callbacks)

    return fake_connect


def _clock(monkeypatch, ticks):
    """Make ``now_ms`` return ``ticks`` in order."""
    it = iter(ticks)
    monkeypatch.setattr(sct, "now_ms", lambda: next(it))


def _phase_record(caplog):
    for rec in caplog.records:
        msg = getattr(rec, "msg", None)
        if getattr(msg, "message", None) == "ssh_connect_phase_timing":
            return msg
    return None


@pytest.mark.asyncio
async def test_logs_tcp_and_login_split(monkeypatch, caplog):
    """connection_made/auth_completed marks become tcp_connect_ms / ssh_login_ms."""
    conn = _fake_conn()
    # now_ms() order: t0, (connection_made), (auth_completed), end
    _clock(monkeypatch, [1000, 1100, 1400, 1450])
    captured = {}
    monkeypatch.setattr(sct.asyncssh, "connect", _connect_returning(conn, capture=captured))

    with caplog.at_level(logging.INFO, logger=MODULE_LOGGER):
        async with connect_with_phase_timing(
            log_extra={"miner_hotkey": "5Fabc"},
            host="1.2.3.4",
            port=2222,
        ) as client:
            assert client is conn

    rec = _phase_record(caplog)
    assert rec is not None, "phase-timing log line not emitted"
    assert rec.extra["tcp_connect_ms"] == 100
    assert rec.extra["ssh_login_ms"] == 300
    assert rec.extra["total_connect_ms"] == 450
    assert rec.extra["host"] == "1.2.3.4"
    assert rec.extra["port"] == 2222
    assert rec.extra["miner_hotkey"] == "5Fabc"

    # host/port were forwarded verbatim to asyncssh.connect.
    assert captured["host"] == "1.2.3.4"
    assert captured["port"] == 2222
    # Connection is closed on context exit, exactly like `async with asyncssh.connect`.
    conn.close.assert_called_once()
    conn.wait_closed.assert_awaited_once()


@pytest.mark.asyncio
async def test_missing_marks_log_none_without_crashing(monkeypatch, caplog):
    """If the callbacks never fire, the split fields are None (not an error)."""
    conn = _fake_conn()
    _clock(monkeypatch, [1000, 1500])  # t0, end only
    monkeypatch.setattr(
        sct.asyncssh, "connect", _connect_returning(conn, fire_callbacks=False)
    )

    with caplog.at_level(logging.INFO, logger=MODULE_LOGGER):
        async with connect_with_phase_timing(host="h", port=22):
            pass

    rec = _phase_record(caplog)
    assert rec is not None
    assert rec.extra["tcp_connect_ms"] is None
    assert rec.extra["ssh_login_ms"] is None
    assert rec.extra["total_connect_ms"] == 500
    conn.wait_closed.assert_awaited_once()


@pytest.mark.asyncio
async def test_custom_client_factory_rejected(monkeypatch):
    """A caller-supplied client_factory is rejected rather than silently dropped."""
    # asyncssh.connect must not even be reached.
    monkeypatch.setattr(
        sct.asyncssh, "connect", AsyncMock(side_effect=AssertionError("must not connect"))
    )
    with pytest.raises(ValueError, match="custom client_factory"):
        async with connect_with_phase_timing(host="h", client_factory=lambda: None):
            pass


@pytest.mark.asyncio
async def test_connect_failure_propagates_and_does_not_log(monkeypatch, caplog):
    """If asyncssh.connect raises, the error propagates and nothing is logged/closed."""
    _clock(monkeypatch, [1000])  # only t0 is consumed

    def boom(*, client_factory, **kwargs):
        raise ConnectionResetError("network")

    monkeypatch.setattr(sct.asyncssh, "connect", boom)

    with caplog.at_level(logging.INFO, logger=MODULE_LOGGER):
        with pytest.raises(ConnectionResetError):
            async with connect_with_phase_timing(host="h", port=22):
                pass

    assert _phase_record(caplog) is None


def test_apply_asyncssh_log_level_off(monkeypatch):
    """Flag off -> asyncssh logger WARNING, debug level not raised."""
    from core import utils

    monkeypatch.setattr(utils.settings.debug, "SSH_DEBUG_LOGGING", False)
    set_debug = MagicMock()
    monkeypatch.setattr(utils.asyncssh, "set_debug_level", set_debug)

    lg = logging.getLogger("asyncssh-test-off")
    level = utils._apply_asyncssh_log_level(lg)

    assert level == logging.WARNING
    assert lg.level == logging.WARNING
    set_debug.assert_not_called()


def test_apply_asyncssh_log_level_on(monkeypatch):
    """Flag on -> asyncssh logger DEBUG, debug level 2 (handshake, no packet dumps)."""
    from core import utils

    monkeypatch.setattr(utils.settings.debug, "SSH_DEBUG_LOGGING", True)
    set_debug = MagicMock()
    monkeypatch.setattr(utils.asyncssh, "set_debug_level", set_debug)

    lg = logging.getLogger("asyncssh-test-on")
    level = utils._apply_asyncssh_log_level(lg)

    assert level == logging.DEBUG
    assert lg.level == logging.DEBUG
    set_debug.assert_called_once_with(2)
