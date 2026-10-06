"""Tests for the SSH connect phase-timing instrumentation (DAH-2272)."""

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

import services.ssh_connect_timing as sct
from services.ssh_connect_timing import connect_with_phase_timing

MODULE_LOGGER = "services.ssh_connect_timing"

_DEFAULT_EXTRA_INFO = {
    "server_version": "SSH-2.0-OpenSSH_8.9p1 Ubuntu",
    "recv_cipher": "aes256-gcm@openssh.com",
    "recv_mac": "",
}


def _fake_conn(extra_info=None):
    info = dict(_DEFAULT_EXTRA_INFO if extra_info is None else extra_info)
    conn = MagicMock(name="ssh_conn")
    conn.close = MagicMock()
    conn.wait_closed = AsyncMock()
    conn.get_extra_info = MagicMock(side_effect=lambda k, default=None: info.get(k, default))
    return conn


class _FakeConnectCM:
    """Stand-in for asyncssh.connect()'s return (an async context manager).

    On enter it drives the SSHClient phase callbacks so the wrapper sees
    realistic marks; on exit it mirrors asyncssh cleanup (and an optional
    abnormal connection_lost).
    """

    def __init__(self, conn, client, *, fire=True, banner=None, lost_exc=None):
        self._conn = conn
        self._client = client
        self._fire = fire
        self._banner = banner
        self._lost_exc = lost_exc

    async def __aenter__(self):
        if self._fire:
            self._client.connection_made(object())  # TCP done
            self._client.begin_auth("root")          # KEX done, auth starting
            if self._banner is not None:
                self._client.auth_banner_received(self._banner, "")
            self._client.auth_completed()            # auth done
        return self._conn

    async def __aexit__(self, *_):
        if self._lost_exc is not None:
            self._client.connection_lost(self._lost_exc)
        self._conn.close()
        await self._conn.wait_closed()
        return False


def _connect_returning(conn, *, fire=True, banner=None, lost_exc=None, capture=None):
    """Build a sync fake for asyncssh.connect that returns a _FakeConnectCM."""

    def fake_connect(*, client_factory, **kwargs):
        if capture is not None:
            capture.update(kwargs)
        client = client_factory()
        return _FakeConnectCM(conn, client, fire=fire, banner=banner, lost_exc=lost_exc)

    return fake_connect


def _clock(monkeypatch, ticks):
    """Make ``now_ms`` return ``ticks`` in order."""
    it = iter(ticks)
    monkeypatch.setattr(sct, "now_ms", lambda: next(it))


def _record(caplog, message):
    for rec in caplog.records:
        if getattr(getattr(rec, "msg", None), "message", None) == message:
            return rec.msg
    return None


@pytest.fixture(autouse=True)
def _ssh_debug_logging_on(monkeypatch):
    """DAH-2272: the phase-timing line is off by default (SSH_DEBUG_LOGGING=False).

    These tests exercise the instrumentation itself, so enable it. The
    suppression test overrides this back to False.
    """
    monkeypatch.setattr(sct.settings, "SSH_DEBUG_LOGGING", True)


@pytest.mark.asyncio
async def test_logs_three_way_phase_split(monkeypatch, caplog):
    """connection_made/begin_auth/auth_completed → tcp / handshake / auth split."""
    conn = _fake_conn()
    # now_ms() order: t0, connection_made, begin_auth, auth_completed, end
    _clock(monkeypatch, [1000, 1100, 1300, 1400, 1450])
    captured = {}
    monkeypatch.setattr(sct.asyncssh, "connect", _connect_returning(conn, capture=captured))

    with caplog.at_level(logging.INFO, logger=MODULE_LOGGER):
        async with connect_with_phase_timing(
            log_extra={"miner_hotkey": "5Fabc"}, host="1.2.3.4", port=2222
        ) as client:
            assert client is conn

    rec = _record(caplog, "ssh_connect_phase_timing")
    assert rec is not None, "phase-timing log line not emitted"
    assert rec.extra["tcp_connect_ms"] == 100      # t0 -> connection_made
    assert rec.extra["ssh_handshake_ms"] == 200    # connection_made -> begin_auth
    assert rec.extra["ssh_auth_ms"] == 100         # begin_auth -> auth_completed
    assert rec.extra["ssh_login_ms"] == 300        # connection_made -> auth_completed
    assert rec.extra["total_connect_ms"] == 450
    assert rec.extra["host"] == "1.2.3.4"
    assert rec.extra["port"] == 2222
    assert rec.extra["miner_hotkey"] == "5Fabc"
    # Negotiated metadata read off the connection.
    assert rec.extra["server_version"] == "SSH-2.0-OpenSSH_8.9p1 Ubuntu"
    assert rec.extra["cipher"] == "aes256-gcm@openssh.com"

    assert captured["host"] == "1.2.3.4"
    conn.close.assert_called_once()
    conn.wait_closed.assert_awaited_once()


@pytest.mark.asyncio
async def test_auth_banner_captured(monkeypatch, caplog):
    """A server pre-auth banner is recorded; absent it, the field is omitted."""
    conn = _fake_conn()
    # extra tick for auth_banner_received between begin_auth and auth_completed
    _clock(monkeypatch, [1000, 1100, 1300, 1350, 1400, 1450])
    monkeypatch.setattr(
        sct.asyncssh, "connect", _connect_returning(conn, banner="Authorized users only")
    )

    with caplog.at_level(logging.INFO, logger=MODULE_LOGGER):
        async with connect_with_phase_timing(host="h", port=22):
            pass

    rec = _record(caplog, "ssh_connect_phase_timing")
    assert rec.extra["auth_banner"] == "Authorized users only"


@pytest.mark.asyncio
async def test_missing_marks_log_none_without_crashing(monkeypatch, caplog):
    """If the callbacks never fire, the split fields are None (not an error)."""
    conn = _fake_conn()
    _clock(monkeypatch, [1000, 1500])  # t0, end only
    monkeypatch.setattr(sct.asyncssh, "connect", _connect_returning(conn, fire=False))

    with caplog.at_level(logging.INFO, logger=MODULE_LOGGER):
        async with connect_with_phase_timing(host="h", port=22):
            pass

    rec = _record(caplog, "ssh_connect_phase_timing")
    assert rec is not None
    assert rec.extra["tcp_connect_ms"] is None
    assert rec.extra["ssh_handshake_ms"] is None
    assert rec.extra["ssh_auth_ms"] is None
    assert rec.extra["ssh_login_ms"] is None
    assert rec.extra["total_connect_ms"] == 500
    conn.wait_closed.assert_awaited_once()


@pytest.mark.asyncio
async def test_abnormal_connection_lost_warns(monkeypatch, caplog):
    """An abnormal connection_lost (exc set) emits a separate warning."""
    conn = _fake_conn()
    _clock(monkeypatch, [1000, 1100, 1300, 1400, 1450])
    exc = ConnectionResetError("keepalive timeout")
    monkeypatch.setattr(sct.asyncssh, "connect", _connect_returning(conn, lost_exc=exc))

    with caplog.at_level(logging.INFO, logger=MODULE_LOGGER):
        async with connect_with_phase_timing(log_extra={"x": 1}, host="9.9.9.9", port=22):
            pass

    lost = _record(caplog, "ssh_connection_lost")
    assert lost is not None
    assert "keepalive timeout" in lost.extra["error"]
    assert lost.extra["host"] == "9.9.9.9"


@pytest.mark.asyncio
async def test_clean_connection_lost_is_silent(monkeypatch, caplog):
    """A clean close (exc=None) must NOT emit an ssh_connection_lost line."""
    conn = _fake_conn()
    _clock(monkeypatch, [1000, 1100, 1300, 1400, 1450])
    # lost_exc stays None -> connection_lost not fired by the fake on exit
    monkeypatch.setattr(sct.asyncssh, "connect", _connect_returning(conn))

    with caplog.at_level(logging.INFO, logger=MODULE_LOGGER):
        async with connect_with_phase_timing(host="h", port=22):
            pass

    assert _record(caplog, "ssh_connection_lost") is None


@pytest.mark.asyncio
async def test_phase_timing_suppressed_when_flag_off(monkeypatch, caplog):
    """DAH-2272: with SSH_DEBUG_LOGGING off the per-connect line is NOT emitted,
    but the connection is still established/closed and abnormal drops still warn."""
    monkeypatch.setattr(sct.settings, "SSH_DEBUG_LOGGING", False)
    conn = _fake_conn()
    _clock(monkeypatch, [1000, 1100, 1300, 1400, 1450])
    exc = ConnectionResetError("keepalive timeout")
    monkeypatch.setattr(sct.asyncssh, "connect", _connect_returning(conn, lost_exc=exc))

    with caplog.at_level(logging.INFO, logger=MODULE_LOGGER):
        async with connect_with_phase_timing(host="h", port=22) as client:
            assert client is conn

    # The per-connection phase-timing line is gated off ...
    assert _record(caplog, "ssh_connect_phase_timing") is None
    # ... but the abnormal-drop warning is independent of the flag and still fires.
    assert _record(caplog, "ssh_connection_lost") is not None
    conn.close.assert_called_once()
    conn.wait_closed.assert_awaited_once()


@pytest.mark.asyncio
async def test_custom_client_factory_rejected(monkeypatch):
    """A caller-supplied client_factory is rejected rather than silently dropped."""
    monkeypatch.setattr(
        sct.asyncssh, "connect", AsyncMock(side_effect=AssertionError("must not connect"))
    )
    with pytest.raises(ValueError, match="custom client_factory"):
        async with connect_with_phase_timing(host="h", client_factory=lambda: None):
            pass


@pytest.mark.asyncio
async def test_connect_failure_propagates_and_does_not_log(monkeypatch, caplog):
    """If asyncssh.connect raises, the error propagates and nothing is logged."""
    _clock(monkeypatch, [1000])  # only t0 is consumed

    def boom(*, client_factory, **kwargs):
        raise ConnectionResetError("network")

    monkeypatch.setattr(sct.asyncssh, "connect", boom)

    with caplog.at_level(logging.INFO, logger=MODULE_LOGGER):
        with pytest.raises(ConnectionResetError):
            async with connect_with_phase_timing(host="h", port=22):
                pass

    assert _record(caplog, "ssh_connect_phase_timing") is None


def test_apply_asyncssh_log_level_off(monkeypatch):
    """Flag off -> asyncssh logger WARNING, debug level not raised."""
    from core import utils

    monkeypatch.setattr(utils.settings, "SSH_DEBUG_LOGGING", False)
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

    monkeypatch.setattr(utils.settings, "SSH_DEBUG_LOGGING", True)
    set_debug = MagicMock()
    monkeypatch.setattr(utils.asyncssh, "set_debug_level", set_debug)

    lg = logging.getLogger("asyncssh-test-on")
    level = utils._apply_asyncssh_log_level(lg)

    assert level == logging.DEBUG
    assert lg.level == logging.DEBUG
    set_debug.assert_called_once_with(2)
