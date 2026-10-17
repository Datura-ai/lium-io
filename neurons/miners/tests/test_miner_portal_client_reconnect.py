import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

import clients.miner_portal_client as portal_client
from clients.miner_portal_client import MinerPortalClient


class _Stop(BaseException):
    pass


class _ClosedConnection:
    """A connection the portal closes at once: no messages."""

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


class _Connect:
    """Like websockets.connect: usable once with async-with, or as an iterator that reconnects with the same
    headers (two connections, then stop)."""

    def __init__(self):
        self.reconnects = 2

    async def __aenter__(self):
        return _ClosedConnection()

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.reconnects:
            raise _Stop
        self.reconnects -= 1
        return _ClosedConnection()


def test_each_reconnect_signs_a_fresh_handshake(monkeypatch):
    client = object.__new__(MinerPortalClient)
    client.logging_extra = {}
    client.handle_send_messages = AsyncMock()
    connects = Mock(side_effect=[_Connect(), _Connect(), _Stop()])
    client.connect = connects
    monkeypatch.setattr(portal_client, "CONNECT_INTERVAL", 0)

    with pytest.raises(_Stop):
        asyncio.run(client.run_forever())

    assert connects.call_count == 3


def test_connect_failures_back_off_and_reset_after_a_connection(monkeypatch):
    client = object.__new__(MinerPortalClient)
    client.logging_extra = {}
    client.handle_send_messages = AsyncMock()
    client.connect = Mock(side_effect=[OSError(), OSError(), _Connect(), OSError(), _Stop()])
    sleeps = AsyncMock()
    monkeypatch.setattr(portal_client.asyncio, "sleep", sleeps)
    monkeypatch.setattr(portal_client.random, "uniform", lambda low, high: high)

    with pytest.raises(_Stop):
        asyncio.run(client.run_forever())

    assert [call.args[0] for call in sleeps.await_args_list] == [2, 4, portal_client.CONNECT_INTERVAL, 2]
