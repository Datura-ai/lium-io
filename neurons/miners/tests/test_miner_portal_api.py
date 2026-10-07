"""
Tests for the MinerPortalAPI snapshot cache (DAH-2469).

The cache must collapse the validator wave into one bulk portal request,
serve stale data when a refresh fails, back off after a failed refresh,
and filter by executor_id locally.
"""

import asyncio
from unittest.mock import Mock

import aiohttp
import bittensor
import pytest

from clients.miner_portal_api import (
    MinerPortalAPI,
)
from core.config import settings

SNAPSHOT = {
    "hotkey-a": [
        {"id": "aaaa-1", "validator_hotkey": "v1"},
        {"id": "aaaa-2", "validator_hotkey": "v1"},
    ],
    "hotkey-b": [{"id": "bbbb-1", "validator_hotkey": "v1"}],
}


@pytest.fixture(autouse=True)
def reset_cache():
    # fresh lock per test: asyncio primitives bind to the first loop that uses them
    MinerPortalAPI._snapshot = {}
    MinerPortalAPI._snapshot_fetched_at = None
    MinerPortalAPI._last_refresh_attempt_at = None
    MinerPortalAPI._refresh_lock = asyncio.Lock()
    yield


class _FakeResponse:
    def __init__(self, status: int, json_body=None, text_body: str = ""):
        self.status = status
        self._json_body = json_body
        self._text_body = text_body

    async def json(self):
        return self._json_body

    async def text(self):
        return self._text_body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class _FakeSession:
    last_url: str | None = None
    last_headers: dict | None = None

    def __init__(self, response: _FakeResponse):
        self._response = response

    def get(self, url, headers=None):
        _FakeSession.last_url = url
        _FakeSession.last_headers = headers
        return self._response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


@pytest.fixture
def portal_http(monkeypatch):
    # replaces the wallet and aiohttp session so _fetch_bulk_snapshot runs for real
    keypair = bittensor.Keypair.create_from_uri("//LiumTestMiner")
    fake_wallet = Mock()
    fake_wallet.get_hotkey.return_value = keypair
    monkeypatch.setattr(type(settings), "get_bittensor_wallet", lambda self: fake_wallet)

    def install_response(response: _FakeResponse):
        monkeypatch.setattr(aiohttp, "ClientSession", lambda timeout: _FakeSession(response))

    return install_response


async def test_bulk_fetch_sends_signature_headers_and_returns_snapshot(portal_http):
    portal_http(_FakeResponse(status=200, json_body=SNAPSHOT))

    snapshot = await MinerPortalAPI._fetch_bulk_snapshot()

    assert snapshot == SNAPSHOT
    assert _FakeSession.last_url.endswith("/miners/executors")
    assert set(_FakeSession.last_headers) == {"hotkey", "timestamp", "signature"}
