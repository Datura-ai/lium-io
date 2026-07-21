"""
Tests for the MinerPortalAPI snapshot cache (DAH-2469).

The cache must collapse the validator wave into one bulk portal request,
serve stale data when a refresh fails, back off after a failed refresh,
and filter by executor_id locally.
"""

import asyncio
from unittest.mock import AsyncMock, Mock

import aiohttp
import bittensor
import pytest

from clients.miner_portal_api import (
    FAILED_REFRESH_RETRY_SECONDS,
    SNAPSHOT_TTL_SECONDS,
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
    MinerPortalAPI._last_refresh_attempt_at = 0.0
    MinerPortalAPI._refresh_lock = asyncio.Lock()
    yield


def _expire_snapshot():
    MinerPortalAPI._snapshot_fetched_at -= SNAPSHOT_TTL_SECONDS + 1
    MinerPortalAPI._last_refresh_attempt_at = 0.0


async def test_one_bulk_call_serves_all_hotkeys(monkeypatch):
    bulk = AsyncMock(return_value=SNAPSHOT)
    monkeypatch.setattr(MinerPortalAPI, "_fetch_bulk_snapshot", bulk)

    executors_a = await MinerPortalAPI.fetch_executors("hotkey-a", None)
    executors_b = await MinerPortalAPI.fetch_executors("hotkey-b", None)
    executors_unknown = await MinerPortalAPI.fetch_executors("hotkey-x", None)

    assert executors_a == SNAPSHOT["hotkey-a"]
    assert executors_b == SNAPSHOT["hotkey-b"]
    assert executors_unknown == []
    assert bulk.await_count == 1


async def test_concurrent_calls_share_one_refresh(monkeypatch):
    async def slow_bulk():
        await asyncio.sleep(0.05)
        return SNAPSHOT

    bulk = AsyncMock(side_effect=slow_bulk)
    monkeypatch.setattr(MinerPortalAPI, "_fetch_bulk_snapshot", bulk)

    results = await asyncio.gather(
        *[MinerPortalAPI.fetch_executors("hotkey-a", None) for _ in range(10)]
    )

    assert bulk.await_count == 1
    assert all(result == SNAPSHOT["hotkey-a"] for result in results)


async def test_expired_snapshot_is_fully_replaced(monkeypatch):
    bulk = AsyncMock(return_value=SNAPSHOT)
    monkeypatch.setattr(MinerPortalAPI, "_fetch_bulk_snapshot", bulk)
    await MinerPortalAPI.fetch_executors("hotkey-a", None)

    _expire_snapshot()
    bulk.return_value = {"hotkey-a": [{"id": "aaaa-3", "validator_hotkey": "v1"}]}

    executors_a = await MinerPortalAPI.fetch_executors("hotkey-a", None)
    executors_b = await MinerPortalAPI.fetch_executors("hotkey-b", None)

    assert executors_a == [{"id": "aaaa-3", "validator_hotkey": "v1"}]
    assert executors_b == []  # old snapshot fully replaced, not merged
    assert bulk.await_count == 2


async def test_stale_snapshot_served_when_refresh_fails(monkeypatch):
    bulk = AsyncMock(return_value=SNAPSHOT)
    monkeypatch.setattr(MinerPortalAPI, "_fetch_bulk_snapshot", bulk)
    await MinerPortalAPI.fetch_executors("hotkey-a", None)

    _expire_snapshot()
    bulk.side_effect = RuntimeError("portal is down")

    executors_a = await MinerPortalAPI.fetch_executors("hotkey-a", None)

    assert executors_a == SNAPSHOT["hotkey-a"]  # stale data, not []
    assert bulk.await_count == 2


async def test_failed_refresh_backs_off_then_recovers(monkeypatch):
    bulk = AsyncMock(side_effect=RuntimeError("portal is down"))
    monkeypatch.setattr(MinerPortalAPI, "_fetch_bulk_snapshot", bulk)

    # cold start with portal down: the only case that still yields []
    assert await MinerPortalAPI.fetch_executors("hotkey-a", None) == []
    # within the backoff window no second attempt is made
    assert await MinerPortalAPI.fetch_executors("hotkey-a", None) == []
    assert bulk.await_count == 1

    # past the backoff window the next call refreshes successfully
    MinerPortalAPI._last_refresh_attempt_at -= FAILED_REFRESH_RETRY_SECONDS + 1
    bulk.side_effect = None
    bulk.return_value = SNAPSHOT

    assert await MinerPortalAPI.fetch_executors("hotkey-a", None) == SNAPSHOT["hotkey-a"]
    assert bulk.await_count == 2


async def test_empty_refresh_keeps_populated_snapshot(monkeypatch):
    bulk = AsyncMock(return_value=SNAPSHOT)
    monkeypatch.setattr(MinerPortalAPI, "_fetch_bulk_snapshot", bulk)
    await MinerPortalAPI.fetch_executors("hotkey-a", None)

    _expire_snapshot()
    bulk.return_value = {}

    executors_a = await MinerPortalAPI.fetch_executors("hotkey-a", None)

    assert executors_a == SNAPSHOT["hotkey-a"]  # sudden 200+{} treated as suspect
    assert bulk.await_count == 2
    # the empty response still stamps freshness: no re-fetch within TTL
    await MinerPortalAPI.fetch_executors("hotkey-a", None)
    assert bulk.await_count == 2


async def test_cold_start_empty_refresh_respects_ttl(monkeypatch):
    bulk = AsyncMock(return_value={})
    monkeypatch.setattr(MinerPortalAPI, "_fetch_bulk_snapshot", bulk)

    assert await MinerPortalAPI.fetch_executors("hotkey-a", None) == []
    assert await MinerPortalAPI.fetch_executors("hotkey-a", None) == []

    assert bulk.await_count == 1  # empty-but-successful fetch is fresh, not a 30s poll


async def test_executor_id_filters_locally(monkeypatch):
    bulk = AsyncMock(return_value=SNAPSHOT)
    monkeypatch.setattr(MinerPortalAPI, "_fetch_bulk_snapshot", bulk)

    matched = await MinerPortalAPI.fetch_executors("hotkey-a", "aaaa-2")
    missing = await MinerPortalAPI.fetch_executors("hotkey-a", "no-such-id")

    assert matched == [{"id": "aaaa-2", "validator_hotkey": "v1"}]
    assert missing == []
    assert bulk.await_count == 1


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


async def test_bulk_fetch_raises_on_http_error(portal_http):
    portal_http(_FakeResponse(status=500, text_body="Database error occurred"))

    with pytest.raises(RuntimeError, match="500"):
        await MinerPortalAPI._fetch_bulk_snapshot()


async def test_bulk_fetch_raises_on_non_dict_body(portal_http):
    portal_http(_FakeResponse(status=200, json_body=[{"id": "aaaa-1"}]))

    with pytest.raises(RuntimeError, match="shape"):
        await MinerPortalAPI._fetch_bulk_snapshot()


async def test_bulk_fetch_raises_on_wrapped_response_envelope(portal_http):
    portal_http(
        _FakeResponse(status=200, json_body={"success": True, "data": {"hotkey-a": []}})
    )

    with pytest.raises(RuntimeError, match="shape"):
        await MinerPortalAPI._fetch_bulk_snapshot()


async def test_bulk_fetch_sends_signature_headers_and_returns_snapshot(portal_http):
    portal_http(_FakeResponse(status=200, json_body=SNAPSHOT))

    snapshot = await MinerPortalAPI._fetch_bulk_snapshot()

    assert snapshot == SNAPSHOT
    assert _FakeSession.last_url.endswith("/miners/executors")
    assert set(_FakeSession.last_headers) == {"hotkey", "timestamp", "signature"}
