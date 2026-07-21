"""
Tests for the MinerPortalAPI snapshot cache (DAH-2469).

The cache must collapse the validator wave into one bulk portal request,
serve stale data when a refresh fails, back off after a failed refresh,
and filter by executor_id locally.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from clients.miner_portal_api import (
    FAILED_REFRESH_RETRY_SECONDS,
    SNAPSHOT_TTL_SECONDS,
    MinerPortalAPI,
)

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
    MinerPortalAPI._snapshot_fetched_at = 0.0
    MinerPortalAPI._last_refresh_attempt_at = 0.0
    MinerPortalAPI._refresh_lock = asyncio.Lock()
    yield


def _expire_snapshot():
    MinerPortalAPI._snapshot_fetched_at -= SNAPSHOT_TTL_SECONDS + 1
    MinerPortalAPI._last_refresh_attempt_at = 0.0


async def test_one_bulk_call_serves_all_hotkeys(monkeypatch):
    bulk = AsyncMock(return_value=SNAPSHOT)
    monkeypatch.setattr(MinerPortalAPI, "_fetch_bulk", bulk)

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
    monkeypatch.setattr(MinerPortalAPI, "_fetch_bulk", bulk)

    results = await asyncio.gather(
        *[MinerPortalAPI.fetch_executors("hotkey-a", None) for _ in range(10)]
    )

    assert bulk.await_count == 1
    assert all(result == SNAPSHOT["hotkey-a"] for result in results)


async def test_expired_snapshot_is_fully_replaced(monkeypatch):
    bulk = AsyncMock(return_value=SNAPSHOT)
    monkeypatch.setattr(MinerPortalAPI, "_fetch_bulk", bulk)
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
    monkeypatch.setattr(MinerPortalAPI, "_fetch_bulk", bulk)
    await MinerPortalAPI.fetch_executors("hotkey-a", None)

    _expire_snapshot()
    bulk.side_effect = RuntimeError("portal is down")

    executors_a = await MinerPortalAPI.fetch_executors("hotkey-a", None)

    assert executors_a == SNAPSHOT["hotkey-a"]  # stale data, not []
    assert bulk.await_count == 2


async def test_failed_refresh_backs_off_then_recovers(monkeypatch):
    bulk = AsyncMock(side_effect=RuntimeError("portal is down"))
    monkeypatch.setattr(MinerPortalAPI, "_fetch_bulk", bulk)

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


async def test_executor_id_filters_locally(monkeypatch):
    bulk = AsyncMock(return_value=SNAPSHOT)
    monkeypatch.setattr(MinerPortalAPI, "_fetch_bulk", bulk)

    matched = await MinerPortalAPI.fetch_executors("hotkey-a", "aaaa-2")
    missing = await MinerPortalAPI.fetch_executors("hotkey-a", "no-such-id")

    assert matched == [{"id": "aaaa-2", "validator_hotkey": "v1"}]
    assert missing == []
    assert bulk.await_count == 1
