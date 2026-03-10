import asyncio
import json
from contextlib import asynccontextmanager

import fakeredis
import pytest
import pytest_asyncio

from neurons.validators.src.services.redis_service import (
    BANDWIDTH_HISTORY_MAX_ENTRIES,
    BANDWIDTH_HISTORY_PREFIX,
    RedisService,
)


@pytest_asyncio.fixture
async def redis_service(monkeypatch):
    """RedisService backed by an in-process async fakeredis instance.

    acquire_executor_lock is stubbed to a no-op because fakeredis does not
    support the Lua-based EVALSHA commands used by the distributed Redis lock.
    Locking correctness is out of scope for these unit tests.
    """
    svc = RedisService.__new__(RedisService)
    svc.lock = asyncio.Lock()
    svc.redis = fakeredis.FakeAsyncRedis()

    @asynccontextmanager
    async def _noop_lock(executor_id: str, **kwargs):
        yield None

    monkeypatch.setattr(svc, "acquire_executor_lock", _noop_lock)
    return svc


@pytest.mark.asyncio
async def test_store_bandwidth_measurement_single(redis_service):
    """Storing one measurement persists correctly."""
    await redis_service.store_bandwidth_measurement("exec-1", 500.0, 300.0)
    history = await redis_service.get_bandwidth_history("exec-1")
    assert len(history) == 1
    assert history[0]["upload"] == 500.0
    assert history[0]["download"] == 300.0
    assert "ts" in history[0]


@pytest.mark.asyncio
async def test_store_bandwidth_measurement_trim(redis_service):
    """History is trimmed to BANDWIDTH_HISTORY_MAX_ENTRIES."""
    for i in range(BANDWIDTH_HISTORY_MAX_ENTRIES + 5):
        await redis_service.store_bandwidth_measurement("exec-trim", float(i), float(i))
    history = await redis_service.get_bandwidth_history("exec-trim")
    assert len(history) == BANDWIDTH_HISTORY_MAX_ENTRIES


@pytest.mark.asyncio
async def test_get_averaged_bandwidth_basic(redis_service):
    """Averages are computed correctly over multiple entries."""
    await redis_service.store_bandwidth_measurement("exec-avg", 100.0, 200.0)
    await redis_service.store_bandwidth_measurement("exec-avg", 300.0, 400.0)
    result = await redis_service.get_averaged_bandwidth("exec-avg")
    assert result["upload_speed"] == pytest.approx(200.0)
    assert result["download_speed"] == pytest.approx(300.0)


@pytest.mark.asyncio
async def test_get_averaged_bandwidth_filters_none(redis_service):
    """None values in history are ignored when averaging."""
    await redis_service.store_bandwidth_measurement("exec-none", None, 200.0)
    await redis_service.store_bandwidth_measurement("exec-none", 100.0, None)
    result = await redis_service.get_averaged_bandwidth("exec-none")
    assert result["upload_speed"] == pytest.approx(100.0)
    assert result["download_speed"] == pytest.approx(200.0)


@pytest.mark.asyncio
async def test_get_averaged_bandwidth_all_none(redis_service):
    """When all values are None, returns None."""
    await redis_service.store_bandwidth_measurement("exec-allnone", None, None)
    result = await redis_service.get_averaged_bandwidth("exec-allnone")
    assert result["upload_speed"] is None
    assert result["download_speed"] is None


@pytest.mark.asyncio
async def test_get_averaged_bandwidth_no_history(redis_service):
    """When there is no history, returns None for both."""
    result = await redis_service.get_averaged_bandwidth("exec-missing")
    assert result["upload_speed"] is None
    assert result["download_speed"] is None


@pytest.mark.asyncio
async def test_get_bandwidth_history_empty(redis_service):
    """No history returns empty list."""
    history = await redis_service.get_bandwidth_history("exec-empty")
    assert history == []
