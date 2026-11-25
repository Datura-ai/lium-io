"""DAH-6376: the Redis claims the port check and pod creation share."""

import asyncio

import pytest
from fakeredis.aioredis import FakeRedis

from services import redis_service as redis_module
from services.redis_service import RedisService


def _redis_service() -> RedisService:
    service = RedisService.__new__(RedisService)
    service.redis = FakeRedis()
    service.lock = asyncio.Lock()
    return service


@pytest.mark.asyncio
async def test_a_pending_pod_port_expires_on_its_own_even_after_a_later_pod(monkeypatch):
    """jam_bot 4571808199: a later pod must not keep an earlier pod's ports out of the check."""
    service = _redis_service()
    now = [1_000.0]
    monkeypatch.setattr(redis_module.time, "time", lambda: now[0])

    await service.record_pending_pod_ports("ex-1", [40001], 900)
    now[0] += 600
    await service.record_pending_pod_ports("ex-1", [40002], 900)
    now[0] += 400

    assert await service.pending_pod_ports("ex-1") == {40002}


@pytest.mark.asyncio
async def test_probe_held_ports_are_released():
    service = _redis_service()

    await service.hold_probe_ports("ex-1", {40001, 40002}, 60)
    held = await service.probe_held_ports("ex-1")
    await service.release_probe_ports("ex-1")

    assert held == {40001, 40002}
    assert await service.probe_held_ports("ex-1") == set()
