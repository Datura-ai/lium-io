"""DAH-2439: kill-strike accounting in Redis.

Pins the per-run dedup (one dead run = one strike, no matter how many cycles observe it) and the
seen-key TTL refresh that stops a single run stuck RUNNING past the window from being double-charged.
"""

import asyncio

import fakeredis.aioredis
import pytest

from services.redis_service import RedisService

TTL = 100
EXECUTOR = "executor-uuid-1"


def _service() -> RedisService:
    # Bypass __init__ (which builds a real aioredis client) — the strike path only needs redis+lock.
    service = RedisService.__new__(RedisService)
    service.redis = fakeredis.aioredis.FakeRedis()
    service.lock = asyncio.Lock()
    return service


@pytest.mark.asyncio
async def test_same_run_observed_repeatedly_counts_one_strike():
    service = _service()
    first = await service.register_filler_kill_strike(EXECUTOR, "run-a", TTL)
    second = await service.register_filler_kill_strike(EXECUTOR, "run-a", TTL)
    third = await service.register_filler_kill_strike(EXECUTOR, "run-a", TTL)
    assert [first, second, third] == [1, 1, 1]


@pytest.mark.asyncio
async def test_distinct_runs_accumulate_strikes_per_executor():
    service = _service()
    first = await service.register_filler_kill_strike(EXECUTOR, "run-a", TTL)
    second = await service.register_filler_kill_strike(EXECUTOR, "run-b", TTL)
    assert [first, second] == [1, 2]


@pytest.mark.asyncio
async def test_repeat_observation_refreshes_seen_key_ttl():
    # Core regression: a run stuck RUNNING must not re-count once its seen marker nears expiry.
    service = _service()
    await service.register_filler_kill_strike(EXECUTOR, "run-a", TTL)
    # Simulate the seen marker aging toward expiry between cycles.
    await service.redis.expire("filler_kill_seen:run-a", 5)

    strikes = await service.register_filler_kill_strike(EXECUTOR, "run-a", TTL)

    assert strikes == 1  # still a single incident, not double-charged
    assert await service.redis.ttl("filler_kill_seen:run-a") > 5  # seen TTL was refreshed back up
