"""DAH-3006: the process-wide Redis command lock.

In a ~500-executor wave every pipeline queues on `RedisService.lock` for its one-command calls,
and the FIFO queue drains at ~0.4 s per command: the duplicate check's single SISMEMBER measured
p50 73 s (Loki, 6 Sep 12:03 wave). With REDIS_COMMAND_LOCK_ENABLED off the lock is a pass-through
and the calls overlap on the connection pool.
"""

import asyncio
import time

import pytest

from core.config import settings
from services import redis_service as redis_service_module
from services.redis_service import RedisService, _PassThroughLock

CALLS = 20
COMMAND_SECONDS = 0.05


class _SlowFakeRedis:
    """A Redis whose every command takes COMMAND_SECONDS, so serialisation is measurable."""

    def __init__(self):
        self.calls: list[tuple] = []

    async def _cmd(self, *args):
        self.calls.append(args)
        await asyncio.sleep(COMMAND_SECONDS)
        return True

    sismember = sadd = hset = hget = _cmd


def _service(monkeypatch, *, lock_enabled: bool) -> RedisService:
    monkeypatch.setattr(settings, "REDIS_COMMAND_LOCK_ENABLED", lock_enabled)
    # The constructor only builds pools from settings; nothing connects until a command runs.
    service = RedisService()
    service.redis = _SlowFakeRedis()
    return service


async def _timed(coro_factory, n: int = CALLS) -> tuple[float, list]:
    started = time.perf_counter()
    results = await asyncio.gather(*(coro_factory(i) for i in range(n)))
    return time.perf_counter() - started, results


@pytest.mark.asyncio
async def test_default_keeps_the_lock_and_serialises_commands(monkeypatch):
    service = _service(monkeypatch, lock_enabled=True)

    assert isinstance(service.lock, asyncio.Lock)
    elapsed, results = await _timed(lambda i: service.is_elem_exists_in_set("set", f"e{i}"))

    # Today's behaviour: one command at a time, N x command time.
    assert elapsed >= CALLS * COMMAND_SECONDS * 0.9
    assert results == [True] * CALLS


@pytest.mark.asyncio
async def test_flag_off_lets_concurrent_reads_overlap(monkeypatch):
    service = _service(monkeypatch, lock_enabled=False)

    assert isinstance(service.lock, _PassThroughLock)
    elapsed, results = await _timed(lambda i: service.is_elem_exists_in_set("set", f"e{i}"))

    # All N calls in flight together: about one command time, not N.
    assert elapsed < CALLS * COMMAND_SECONDS * 0.3
    assert results == [True] * CALLS
    assert len(service.redis.calls) == CALLS


@pytest.mark.asyncio
async def test_flag_off_writes_still_reach_the_client(monkeypatch):
    service = _service(monkeypatch, lock_enabled=False)

    await service.sadd("set", "elem")
    await service.hset("hash", "field", "value")
    await service.set_verified_job_info("miner", "executor", prev_info={}, success=True)

    commands = [call[0] for call in service.redis.calls]
    assert commands == ["set", "hash", redis_service_module.VERIFIED_JOB_COUNT_KEY]


@pytest.mark.asyncio
async def test_pass_through_lock_is_reusable_and_never_held():
    lock = _PassThroughLock()

    for _ in range(3):
        async with lock:
            assert lock.locked() is False
    assert lock.locked() is False
