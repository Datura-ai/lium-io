"""DAH-7077: one lock per volume between a storage dispatch and the delete's wipe."""

import time
from unittest.mock import AsyncMock
from uuid import uuid4

import fakeredis
import lupa  # noqa: F401  fakeredis runs the Lua through it; a missing lupa must fail, not skip
import pytest
from services.storage_operations import (
    StorageVolumeBeingWiped,
    new_wipe_token,
    release_storage_volume_lock,
    storage_dispatch_holds_volume,
    take_storage_volume_lock,
)


class VolumeLockRedis:
    """RedisService's two volume-hold calls, in memory, with the Lua script's rules (deadline per token)."""

    def __init__(self):
        self.holders: dict[str, dict[str, float]] = {}

    async def add_volume_hold(self, key, token, exclusive, ex, now=None):
        now = time.time() if now is None else now
        holders = self.holders.setdefault(key, {})
        for holder in [h for h, deadline in holders.items() if deadline <= now]:
            del holders[holder]
        if any(h != token and (exclusive or h.startswith("wipe:")) for h in holders):
            return False
        holders[token] = now + ex
        return True

    async def remove_volume_hold(self, key, token):
        self.holders.get(key, {}).pop(token, None)


async def _wipe(redis, volume):
    return await take_storage_volume_lock(redis, volume, new_wipe_token(), exclusive=True)


@pytest.mark.asyncio
async def test_a_dispatch_holds_the_volume_against_the_wipe_until_its_block_ends():
    redis = VolumeLockRedis()

    async with storage_dispatch_holds_volume(redis, "volume_a", "op-1"):
        assert not await _wipe(redis, "volume_a")
        assert not await _wipe(redis, "volume_a")  # a refused wipe left no token behind
        assert await _wipe(redis, "volume_b")

    assert await _wipe(redis, "volume_a")


@pytest.mark.asyncio
async def test_a_dispatch_is_refused_while_the_wipe_holds_the_volume():
    redis = VolumeLockRedis()
    await _wipe(redis, "volume_a")

    with pytest.raises(StorageVolumeBeingWiped):
        async with storage_dispatch_holds_volume(redis, "volume_a", "op-1"):
            pytest.fail("the dispatch ran on a volume being wiped")


@pytest.mark.asyncio
async def test_a_release_drops_only_its_own_token():
    redis = VolumeLockRedis()
    wipe = new_wipe_token()
    assert await take_storage_volume_lock(redis, "volume_a", wipe, exclusive=True)

    await release_storage_volume_lock(redis, "volume_a", new_wipe_token())

    assert set(redis.holders["storage_volume_holders:volume_a"]) == {wipe}


@pytest.mark.asyncio
async def test_a_failed_attempt_of_one_operation_never_drops_another_attempts_hold():
    # Pixel's review: two attempts of one operation, one fails and releases, the other still holds
    redis = VolumeLockRedis()
    op = uuid4()

    async with storage_dispatch_holds_volume(redis, "vol", op):
        with pytest.raises(RuntimeError):
            async with storage_dispatch_holds_volume(redis, "vol", op):
                raise RuntimeError("ssh failed")
        assert not await _wipe(redis, "vol")

    assert await _wipe(redis, "vol")


@pytest.mark.asyncio
async def test_the_redis_script_keeps_each_attempts_hold_and_lets_a_wipe_in_only_alone():
    from services.redis_service import RedisService

    service = RedisService.__new__(RedisService)
    service.redis = fakeredis.FakeAsyncRedis()
    op = uuid4()

    async with storage_dispatch_holds_volume(service, "vol", op):
        with pytest.raises(RuntimeError):
            async with storage_dispatch_holds_volume(service, "vol", op):
                raise RuntimeError("ssh failed")
        assert not await _wipe(service, "vol")

    assert await _wipe(service, "vol")
    assert 0 < await service.redis.ttl("storage_volume_holders:vol") <= 3600
    with pytest.raises(StorageVolumeBeingWiped):
        async with storage_dispatch_holds_volume(service, "vol", uuid4()):
            pass


async def _lapse_cases(redis):
    # a holder that died (validator crash, restart) leaves its token: past its deadline it no longer blocks
    key, ttl = "storage_volume_holders:vol", 3600
    t0 = 1_000_000.0
    assert await redis.add_volume_hold(key, "operation:dead:1", False, ttl, now=t0)
    assert not await redis.add_volume_hold(key, "wipe:a", True, ttl, now=t0 + ttl - 1)
    assert await redis.add_volume_hold(key, "wipe:a", True, ttl, now=t0 + ttl + 1)
    await redis.remove_volume_hold(key, "wipe:a")
    # a live holder that renews its own token keeps blocking past the first deadline
    assert await redis.add_volume_hold(key, "operation:live:1", False, ttl, now=t0)
    assert await redis.add_volume_hold(key, "operation:live:1", False, ttl, now=t0 + ttl - 10)
    assert not await redis.add_volume_hold(key, "wipe:b", True, ttl, now=t0 + ttl + 1)


@pytest.mark.asyncio
async def test_a_release_error_after_the_dispatch_never_fails_the_started_operation():
    service = VolumeLockRedis()
    service.remove_volume_hold = AsyncMock(side_effect=ConnectionError("redis down"))
    async with storage_dispatch_holds_volume(service, "vol", uuid4()):
        pass


@pytest.mark.asyncio
async def test_the_redis_script_drops_lapsed_tokens_and_keeps_renewed_ones():
    from services.redis_service import RedisService

    service = RedisService.__new__(RedisService)
    service.redis = fakeredis.FakeAsyncRedis()
    await _lapse_cases(service)


@pytest.mark.asyncio
async def test_two_deletes_of_one_container_never_release_each_others_hold():
    redis = VolumeLockRedis()
    first, second = new_wipe_token(), new_wipe_token()
    assert await take_storage_volume_lock(redis, "vol", first, exclusive=True)
    assert not await take_storage_volume_lock(redis, "vol", second, exclusive=True)
    await release_storage_volume_lock(redis, "vol", second)
    with pytest.raises(StorageVolumeBeingWiped):
        async with storage_dispatch_holds_volume(redis, "vol", uuid4()):
            pass
