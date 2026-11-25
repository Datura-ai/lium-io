"""DAH-7077: one lock per volume between a storage dispatch and the delete's wipe."""

from uuid import uuid4

import pytest

from services.storage_operations import (
    StorageVolumeBeingWiped,
    wipe_token,
    release_storage_volume_lock,
    storage_dispatch_holds_volume,
    take_storage_volume_lock,
)


class VolumeLockRedis:
    """RedisService's two volume-hold calls, in memory, with the Lua script's rules."""

    def __init__(self):
        self.holders: dict[str, set[str]] = {}

    async def add_volume_hold(self, key, token, exclusive, ex):
        holders = self.holders.setdefault(key, set())
        if any(h != token and (exclusive or h.startswith("wipe:")) for h in holders):
            return False
        holders.add(token)
        return True

    async def remove_volume_hold(self, key, token):
        self.holders.get(key, set()).discard(token)


async def _wipe(redis, volume):
    return await take_storage_volume_lock(redis, volume, wipe_token("pod_" + volume), exclusive=True)


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
    wipe = wipe_token("pod_a")
    assert await take_storage_volume_lock(redis, "volume_a", wipe, exclusive=True)

    await release_storage_volume_lock(redis, "volume_a", wipe_token("pod_other"))

    assert redis.holders["storage_volume_holders:volume_a"] == {wipe}


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
    fakeredis = pytest.importorskip("fakeredis")
    pytest.importorskip("lupa")  # fakeredis runs the Lua through lupa
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
