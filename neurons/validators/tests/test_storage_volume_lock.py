"""DAH-7077: one lock per volume between a storage dispatch and the delete's wipe."""

import pytest

from services.storage_operations import (
    WIPE_LOCK_HOLDER,
    StorageVolumeBeingWiped,
    release_storage_volume_lock,
    storage_dispatch_holds_volume,
    take_storage_volume_lock,
)


class VolumeLockRedis:
    """The three RedisService calls the volume lock makes, in memory."""

    def __init__(self):
        self.values: dict[str, str] = {}

    async def set_if_absent(self, key, value, ex):
        if key in self.values:
            return False
        self.values[key] = value
        return True

    async def get(self, key):
        return self.values.get(key)

    async def delete(self, key):
        self.values.pop(key, None)


@pytest.mark.asyncio
async def test_a_dispatch_holds_the_volume_against_the_wipe_until_its_block_ends():
    redis = VolumeLockRedis()

    async with storage_dispatch_holds_volume(redis, "volume_a", "op-1"):
        assert not await take_storage_volume_lock(redis, "volume_a", WIPE_LOCK_HOLDER)
        assert await take_storage_volume_lock(redis, "volume_b", WIPE_LOCK_HOLDER)

    assert await take_storage_volume_lock(redis, "volume_a", WIPE_LOCK_HOLDER)


@pytest.mark.asyncio
async def test_a_dispatch_is_refused_while_the_wipe_holds_the_volume():
    redis = VolumeLockRedis()
    await take_storage_volume_lock(redis, "volume_a", WIPE_LOCK_HOLDER)

    with pytest.raises(StorageVolumeBeingWiped):
        async with storage_dispatch_holds_volume(redis, "volume_a", "op-1"):
            pytest.fail("the dispatch ran on a volume being wiped")


@pytest.mark.asyncio
async def test_a_repeated_dispatch_of_the_same_operation_keeps_its_hold():
    redis = VolumeLockRedis()
    await take_storage_volume_lock(redis, "volume_a", "operation:op-1")

    async with storage_dispatch_holds_volume(redis, "volume_a", "op-1"):
        pass

    # the release is by holder: another holder's lock is never dropped
    await take_storage_volume_lock(redis, "volume_a", "operation:op-2")
    await release_storage_volume_lock(redis, "volume_a", WIPE_LOCK_HOLDER)
    assert list(redis.values.values()) == ["operation:op-2"]
