import pytest

from services.redis_service import FORCED_VALIDATION_CYCLE_KEY, RedisService


class _Redis:
    """Just enough Redis for the forced-cycle request: SET, GET and the compare-and-delete script."""

    def __init__(self):
        self.kv: dict[str, str] = {}

    async def set(self, key, value, ex=None):
        self.kv[key] = value

    async def get(self, key):
        return self.kv.get(key)

    async def eval(self, script, numkeys, key, token):
        if self.kv.get(key) == token:
            del self.kv[key]


def _service() -> RedisService:
    service = RedisService.__new__(RedisService)
    service.redis = _Redis()
    return service


@pytest.mark.asyncio
async def test_clearing_a_cycles_request_keeps_one_made_while_the_cycle_was_prepared():
    service = _service()
    await service.request_forced_validation_cycle()
    read = await service.forced_validation_cycle_request()
    await service.request_forced_validation_cycle()

    await service.clear_forced_validation_cycle_request(read)

    newer = await service.forced_validation_cycle_request()
    assert newer is not None and newer != read


@pytest.mark.asyncio
async def test_clearing_the_request_the_cycle_read_removes_it():
    service = _service()
    await service.request_forced_validation_cycle()

    await service.clear_forced_validation_cycle_request(await service.forced_validation_cycle_request())

    assert FORCED_VALIDATION_CYCLE_KEY not in service.redis.kv
