from unittest.mock import patch

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
            return None
        return self.kv.get(key)


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

    carried = await service.clear_forced_validation_cycle_request(read)

    newer = await service.forced_validation_cycle_request()
    assert newer is not None and newer != read and carried == newer


@pytest.mark.asyncio
async def test_clearing_the_request_the_cycle_read_removes_it():
    service = _service()
    await service.request_forced_validation_cycle()

    await service.clear_forced_validation_cycle_request(await service.forced_validation_cycle_request())

    assert FORCED_VALIDATION_CYCLE_KEY not in service.redis.kv


@pytest.mark.asyncio
async def test_a_newer_request_whose_key_expired_during_the_wave_still_starts_the_next_cycle(validator_with_mocks):
    validator = validator_with_mocks
    validator.redis_service = _service()
    await validator.redis_service.request_forced_validation_cycle()
    read = await validator.an_operator_asked_for_a_cycle_now()
    await validator.redis_service.request_forced_validation_cycle()
    validator.carried_forced_request = await validator.redis_service.clear_forced_validation_cycle_request(read)
    validator.redis_service.redis.kv.clear()  # the newer key's TTL ran out mid-wave

    with patch("core.validator.settings.DEPLOY_ENV", "STAGE"):
        assert await validator.an_operator_asked_for_a_cycle_now() == validator.carried_forced_request
