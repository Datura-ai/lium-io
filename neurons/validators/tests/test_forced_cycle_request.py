from unittest.mock import patch

import pytest

from services.redis_service import FORCED_VALIDATION_CYCLE_ACKED_KEY, FORCED_VALIDATION_CYCLE_KEY, RedisService

pytest_plugins = ["fixtures.incentive_fixtures"]


class _Redis:
    """Just enough Redis for the forced-cycle request: INCR, MGET and the acknowledge script. No TTL support, so a
    request that relied on one would fail here."""

    def __init__(self):
        self.kv: dict[str, str] = {}

    async def incr(self, key):
        self.kv[key] = str(int(self.kv.get(key, 0)) + 1)

    async def mget(self, *keys):
        return [self.kv.get(key) for key in keys]

    async def eval(self, script, numkeys, count_key, acked_key, token):
        count = int(self.kv.get(count_key, 0))
        acked = max(int(self.kv.get(acked_key, 0)), int(token))
        self.kv[acked_key] = str(acked)
        return str(count) if count > acked else None


def _service() -> RedisService:
    service = RedisService.__new__(RedisService)
    service.redis = _Redis()
    return service


@pytest.mark.asyncio
async def test_acknowledging_a_cycles_request_keeps_one_made_while_the_cycle_was_prepared():
    service = _service()
    await service.request_forced_validation_cycle()
    read = await service.forced_validation_cycle_request()
    await service.request_forced_validation_cycle()

    carried = await service.clear_forced_validation_cycle_request(read)

    newer = await service.forced_validation_cycle_request()
    assert newer is not None and newer != read and carried == newer


@pytest.mark.asyncio
async def test_acknowledging_the_request_the_cycle_read_ends_it():
    service = _service()
    await service.request_forced_validation_cycle()

    assert await service.clear_forced_validation_cycle_request(await service.forced_validation_cycle_request()) is None

    assert await service.forced_validation_cycle_request() is None
    assert service.redis.kv[FORCED_VALIDATION_CYCLE_ACKED_KEY] == service.redis.kv[FORCED_VALIDATION_CYCLE_KEY]


@pytest.mark.asyncio
async def test_a_request_made_while_the_cycle_was_prepared_starts_the_next_cycle_however_long_the_wave(
    validator_with_mocks,
):
    # review 4580791049: the request has no TTL, so a slow build, rollout or wave cannot drop it before it is read
    validator = validator_with_mocks
    validator.redis_service = _service()
    await validator.redis_service.request_forced_validation_cycle()
    with patch("core.validator.settings.DEPLOY_ENV", "STAGE"):
        read = await validator.an_operator_asked_for_a_cycle_now()
        await validator.redis_service.request_forced_validation_cycle()
        validator.carried_forced_request = await validator.redis_service.clear_forced_validation_cycle_request(read)
        validator.carried_forced_request = None  # in-memory copy lost too, e.g. a restart

        assert await validator.an_operator_asked_for_a_cycle_now() not in (None, read)
