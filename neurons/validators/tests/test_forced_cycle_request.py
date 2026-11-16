from unittest.mock import patch

import pytest

from services.redis_service import (
    FORCED_VALIDATION_CYCLE_ACKED_KEY,
    FORCED_VALIDATION_CYCLE_KEY,
    LEGACY_FORCED_VALIDATION_CYCLE_KEY,
    RedisService,
)

pytest_plugins = ["fixtures.incentive_fixtures"]


class _Redis:
    """Just enough Redis for the forced-cycle request: INCR, SET, MGET and the acknowledge script. TTLs are recorded,
    never applied: only the legacy key may carry one."""

    def __init__(self):
        self.kv: dict[str, str] = {}
        self.ttl: dict[str, int] = {}

    async def incr(self, key):
        self.kv[key] = str(int(self.kv.get(key, 0)) + 1)

    async def set(self, key, value, ex=None):
        self.kv[key] = value
        self.ttl[key] = ex

    async def mget(self, *keys):
        return [self.kv.get(key) for key in keys]

    async def eval(self, script, numkeys, count_key, acked_key, legacy_key, token):
        read_count, read_legacy = token.split(":", 1)
        acked = max(int(self.kv.get(acked_key, 0)), int(read_count))
        self.kv[acked_key] = str(acked)
        if self.kv.get(legacy_key) == read_legacy:
            del self.kv[legacy_key]
        count, legacy = int(self.kv.get(count_key, 0)), self.kv.get(legacy_key)
        return f"{count}:{legacy or ''}" if count > acked or legacy else None


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

    assert (
        await service.clear_forced_validation_cycle_request(
            await service.forced_validation_cycle_request()
        )
        is None
    )

    assert await service.forced_validation_cycle_request() is None
    assert (
        service.redis.kv[FORCED_VALIDATION_CYCLE_ACKED_KEY]
        == service.redis.kv[FORCED_VALIDATION_CYCLE_KEY]
    )
    assert LEGACY_FORCED_VALIDATION_CYCLE_KEY not in service.redis.kv


@pytest.mark.asyncio
async def test_a_request_from_a_connector_on_the_old_version_still_starts_a_cycle():
    # review 4580868787: mid-rollout the old connector writes only the legacy key
    service = _service()
    service.redis.kv[LEGACY_FORCED_VALIDATION_CYCLE_KEY] = "old-token"

    read = await service.forced_validation_cycle_request()

    assert read is not None
    assert await service.clear_forced_validation_cycle_request(read) is None
    assert await service.forced_validation_cycle_request() is None


@pytest.mark.asyncio
async def test_a_request_also_writes_the_legacy_key_an_old_validator_reads_with_its_ttl():
    service = _service()

    await service.request_forced_validation_cycle()

    assert service.redis.kv[LEGACY_FORCED_VALIDATION_CYCLE_KEY]
    assert service.redis.ttl[LEGACY_FORCED_VALIDATION_CYCLE_KEY] == 60
    assert service.redis.ttl.get(FORCED_VALIDATION_CYCLE_KEY) is None


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
        validator.carried_forced_request = (
            await validator.redis_service.clear_forced_validation_cycle_request(read)
        )
        validator.carried_forced_request = None  # in-memory copy lost too, e.g. a restart

        assert await validator.an_operator_asked_for_a_cycle_now() not in (None, read)
