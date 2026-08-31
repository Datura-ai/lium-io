"""DAH-2090 cross-process check on a REAL Redis, not a mock.

The connector process writes the key and the validator process reads it. Mocks cannot show
that the two agree on the key name and on what counts as set, so this test uses a live Redis.
Skipped when none is reachable.
"""

import os

import pytest
import pytest_asyncio

from core.validator import Validator
from services.miner_service import MinerService
from services.redis_service import FORCED_VALIDATION_CYCLE_KEY, RedisService

FORCED_CYCLE_TEST_REDIS_PORT = os.environ.get("FORCED_CYCLE_TEST_REDIS_PORT")


@pytest_asyncio.fixture
async def redis_service(monkeypatch):
    if not FORCED_CYCLE_TEST_REDIS_PORT:
        pytest.skip("set FORCED_CYCLE_TEST_REDIS_PORT to run the live-Redis handoff check")

    from core.config import settings

    monkeypatch.setattr(settings, "REDIS_HOST", "localhost")
    monkeypatch.setattr(settings, "REDIS_PORT", int(FORCED_CYCLE_TEST_REDIS_PORT))
    service = RedisService()
    await service.delete(FORCED_VALIDATION_CYCLE_KEY)
    yield service
    await service.delete(FORCED_VALIDATION_CYCLE_KEY)


def _make_validator(redis_service: RedisService, last_job_run_blocks: int) -> Validator:
    validator = Validator.__new__(Validator)
    validator.redis_service = redis_service
    validator.default_extra = {}
    validator.last_job_run_blocks = last_job_run_blocks
    return validator


@pytest.mark.asyncio
async def test_the_connector_request_reaches_the_validator_loop(redis_service) -> None:
    from core.config import settings

    connector_side = MinerService.__new__(MinerService)
    connector_side.redis_service = redis_service
    # A second client, as in production: the two processes share Redis, not memory.
    validator_side = _make_validator(RedisService(), last_job_run_blocks=5_000_000)

    await validator_side.start_cycle_now_if_an_operator_asked(current_block=5_000_010)
    assert validator_side.last_job_run_blocks == 5_000_000

    await connector_side.request_validation_cycle_now()
    await validator_side.start_cycle_now_if_an_operator_asked(current_block=5_000_010)

    assert validator_side.last_job_run_blocks == 0
    # The reset is only useful if it opens the gate the cycle waits on.
    assert 5_000_010 - validator_side.last_job_run_blocks >= settings.BLOCKS_FOR_JOB


@pytest.mark.asyncio
async def test_one_request_starts_exactly_one_cycle(redis_service) -> None:
    connector_side = MinerService.__new__(MinerService)
    connector_side.redis_service = redis_service
    validator_side = _make_validator(RedisService(), last_job_run_blocks=5_000_000)

    await connector_side.request_validation_cycle_now()
    await validator_side.start_cycle_now_if_an_operator_asked(current_block=5_000_010)

    validator_side.last_job_run_blocks = 5_000_000
    await validator_side.start_cycle_now_if_an_operator_asked(current_block=5_000_010)

    assert validator_side.last_job_run_blocks == 5_000_000
    assert await redis_service.get(FORCED_VALIDATION_CYCLE_KEY) is None
