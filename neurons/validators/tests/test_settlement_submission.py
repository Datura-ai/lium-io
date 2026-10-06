"""DAH-4001: what the validator submits at tempo under SETTLEMENT_MODE=enforce."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from clients.backend_client import WeightBatch
from core.validator import LAST_WEIGHT_BATCH_KEY, Validator


def _validator(
    *, claimed: WeightBatch | None, cached: WeightBatch | None, accepted: bool
) -> Validator:
    validator = Validator.__new__(Validator)
    validator.default_extra = {}
    validator.active_hotkeys = set()
    validator.backend_client = MagicMock(
        claim_weight_batch=AsyncMock(return_value=claimed),
        report_weight_batch_result=AsyncMock(return_value=None),
    )
    validator.redis_service = MagicMock(
        get=AsyncMock(return_value=cached.model_dump_json() if cached else None),
        set=AsyncMock(),
    )
    validator.subtensor_client = MagicMock(
        set_weights=AsyncMock(return_value=accepted),
        get_current_block=MagicMock(return_value=123),
        get_last_update=MagicMock(return_value=50),  # last accepted weights at block 73
        netuid=51,
    )
    validator.subtensor_client.subtensor.get_subnet_hyperparameters.return_value = MagicMock(
        activity_cutoff=12000
    )
    return validator


BATCH = WeightBatch(
    batch_id="b1",
    cycle_ids=["c1", "c2"],
    hotkey_scores={"hk": 0.7, "burn": 0.3},
    attempts=0,
    status="open",
)


@pytest.mark.asyncio
async def test_matured_batch_is_submitted_cached_and_reported():
    validator = _validator(claimed=BATCH, cached=None, accepted=True)

    await validator.submit_settled_batch()

    validator.subtensor_client.set_weights.assert_awaited_once_with(
        miner_scores=BATCH.hotkey_scores, active_hotkeys=set(), wait_for_inclusion=True
    )
    validator.redis_service.set.assert_awaited_once()
    assert json.loads(validator.redis_service.set.await_args.args[1])["batch_id"] == "b1"
    validator.backend_client.report_weight_batch_result.assert_awaited_once_with(
        "b1", success=True, block=123, error=None, shadow=False
    )


@pytest.mark.asyncio
async def test_failed_submission_is_reported_and_not_cached():
    validator = _validator(claimed=BATCH, cached=None, accepted=False)

    await validator.submit_settled_batch()

    validator.redis_service.set.assert_not_awaited()
    validator.backend_client.report_weight_batch_result.assert_awaited_once_with(
        "b1", success=False, block=123, error="set_weights failed", shadow=False
    )


@pytest.mark.asyncio
async def test_nothing_matured_resubmits_the_last_accepted_batch_without_reporting():
    validator = _validator(claimed=None, cached=BATCH, accepted=True)

    await validator.submit_settled_batch()

    validator.subtensor_client.set_weights.assert_awaited_once()
    validator.backend_client.report_weight_batch_result.assert_not_awaited()


@pytest.mark.asyncio
async def test_nothing_matured_and_nothing_cached_skips_the_tempo():
    validator = _validator(claimed=None, cached=None, accepted=True)

    await validator.submit_settled_batch()

    validator.subtensor_client.set_weights.assert_not_awaited()
    validator.redis_service.get.assert_awaited_once_with(LAST_WEIGHT_BATCH_KEY)


@pytest.mark.asyncio
async def test_retry_whose_earlier_attempt_landed_is_closed_without_resubmitting():
    retried = BATCH.model_copy(
        update={"attempts": 1, "first_attempt_block": 70}
    )  # LastUpdate at 73 >= 70
    validator = _validator(claimed=retried, cached=None, accepted=True)

    await validator.submit_settled_batch()

    validator.subtensor_client.set_weights.assert_not_awaited()
    validator.backend_client.report_weight_batch_result.assert_awaited_once_with(
        "b1", success=True, block=123, error=None, shadow=False
    )


@pytest.mark.asyncio
async def test_retry_whose_earlier_attempt_did_not_land_is_resubmitted():
    retried = BATCH.model_copy(
        update={"attempts": 1, "first_attempt_block": 100}
    )  # LastUpdate at 73 < 100
    validator = _validator(claimed=retried, cached=None, accepted=True)

    await validator.submit_settled_batch()

    validator.subtensor_client.set_weights.assert_awaited_once()


@pytest.mark.asyncio
async def test_set_weights_raising_is_reported_as_a_failure_with_the_block():
    validator = _validator(claimed=BATCH, cached=None, accepted=True)
    validator.subtensor_client.set_weights = AsyncMock(side_effect=TimeoutError("rpc"))

    await validator.submit_settled_batch()

    validator.redis_service.set.assert_not_awaited()
    validator.backend_client.report_weight_batch_result.assert_awaited_once_with(
        "b1", success=False, block=123, error="set_weights raised: rpc", shadow=False
    )
