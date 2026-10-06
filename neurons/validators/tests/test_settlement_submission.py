"""DAH-4001: what the validator submits at tempo under SETTLEMENT_MODE=enforce."""

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from clients.backend_client import WeightBatch
from core.validator import LAST_WEIGHT_BATCH_KEY, UNACKED_CYCLE_REPORTS_KEY, Validator


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
        lrange=AsyncMock(return_value=[]),
        lpush=AsyncMock(),
        ltrim=AsyncMock(),
        lrem=AsyncMock(),
    )
    validator.backend_client.keypair = MagicMock(ss58_address="validator-hotkey")
    validator.backend_client.report_cycle_scores = AsyncMock(return_value=None)
    validator.subtensor_client = MagicMock(
        set_weights=AsyncMock(return_value=accepted),
        get_current_block=MagicMock(return_value=123),
        get_last_update=MagicMock(return_value=50),
        get_weights_rate_limit=MagicMock(return_value=100),
        netuid=51,
    )
    validator._settled_submission_block = None
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
async def test_set_weights_raising_is_reported_as_a_failure_with_the_block():
    validator = _validator(claimed=BATCH, cached=None, accepted=True)
    validator.subtensor_client.set_weights = AsyncMock(side_effect=TimeoutError("rpc"))

    await validator.submit_settled_batch()

    validator.redis_service.set.assert_not_awaited()
    validator.backend_client.report_weight_batch_result.assert_awaited_once_with(
        "b1", success=False, block=123, error="set_weights raised: rpc", shadow=False
    )


@pytest.mark.asyncio
async def test_a_cache_write_failure_does_not_stop_the_result_report():
    validator = _validator(claimed=BATCH, cached=None, accepted=True)
    validator.redis_service.set = AsyncMock(side_effect=ConnectionError("redis down"))

    await validator.submit_settled_batch()

    validator.backend_client.report_weight_batch_result.assert_awaited_once_with(
        "b1", success=True, block=123, error=None, shadow=False
    )


@pytest.mark.asyncio
async def test_an_unacknowledged_cycle_report_is_kept_and_replayed_next_cycle():
    validator = _validator(claimed=None, cached=None, accepted=True)
    miners = [MagicMock(uid=47, hotkey="burn")]
    scored_at = datetime(2026, 10, 6, 10, 0, tzinfo=UTC)

    await validator.report_cycle_scores(
        {"hk": 0.8, "burn": 0.2}, "2026-10-06 10:00:00", 500, scored_at, miners, 3
    )

    validator.redis_service.lpush.assert_awaited_once()
    kept = validator.redis_service.lpush.await_args.args[1]
    assert (
        json.loads(kept)["cycle_id"] == "2026-10-06 10:00:00"
        and json.loads(kept)["idle_executor_count"] == 3
    )

    validator.redis_service.lrange = AsyncMock(return_value=[kept])
    validator.backend_client.report_cycle_scores = AsyncMock(
        return_value=MagicMock(cycle_id="2026-10-06 10:00:00", matures_at=scored_at, created=True)
    )
    await validator.report_cycle_scores(
        {"hk": 1.0}, "2026-10-06 10:15:00", 575, scored_at, miners, 1
    )

    validator.redis_service.lrem.assert_awaited_once_with(UNACKED_CYCLE_REPORTS_KEY, kept)
    assert validator.backend_client.report_cycle_scores.await_count == 2


@pytest.mark.asyncio
async def test_a_tick_inside_the_rate_limit_after_an_accepted_submission_does_nothing():
    validator = _validator(claimed=BATCH, cached=None, accepted=True)

    await validator.submit_settled_batch()  # accepted at block 123
    await validator.submit_settled_batch()  # same block: inside the 100-block rate limit

    validator.backend_client.claim_weight_batch.assert_awaited_once()
    validator.subtensor_client.set_weights.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_claimed_batch_the_cache_says_was_accepted_is_reported_not_resubmitted():
    accepted_earlier = BATCH.model_copy(update={"submitted_block": 100})
    validator = _validator(claimed=BATCH, cached=accepted_earlier, accepted=True)

    await validator.submit_settled_batch()

    validator.subtensor_client.set_weights.assert_not_awaited()
    validator.backend_client.report_weight_batch_result.assert_awaited_once_with(
        "b1", success=True, block=100, error=None, shadow=False
    )


@pytest.mark.asyncio
async def test_replay_stops_at_the_first_report_that_still_fails():
    validator = _validator(claimed=None, cached=None, accepted=True)
    older, newer = (
        json.dumps({"cycle_id": "older", "hotkey_scores": {}}).encode(),
        json.dumps({"cycle_id": "newer", "hotkey_scores": {}}).encode(),
    )
    validator.redis_service.lrange = AsyncMock(
        return_value=[newer, older]
    )  # lpush order: newest first

    await validator._replay_unacked_cycle_reports()

    validator.backend_client.report_cycle_scores.assert_awaited_once()  # the oldest, which failed; the newer one waits
    assert validator.backend_client.report_cycle_scores.await_args.args[0]["cycle_id"] == "older"


@pytest.mark.asyncio
async def test_a_failed_attempt_is_not_retried_until_the_rate_limit_has_passed():
    validator = _validator(claimed=BATCH, cached=None, accepted=False)

    await validator.submit_settled_batch()  # fails at block 123
    await validator.submit_settled_batch()  # same block: no second attempt yet

    validator.subtensor_client.set_weights.assert_awaited_once()
    validator.backend_client.report_weight_batch_result.assert_awaited_once()


@pytest.mark.asyncio
async def test_resubmitting_the_cached_batch_keeps_its_first_acceptance_block():
    accepted_earlier = BATCH.model_copy(update={"submitted_block": 100})
    validator = _validator(
        claimed=None, cached=accepted_earlier, accepted=True
    )  # accepted again at block 123

    await validator.submit_settled_batch()

    assert json.loads(validator.redis_service.set.await_args.args[1])["submitted_block"] == 100
