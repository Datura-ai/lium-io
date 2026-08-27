"""DAH-2786 — a host that puts its own GPU power limit back loses the unrented incentive.

Case 3 of the three power-cap problems: the cap applies, the validator reads it back, and
then something on the provider's host raises the limit again. The PEARL filler is killed by
the backend grace check, so the node runs no Lium job — and today it still collects the full
unrented incentive. This suite covers the two halves: the validator records each revert it
observes at restore time, and the incentive gate withholds the unrented incentive once the
node has reverted often enough inside the window.
"""

import json
import logging
import time
from unittest.mock import AsyncMock

import pytest
from datura.requests.miner_requests import ExecutorSSHInfo

from core.config import settings
from incentive.config import IncentiveConfig
from incentive.rental_price import MIN_POWER_CAP_REVERTS_TO_PENALIZE, RentalPriceIncentive
from services.gpu_power_limit import (
    CAP_REVERT_WINDOW_SECONDS,
    MAX_TRACKED_CAP_REVERTS,
    GpuPowerCapRevert,
    GpuPowerCapRevertHistory,
    GpuPowerRestoreRecord,
    _restore_key,
    _revert_key,
    _reverted_cap_target,
    read_gpu_power_cap_reverts,
    restore_tracked_gpu_power_limits,
)
from services.task_service import JobResult

EXECUTOR_ID = "executor-1"
POD_ID = "pod-1"
H200 = "NVIDIA H200"  # base model H200 is rental-eligible by default


# ---------------------------- _reverted_cap_target (pure) ----------------------------


def _restore_record(watts: int, capped_to_watts: int | None) -> GpuPowerRestoreRecord:
    return GpuPowerRestoreRecord(
        gpu_uuid="GPU-a",
        watts=watts,
        capped_to_watts=capped_to_watts,
        pod_id=POD_ID,
        executor_id=EXECUTOR_ID,
        capped_at=time.time(),
    )


def test_limit_back_at_the_pre_cap_value_is_a_revert() -> None:
    # The prod shape: 103 of 113 reverts land exactly on the miner's own pre-cap value.
    assert _reverted_cap_target(_restore_record(watts=409, capped_to_watts=315), watts_before=409) == 315


def test_limit_back_at_the_card_default_is_a_revert() -> None:
    # The other 10: above the pre-cap value, i.e. a driver reload or a reset by the host.
    assert _reverted_cap_target(_restore_record(watts=409, capped_to_watts=315), watts_before=450) == 315


def test_cap_that_still_holds_is_not_a_revert() -> None:
    assert _reverted_cap_target(_restore_record(watts=409, capped_to_watts=315), watts_before=315) is None


def test_partly_raised_limit_is_not_a_revert() -> None:
    # Conservative on purpose: only a limit back at or above where the miner had it counts,
    # so a frozen record from an earlier failed restore can never invent a breach.
    assert _reverted_cap_target(_restore_record(watts=409, capped_to_watts=315), watts_before=350) is None


def test_cap_that_never_lowered_the_limit_is_not_a_revert() -> None:
    # A miner running below our target is capped upwards; the restore then reads our own
    # target back and must not be read as the host raising it.
    assert _reverted_cap_target(_restore_record(watts=250, capped_to_watts=315), watts_before=315) is None


def test_record_without_a_cap_target_is_not_a_revert() -> None:
    # Records written before this change carry no target: fail open, never penalize.
    assert _reverted_cap_target(_restore_record(watts=409, capped_to_watts=None), watts_before=409) is None


def test_unknown_live_limit_is_not_a_revert() -> None:
    # The state query failed, so there is no reading to judge.
    assert _reverted_cap_target(_restore_record(watts=409, capped_to_watts=315), watts_before=None) is None


# ---------------------------- recording at restore time ----------------------------


class FakeRun:
    def __init__(self, stdout: str = "", stderr: str = "", exit_status: int = 0):
        self.stdout = stdout
        self.stderr = stderr
        self.exit_status = exit_status


def fake_ssh(*responses: FakeRun) -> AsyncMock:
    ssh = AsyncMock()
    ssh.run.side_effect = responses
    return ssh


def _set_ok(readback_watts: int) -> list[FakeRun]:
    return [FakeRun(), FakeRun(), FakeRun(stdout=f"{readback_watts}.00, Enabled\n")]


class FakeRedis:
    def __init__(self, initial: dict[str, str] | None = None):
        self.store: dict[str, str] = dict(initial or {})
        self.expiries: dict[str, int] = {}

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.store[key] = value
        if ex is not None:
            self.expiries[key] = ex

    async def get(self, key: str) -> str | None:
        return self.store.get(key)

    async def delete(self, key: str) -> None:
        self.store.pop(key, None)


class BrokenRedis(FakeRedis):
    async def get(self, key: str) -> str | None:
        raise ConnectionError("redis down")

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        raise ConnectionError("redis down")


def _stored_history(redis: FakeRedis) -> GpuPowerCapRevertHistory:
    return GpuPowerCapRevertHistory.model_validate_json(redis.store[_revert_key(EXECUTOR_ID)])


# uuid, current, default, min, max — GPU-a sits back at its pre-cap 400W
REVERTED_STATE_CSV = "GPU-a, 400, 400, 100, 400\n"
HELD_STATE_CSV = "GPU-a, 280, 400, 100, 400\n"


def _stored_restore_record(watts: int = 400, capped_to_watts: int | None = 280) -> str:
    return _restore_record(watts=watts, capped_to_watts=capped_to_watts).model_dump_json()


@pytest.mark.asyncio
async def test_restore_records_a_revert_when_the_limit_came_back() -> None:


    ssh = fake_ssh(FakeRun(stdout=REVERTED_STATE_CSV), *_set_ok(400))
    redis = FakeRedis({_restore_key("GPU-a"): _stored_restore_record()})

    await restore_tracked_gpu_power_limits(ssh, redis, ["GPU-a"])

    history = _stored_history(redis)
    assert history.executor_id == EXECUTOR_ID
    assert len(history.reverts) == 1
    assert history.reverts[0].gpu_uuid == "GPU-a"
    assert history.reverts[0].capped_to_watts == 280
    assert history.reverts[0].found_watts == 400
    assert redis.expiries[_revert_key(EXECUTOR_ID)] == CAP_REVERT_WINDOW_SECONDS


@pytest.mark.asyncio
async def test_restore_records_nothing_when_the_cap_held() -> None:


    ssh = fake_ssh(FakeRun(stdout=HELD_STATE_CSV), *_set_ok(400))
    redis = FakeRedis({_restore_key("GPU-a"): _stored_restore_record()})

    await restore_tracked_gpu_power_limits(ssh, redis, ["GPU-a"])

    assert _revert_key(EXECUTOR_ID) not in redis.store


@pytest.mark.asyncio
async def test_reverts_accumulate_and_old_ones_drop_out_of_the_window() -> None:


    stale = GpuPowerCapRevert(
        at=time.time() - CAP_REVERT_WINDOW_SECONDS - 60,
        gpu_uuid="GPU-a",
        capped_to_watts=280,
        found_watts=400,
    )
    fresh = GpuPowerCapRevert(at=time.time() - 60, gpu_uuid="GPU-a", capped_to_watts=280, found_watts=400)
    history = GpuPowerCapRevertHistory(executor_id=EXECUTOR_ID, reverts=[stale, fresh])
    ssh = fake_ssh(FakeRun(stdout=REVERTED_STATE_CSV), *_set_ok(400))
    redis = FakeRedis({
        _restore_key("GPU-a"): _stored_restore_record(),
        _revert_key(EXECUTOR_ID): history.model_dump_json(),
    })

    await restore_tracked_gpu_power_limits(ssh, redis, ["GPU-a"])

    assert len(_stored_history(redis).reverts) == 2  # the stale one is dropped, the new one added


@pytest.mark.asyncio
async def test_history_is_bounded() -> None:


    now = time.time()
    reverts = [
        GpuPowerCapRevert(at=now - index, gpu_uuid="GPU-a", capped_to_watts=280, found_watts=400)
        for index in range(MAX_TRACKED_CAP_REVERTS + 10)
    ]
    ssh = fake_ssh(FakeRun(stdout=REVERTED_STATE_CSV), *_set_ok(400))
    redis = FakeRedis({
        _restore_key("GPU-a"): _stored_restore_record(),
        _revert_key(EXECUTOR_ID): GpuPowerCapRevertHistory(
            executor_id=EXECUTOR_ID, reverts=reverts
        ).model_dump_json(),
    })

    await restore_tracked_gpu_power_limits(ssh, redis, ["GPU-a"])

    assert len(_stored_history(redis).reverts) == MAX_TRACKED_CAP_REVERTS


@pytest.mark.asyncio
async def test_restore_still_succeeds_when_redis_cannot_store_the_revert(caplog) -> None:
    # Teardown must never be blocked by the bookkeeping this gate needs.


    ssh = fake_ssh(FakeRun(stdout=REVERTED_STATE_CSV), *_set_ok(400))
    redis = FakeRedis({_restore_key("GPU-a"): _stored_restore_record()})
    redis.set = BrokenRedis().set

    with caplog.at_level(logging.ERROR):
        restored = await restore_tracked_gpu_power_limits(ssh, redis, ["GPU-a"])

    assert restored == 1


# ---------------------------- read_gpu_power_cap_reverts ----------------------------


@pytest.mark.asyncio
async def test_read_returns_only_reverts_inside_the_window() -> None:
    now = time.time()
    history = GpuPowerCapRevertHistory(
        executor_id=EXECUTOR_ID,
        reverts=[
            GpuPowerCapRevert(at=now - CAP_REVERT_WINDOW_SECONDS - 1, gpu_uuid="GPU-a", capped_to_watts=280, found_watts=400),
            GpuPowerCapRevert(at=now - 10, gpu_uuid="GPU-b", capped_to_watts=280, found_watts=400),
        ],
    )
    redis = FakeRedis({_revert_key(EXECUTOR_ID): history.model_dump_json()})

    reverts = await read_gpu_power_cap_reverts(redis, EXECUTOR_ID)

    assert [revert.gpu_uuid for revert in reverts] == ["GPU-b"]


@pytest.mark.asyncio
async def test_read_is_empty_without_a_record() -> None:
    assert await read_gpu_power_cap_reverts(FakeRedis(), EXECUTOR_ID) == []


@pytest.mark.asyncio
async def test_read_fails_open_when_redis_is_down() -> None:
    # Redis resilience: our own outage must never zero an innocent provider's incentive.
    assert await read_gpu_power_cap_reverts(BrokenRedis(), EXECUTOR_ID) == []


@pytest.mark.asyncio
async def test_read_fails_open_on_a_corrupt_record() -> None:
    redis = FakeRedis({_revert_key(EXECUTOR_ID): json.dumps({"nonsense": True})})

    assert await read_gpu_power_cap_reverts(redis, EXECUTOR_ID) == []


# ---------------------------- the incentive gate ----------------------------


def _build_incentive(redis) -> RentalPriceIncentive:
    return RentalPriceIncentive(IncentiveConfig(), redis, {}, {})


def _redis_with_reverts(count: int) -> FakeRedis:
    now = time.time()
    history = GpuPowerCapRevertHistory(
        executor_id=EXECUTOR_ID,
        reverts=[
            GpuPowerCapRevert(at=now - 60 * index, gpu_uuid="GPU-a", capped_to_watts=280, found_watts=400)
            for index in range(count)
        ],
    )
    return FakeRedis({_revert_key(EXECUTOR_ID): history.model_dump_json()})


def _make_job(is_rented: bool = False) -> JobResult:
    return JobResult(
        spec={"container_cap_eff": "000001ffffffffff", "nvidiactl_owner_uid": 0},
        executor_info=ExecutorSSHInfo(
            uuid=EXECUTOR_ID,
            address="10.0.0.1",
            port=8080,
            ssh_username="root",
            ssh_port=22,
            python_path="/usr/bin/python3",
            root_dir="/tmp",
            price_per_gpu=1.0,
        ),
        score=1.0,
        job_score=1.0,
        job_batch_id="batch",
        log_status="success",
        log_text="ok",
        gpu_model=H200,
        gpu_count=8,
        is_rented=is_rented,
        collateral_deposited=True,
        sysbox_runtime=True,
    )


@pytest.mark.asyncio
async def test_one_revert_does_not_reach_the_threshold() -> None:
    # A driver reset can raise a limit on its own, so a single observation is not a breach.
    incentive = _build_incentive(_redis_with_reverts(1))

    assert await incentive._power_cap_reverted(_make_job()) is None


@pytest.mark.asyncio
async def test_repeated_reverts_are_flagged() -> None:
    incentive = _build_incentive(_redis_with_reverts(MIN_POWER_CAP_REVERTS_TO_PENALIZE))

    reverted = await incentive._power_cap_reverted(_make_job())

    assert reverted is not None
    assert reverted.revert_count == MIN_POWER_CAP_REVERTS_TO_PENALIZE
    assert reverted.capped_to_watts == 280
    assert reverted.found_watts == 400


@pytest.mark.asyncio
async def test_gate_fails_open_when_redis_is_down() -> None:
    incentive = _build_incentive(BrokenRedis())

    assert await incentive._power_cap_reverted(_make_job()) is None


@pytest.mark.asyncio
async def test_shadow_mode_keeps_rental_eligibility(monkeypatch) -> None:
    monkeypatch.setattr(settings, "ENABLE_UNRENTED_POWER_CAP_REVERT_LIMIT", False)
    incentive = _build_incentive(_redis_with_reverts(MIN_POWER_CAP_REVERTS_TO_PENALIZE))

    result = await incentive.calculate_executor_score(_make_job())

    assert result.eligible_for_rental_share is True
    assert "reverts_gpu_power_cap" not in "\n".join(result.incentive_logs)


@pytest.mark.asyncio
async def test_shadow_mode_emits_the_measurement_log(monkeypatch, caplog) -> None:
    # The shadow numbers are the whole deliverable of the first deploy.
    monkeypatch.setattr(settings, "ENABLE_UNRENTED_POWER_CAP_REVERT_LIMIT", False)
    incentive = _build_incentive(_redis_with_reverts(MIN_POWER_CAP_REVERTS_TO_PENALIZE))

    with caplog.at_level(logging.INFO):
        await incentive.calculate_executor_score(_make_job())

    breach = next(
        record.msg for record in caplog.records
        if hasattr(record.msg, "extra") and record.msg.extra.get("reason") == "reverts_gpu_power_cap"
    )
    assert "shadow only - flag off" in breach.message
    assert breach.extra["power_cap_revert_count"] == MIN_POWER_CAP_REVERTS_TO_PENALIZE
    assert breach.extra["enforced"] is False
    assert breach.extra["pool"] == "rental_kept_shadow"


@pytest.mark.asyncio
async def test_enforced_drops_rental_eligibility(monkeypatch) -> None:
    monkeypatch.setattr(settings, "ENABLE_UNRENTED_POWER_CAP_REVERT_LIMIT", True)
    incentive = _build_incentive(_redis_with_reverts(MIN_POWER_CAP_REVERTS_TO_PENALIZE))

    result = await incentive.calculate_executor_score(_make_job())

    assert result.eligible_for_rental_share is False
    assert result.mining_score == 0
    assert "reverts_gpu_power_cap" in "\n".join(result.incentive_logs)


@pytest.mark.asyncio
async def test_rented_executor_is_never_gated(monkeypatch) -> None:
    # The gate only withholds the UNRENTED incentive; a rented node earns from the mining pool.
    monkeypatch.setattr(settings, "ENABLE_UNRENTED_POWER_CAP_REVERT_LIMIT", True)
    incentive = _build_incentive(_redis_with_reverts(MIN_POWER_CAP_REVERTS_TO_PENALIZE))

    assert await incentive._power_cap_reverted(_make_job(is_rented=True)) is not None  # gate is state-only
    result = await incentive.calculate_executor_score(_make_job(is_rented=True))

    assert "reverts_gpu_power_cap" not in "\n".join(result.incentive_logs)


@pytest.mark.asyncio
async def test_enforcement_ships_off() -> None:
    # Rollout contract: this gate infers provider intent, so it starts in shadow.
    default = type(settings).model_fields["ENABLE_UNRENTED_POWER_CAP_REVERT_LIMIT"].default

    assert default is False
