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
from payload_models.payloads import GpuPowerLimit
from services.gpu_power_limit import (
    CAP_REVERT_WINDOW_SECONDS,
    MAX_TRACKED_CAP_REVERTS,
    GpuPowerCapRevert,
    GpuPowerCapRevertHistory,
    GpuPowerRestoreRecord,
    GpuPowerState,
    _restore_key,
    _revert_key,
    _detect_cap_revert,
    apply_filler_gpu_power_limits,
    read_gpu_power_cap_reverts,
    restore_tracked_gpu_power_limits,
)
from services.task_service import JobResult

EXECUTOR_ID = "executor-1"
POD_ID = "pod-1"
H200 = "NVIDIA H200"  # base model H200 is rental-eligible by default


# ---------------------------- _detect_cap_revert (pure) ----------------------------


def _restore_record(watts: int, capped_to_watts: int | None) -> GpuPowerRestoreRecord:
    return GpuPowerRestoreRecord(
        gpu_uuid="GPU-a",
        watts=watts,
        capped_to_watts=capped_to_watts,
        pod_id=POD_ID,
        executor_id=EXECUTOR_ID,
        capped_at=time.time(),
    )


def _state(current_watts: int, default_watts: int | None = 450) -> GpuPowerState:
    return GpuPowerState(
        current_watts=current_watts, min_watts=100, max_watts=600, default_watts=default_watts
    )


def _observe(watts: int, capped_to_watts: int | None, found: int, default: int | None = 450):
    return _detect_cap_revert(_restore_record(watts, capped_to_watts), _state(found, default), observed_at=1.0)


# RTX 4090 numbers throughout: default 450 W, our cap 315 W (0.70), uncapped line 405 W (0.90).


def test_limit_pinned_just_above_our_floor_is_a_revert() -> None:
    # The prod shape: the host guard pins 409 W, one point above the 0.9 line we kill at.
    observation = _observe(watts=409, capped_to_watts=315, found=409)

    assert observation is not None
    assert observation.capped_to_watts == 315
    assert observation.found_watts == 409
    assert observation.pod_id == POD_ID


def test_limit_back_at_the_card_default_is_a_revert() -> None:
    assert _observe(watts=409, capped_to_watts=315, found=450) is not None


def test_host_floor_below_the_pre_cap_reading_is_still_a_revert() -> None:
    # Guard script A restores the DEFAULT every 30 s during cold start, then pins 0.91 of it.
    # The pre-cap reading is then the default and the live limit sits under it - still an
    # uncapped run, and a pre-cap comparison alone would miss it.
    assert _observe(watts=450, capped_to_watts=315, found=409) is not None


def test_cap_that_still_holds_is_not_a_revert() -> None:
    assert _observe(watts=409, capped_to_watts=315, found=315) is None


def test_limit_below_the_uncapped_line_is_not_a_revert() -> None:
    # The job still ran capped, so it did the work it is paid for. Drift is not a breach.
    assert _observe(watts=409, capped_to_watts=315, found=350) is None


def test_the_line_is_not_rounded_below_the_backend_kill_ratio() -> None:
    # 5090: 0.9 x 575 = 517.5. A limit of 517 W is 0.8991 of default, which the backend guard
    # lets live - penalizing a run nobody killed would be a rule of our own.
    assert _observe(watts=518, capped_to_watts=402, found=517, default=575) is None
    assert _observe(watts=518, capped_to_watts=402, found=518, default=575) is not None


def test_cap_that_never_lowered_the_limit_is_not_a_revert() -> None:
    # A miner running below our target is capped upwards; reading our own target back at
    # restore time is not the host raising it.
    assert _observe(watts=250, capped_to_watts=315, found=315) is None


def test_a_cap_that_never_landed_is_not_a_revert() -> None:
    # The whole DAH-2715 population: the container cannot cap at all, so the failed apply is
    # undone through the same restore path. No stamped target, no accusation.
    assert _observe(watts=409, capped_to_watts=None, found=409) is None


def test_unknown_default_falls_back_to_the_pre_cap_limit() -> None:
    # Some GPUs report "[N/A]" for the default limit. The miner's own pre-cap value is the line.
    assert _observe(watts=409, capped_to_watts=315, found=409, default=None) is not None
    assert _observe(watts=409, capped_to_watts=315, found=380, default=None) is None


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


def _set_ok(readback_watts: int, persistence: str = "Enabled") -> list[FakeRun]:
    return [FakeRun(), FakeRun(), FakeRun(stdout=f"{readback_watts}.00, {persistence}\n")]


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


def _stored_restore_record(
    gpu_uuid: str = "GPU-a", executor_id: str = EXECUTOR_ID, pod_id: str = POD_ID
) -> str:
    record = _restore_record(watts=400, capped_to_watts=280)
    return record.model_copy(
        update={"gpu_uuid": gpu_uuid, "executor_id": executor_id, "pod_id": pod_id}
    ).model_dump_json()


@pytest.mark.asyncio
async def test_restore_records_a_revert_when_the_limit_came_back() -> None:
    ssh = fake_ssh(FakeRun(stdout=REVERTED_STATE_CSV), *_set_ok(400))
    redis = FakeRedis({_restore_key("GPU-a"): _stored_restore_record()})

    await restore_tracked_gpu_power_limits(ssh, redis, ["GPU-a"])

    history = _stored_history(redis)
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
async def test_a_whole_node_revert_counts_once() -> None:
    # A host guard resets every GPU in the same tick. Counting eight of those as eight
    # breaches would zero an 8x node on a single event.
    state_csv = "GPU-a, 400, 400, 100, 400\nGPU-b, 400, 400, 100, 400\n"
    ssh = fake_ssh(FakeRun(stdout=state_csv), *_set_ok(400), *_set_ok(400))
    redis = FakeRedis({
        _restore_key("GPU-a"): _stored_restore_record(),
        _restore_key("GPU-b"): _stored_restore_record(gpu_uuid="GPU-b"),
    })

    await restore_tracked_gpu_power_limits(ssh, redis, ["GPU-a", "GPU-b"])

    history = _stored_history(redis)
    assert len(history.reverts) == 1
    assert history.reverts[0].pod_id == POD_ID


@pytest.mark.asyncio
async def test_two_executors_on_one_host_each_get_their_own_revert() -> None:
    # A whole-host restore sweeps every executor on the box; one provider's breach must not
    # land on its neighbour's record.
    other_executor = "executor-2"
    state_csv = "GPU-a, 400, 400, 100, 400\nGPU-b, 400, 400, 100, 400\n"
    ssh = fake_ssh(FakeRun(stdout=state_csv), *_set_ok(400), *_set_ok(400))
    redis = FakeRedis({
        _restore_key("GPU-a"): _stored_restore_record(),
        _restore_key("GPU-b"): _stored_restore_record(gpu_uuid="GPU-b", executor_id=other_executor),
    })

    await restore_tracked_gpu_power_limits(ssh, redis, ["GPU-a", "GPU-b"])

    assert len(_stored_history(redis).reverts) == 1
    other = GpuPowerCapRevertHistory.model_validate_json(redis.store[_revert_key(other_executor)])
    assert [revert.gpu_uuid for revert in other.reverts] == ["GPU-b"]


@pytest.mark.asyncio
async def test_reverts_accumulate_and_old_ones_drop_out_of_the_window() -> None:
    stale = GpuPowerCapRevert(
        observed_at=time.time() - CAP_REVERT_WINDOW_SECONDS - 60,
        pod_id="pod-stale",
        gpu_uuid="GPU-a",
        capped_to_watts=280,
        found_watts=400,
    )
    fresh = GpuPowerCapRevert(observed_at=time.time() - 60, pod_id="pod-fresh", gpu_uuid="GPU-a", capped_to_watts=280, found_watts=400)
    history = GpuPowerCapRevertHistory(reverts=[stale, fresh])
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
        GpuPowerCapRevert(
            observed_at=now - index, pod_id=f"seed-{index}", gpu_uuid="GPU-a", capped_to_watts=280, found_watts=400
        )
        for index in range(MAX_TRACKED_CAP_REVERTS + 10)
    ]
    ssh = fake_ssh(FakeRun(stdout=REVERTED_STATE_CSV), *_set_ok(400))
    redis = FakeRedis({
        _restore_key("GPU-a"): _stored_restore_record(),
        _revert_key(EXECUTOR_ID): GpuPowerCapRevertHistory(reverts=reverts).model_dump_json(),
    })

    await restore_tracked_gpu_power_limits(ssh, redis, ["GPU-a"])

    assert len(_stored_history(redis).reverts) == MAX_TRACKED_CAP_REVERTS


@pytest.mark.asyncio
async def test_a_failed_apply_is_never_recorded_as_a_revert() -> None:
    # The DAH-2715 population: the container cannot run nvidia-smi -pl at all. The failed apply
    # is undone through the same restore path, and the undo reads the untouched pre-cap limit.
    # Without the post-set stamp, every retry of a node that CANNOT cap would count as a node
    # that WILL NOT cap, and two retries reach the threshold.
    state_csv = "GPU-a, 400, 400, 100, 400\n"
    ssh = fake_ssh(
        FakeRun(stdout=state_csv),          # plan: state query
        FakeRun(), FakeRun(exit_status=4),  # -pm, then -pl fails (no readback follows)
        FakeRun(stdout=state_csv),          # undo: state query
        *_set_ok(400),                      # undo: restore to the recorded 400 W
    )
    redis = FakeRedis()

    applied = await apply_filler_gpu_power_limits(
        ssh, [GpuPowerLimit(gpu_uuid="GPU-a", watts=280)], redis, POD_ID, EXECUTOR_ID
    )

    assert applied is False
    assert _revert_key(EXECUTOR_ID) not in redis.store


@pytest.mark.asyncio
async def test_a_failed_apply_clears_a_stale_claim_from_an_earlier_cap() -> None:
    # A record frozen by an earlier FAILED restore still carries the watts that cap applied.
    # If the next attempt cannot cap, the undo restores against an untouched high limit - and
    # without clearing the claim first, that reads as the host reverting us.
    state_csv = "GPU-a, 400, 400, 100, 400\n"
    ssh = fake_ssh(
        FakeRun(stdout=state_csv),          # plan: state query
        FakeRun(), FakeRun(exit_status=4),  # -pm, then -pl fails (no readback follows)
        FakeRun(stdout=state_csv),          # undo: state query
        *_set_ok(400),                      # undo: restore to the frozen 400 W
    )
    redis = FakeRedis({_restore_key("GPU-a"): _stored_restore_record(pod_id="pod-earlier")})

    applied = await apply_filler_gpu_power_limits(
        ssh, [GpuPowerLimit(gpu_uuid="GPU-a", watts=280)], redis, POD_ID, EXECUTOR_ID
    )

    assert applied is False
    assert _revert_key(EXECUTOR_ID) not in redis.store


@pytest.mark.asyncio
async def test_a_verified_cap_is_stamped_so_a_later_revert_is_attributable() -> None:
    state_csv = "GPU-a, 400, 400, 100, 400\n"
    ssh = fake_ssh(FakeRun(stdout=state_csv), *_set_ok(280))
    redis = FakeRedis()

    applied = await apply_filler_gpu_power_limits(
        ssh, [GpuPowerLimit(gpu_uuid="GPU-a", watts=280)], redis, POD_ID, EXECUTOR_ID
    )

    assert applied is True
    stored = GpuPowerRestoreRecord.model_validate_json(redis.store[_restore_key("GPU-a")])
    assert stored.watts == 400  # the way back is still the miner's own limit
    assert stored.capped_to_watts == 280


@pytest.mark.asyncio
async def test_a_cap_without_persistence_mode_is_never_stamped() -> None:
    # DAH-2702: with persistence off the driver unloads on an idle GPU and the stock limit
    # comes back on its own. At teardown that is indistinguishable from a provider raising it,
    # so two healthy jobs on such a host would cost the provider the idle payout.
    state_csv = "GPU-a, 400, 400, 100, 400\n"
    ssh = fake_ssh(FakeRun(stdout=state_csv), *_set_ok(280, persistence="Disabled"))
    redis = FakeRedis()

    applied = await apply_filler_gpu_power_limits(
        ssh, [GpuPowerLimit(gpu_uuid="GPU-a", watts=280)], redis, POD_ID, EXECUTOR_ID
    )

    assert applied is True  # the cap holds for now, so the filler still runs
    stored = GpuPowerRestoreRecord.model_validate_json(redis.store[_restore_key("GPU-a")])
    assert stored.capped_to_watts is None


@pytest.mark.asyncio
async def test_the_stamp_moves_a_frozen_record_onto_the_capping_pod() -> None:
    # A record frozen by an earlier failed restore still names the pod that first capped the
    # GPU. Reverts are deduplicated per job, so leaving the old pod on it would make every
    # later reverted job look like one already counted.
    state_csv = "GPU-a, 400, 400, 100, 400\n"
    ssh = fake_ssh(FakeRun(stdout=state_csv), *_set_ok(280))
    redis = FakeRedis({_restore_key("GPU-a"): _stored_restore_record(pod_id="pod-earlier")})

    await apply_filler_gpu_power_limits(
        ssh, [GpuPowerLimit(gpu_uuid="GPU-a", watts=280)], redis, POD_ID, EXECUTOR_ID
    )

    stored = GpuPowerRestoreRecord.model_validate_json(redis.store[_restore_key("GPU-a")])
    assert stored.pod_id == POD_ID
    assert stored.watts == 400  # the frozen way back is untouched


@pytest.mark.asyncio
async def test_our_own_restore_is_never_read_as_a_host_revert() -> None:
    # The restore raises the limit itself. If clearing the record afterwards fails, the record
    # outlives it — and the next pass would read OUR raise as the provider raising it, taking
    # the idle payout from a host whose cap held all along.
    class DeleteBlindRedis(FakeRedis):
        async def delete(self, key: str) -> None:
            raise ConnectionError("redis down")

    held_state_csv = "GPU-a, 280, 400, 100, 400\n"   # the cap held for the whole job
    redis = DeleteBlindRedis({_restore_key("GPU-a"): _stored_restore_record()})
    ssh = fake_ssh(FakeRun(stdout=held_state_csv), *_set_ok(400))
    await restore_tracked_gpu_power_limits(ssh, redis, ["GPU-a"])

    # second pass: the record survived and the GPU now sits at the 400 W we restored
    ssh = fake_ssh(FakeRun(stdout=REVERTED_STATE_CSV), *_set_ok(400))
    await restore_tracked_gpu_power_limits(ssh, redis, ["GPU-a"])

    assert _revert_key(EXECUTOR_ID) not in redis.store


@pytest.mark.asyncio
async def test_a_retried_restore_does_not_count_the_same_job_twice() -> None:
    # A failed restore keeps its record on purpose so a safety net can retry it. The retry sees
    # the same high limit and must not turn one killed job into two breaches.
    redis = FakeRedis({_restore_key("GPU-a"): _stored_restore_record()})

    for _ in range(2):
        ssh = fake_ssh(FakeRun(stdout=REVERTED_STATE_CSV), FakeRun(), FakeRun(), FakeRun(stdout="399.00, Enabled\n"))
        await restore_tracked_gpu_power_limits(ssh, redis, ["GPU-a"])

    assert _restore_key("GPU-a") in redis.store  # the failed restore kept its record
    assert len(_stored_history(redis).reverts) == 1


@pytest.mark.asyncio
async def test_restore_still_succeeds_when_redis_cannot_store_the_revert() -> None:
    # Teardown must never be blocked by the bookkeeping this gate needs.
    ssh = fake_ssh(FakeRun(stdout=REVERTED_STATE_CSV), *_set_ok(400))
    redis = FakeRedis({_restore_key("GPU-a"): _stored_restore_record()})
    redis.set = BrokenRedis().set

    restored = await restore_tracked_gpu_power_limits(ssh, redis, ["GPU-a"])

    assert restored == 1


# ---------------------------- read_gpu_power_cap_reverts ----------------------------


@pytest.mark.asyncio
async def test_read_returns_only_reverts_inside_the_window() -> None:
    now = time.time()
    history = GpuPowerCapRevertHistory(
        reverts=[
            GpuPowerCapRevert(observed_at=now - CAP_REVERT_WINDOW_SECONDS - 1, pod_id="pod-old", gpu_uuid="GPU-a", capped_to_watts=280, found_watts=400),
            GpuPowerCapRevert(observed_at=now - 10, pod_id="pod-new", gpu_uuid="GPU-b", capped_to_watts=280, found_watts=400),
        ],
    )
    redis = FakeRedis({_revert_key(EXECUTOR_ID): history.model_dump_json()})

    reverts = await read_gpu_power_cap_reverts(redis, EXECUTOR_ID)

    assert [revert.gpu_uuid for revert in reverts] == ["GPU-b"]


@pytest.mark.asyncio
async def test_read_is_empty_without_a_record() -> None:
    assert await read_gpu_power_cap_reverts(FakeRedis(), EXECUTOR_ID) == []


@pytest.mark.asyncio
async def test_a_failed_read_never_overwrites_the_stored_history() -> None:
    # A transient GET failure used to look like "this executor has no reverts", and the write
    # that followed replaced the real history with just this job's reverts.
    class ReadBlindRedis(FakeRedis):
        async def get(self, key: str) -> str | None:
            if key == _revert_key(EXECUTOR_ID):
                raise ConnectionError("redis down")
            return self.store.get(key)

    earlier = GpuPowerCapRevertHistory(
        reverts=[
            GpuPowerCapRevert(
                observed_at=time.time() - 60, pod_id="pod-earlier", gpu_uuid="GPU-a",
                capped_to_watts=280, found_watts=400,
            )
        ]
    )
    redis = ReadBlindRedis({
        _restore_key("GPU-a"): _stored_restore_record(),
        _revert_key(EXECUTOR_ID): earlier.model_dump_json(),
    })
    ssh = fake_ssh(FakeRun(stdout=REVERTED_STATE_CSV), *_set_ok(400))

    await restore_tracked_gpu_power_limits(ssh, redis, ["GPU-a"])

    kept = GpuPowerCapRevertHistory.model_validate_json(redis.store[_revert_key(EXECUTOR_ID)])
    assert [revert.pod_id for revert in kept.reverts] == ["pod-earlier"]


@pytest.mark.asyncio
async def test_the_stamp_moves_a_frozen_record_onto_the_capping_executor() -> None:
    # A GPU whose record was frozen under an earlier executor must not file its next breach
    # against that executor - the provider running it now is the one being judged.
    state_csv = "GPU-a, 400, 400, 100, 400\n"
    ssh = fake_ssh(FakeRun(stdout=state_csv), *_set_ok(280))
    redis = FakeRedis({
        _restore_key("GPU-a"): _stored_restore_record(executor_id="executor-earlier", pod_id="pod-earlier")
    })

    await apply_filler_gpu_power_limits(
        ssh, [GpuPowerLimit(gpu_uuid="GPU-a", watts=280)], redis, POD_ID, EXECUTOR_ID
    )

    stored = GpuPowerRestoreRecord.model_validate_json(redis.store[_restore_key("GPU-a")])
    assert stored.executor_id == EXECUTOR_ID
    assert stored.pod_id == POD_ID


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
        reverts=[
            GpuPowerCapRevert(
                observed_at=now - 60 * index, pod_id=f"pod-{index}", gpu_uuid="GPU-a",
                capped_to_watts=280, found_watts=400,
            )
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
