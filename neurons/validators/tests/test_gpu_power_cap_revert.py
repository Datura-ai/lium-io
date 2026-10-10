"""DAH-2786 — a host that puts its own GPU power limit back loses the unrented incentive.

Case 3 of the three power-cap problems: the cap applies, the validator reads it back, and
then something on the provider's host raises the limit again. The PEARL filler is killed by
the backend grace check, so the node runs no Lium job — and today it still collects the full
unrented incentive.

The revert is observed WHILE the Lium job runs, in the validation check that already skips the
power-limit penalty for a live filler. That is the only moment it can be told apart from a host
that leaves our cap alone and merely resets the limit after the job ends, which harms nothing.
This suite covers the three halves: the rule, the recording, and the incentive gate.
"""

import json
import logging
import time
from unittest.mock import AsyncMock

import pytest
from datura.requests.miner_requests import ExecutorSSHInfo
from helpers import build_services, build_state
from protocol.vc_protocol.compute_requests import RentedExecutorsResponse

from core.config import settings
from incentive.config import IncentiveConfig
from incentive.rental_price import MIN_POWER_CAP_REVERTS_TO_PENALIZE, RentalPriceIncentive
from payload_models.payloads import GpuPowerLimit
from services.const import DEFAULT_JOB_OWNER_LIUM
from services.gpu_power_limit import (
    CAP_REVERT_WINDOW_SECONDS,
    MAX_TRACKED_CAP_REVERTS,
    GpuPowerCapRevert,
    GpuPowerCapRevertHistory,
    GpuPowerRestoreReadResult,
    GpuPowerRestoreRecord,
    _restore_key,
    _revert_key,
    apply_filler_gpu_power_limits,
    detect_cap_revert,
    read_gpu_power_cap_reverts,
    restore_tracked_gpu_power_limits,
)
from services.task.checks import gpu_power_limit as check_module
from services.task.checks.gpu_power_limit import GpuPowerLimitCheck
from services.task.pipeline import ContextState
from services.task_service import JobResult

EXECUTOR_ID = "executor-1"
POD_ID = "pod-1"
H200 = "NVIDIA H200"  # base model H200 is rental-eligible by default
# matches default_executor().uuid in helpers.py — the executor context_factory builds
CHECK_EXECUTOR_UUID = "executor-123"


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


def _observe(
    watts: int, capped_to_watts: int | None, found: int, default: int | None = 450
) -> GpuPowerCapRevert | None:
    return detect_cap_revert(
        _restore_record(watts, capped_to_watts),
        current_watts=found,
        default_watts=default,
        observed_at=1.0,
    )


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


# ---------------------------- recording while the Lium job runs ----------------------------


class FakeRun:
    def __init__(self, stdout: str = "", stderr: str = "", exit_status: int = 0) -> None:
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
    def __init__(self, initial: dict[str, str] | None = None) -> None:
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


def _stored_history(redis: FakeRedis, executor_id: str = CHECK_EXECUTOR_UUID) -> GpuPowerCapRevertHistory:
    return GpuPowerCapRevertHistory.model_validate_json(redis.store[_revert_key(executor_id)])


def _stored_restore_record(
    gpu_uuid: str = "GPU-a", executor_id: str = EXECUTOR_ID, pod_id: str = POD_ID
) -> str:
    record = _restore_record(watts=400, capped_to_watts=280)
    return record.model_copy(
        update={"gpu_uuid": gpu_uuid, "executor_id": executor_id, "pod_id": pod_id}
    ).model_dump_json()


def _record(
    gpu_uuid: str = "GPU-a",
    executor_id: str = CHECK_EXECUTOR_UUID,
    pod_id: str = POD_ID,
    capped_to_watts: int | None = 280,
) -> GpuPowerRestoreRecord:
    return GpuPowerRestoreRecord(
        gpu_uuid=gpu_uuid,
        watts=400,
        capped_to_watts=capped_to_watts,
        pod_id=pod_id,
        executor_id=executor_id,
        capped_at=time.time(),
    )


def _running_filler_state(*live_watts_by_uuid: tuple[str, float]) -> ContextState:
    """Node state while a Lium filler runs: GPU readings plus the owner flag the check keys off."""
    return build_state(
        gpu_model="NVIDIA L40S",
        gpu_count=len(live_watts_by_uuid),
        gpu_details=[
            {
                "name": "NVIDIA L40S",
                "uuid": gpu_uuid,
                "power_limit": watts,
                "power_default_limit": 400.0,
                "power_max_limit": 450.0,
            }
            for gpu_uuid, watts in live_watts_by_uuid
        ],
        rented_data=RentedExecutorsResponse(
            executors={},
            default_job_owner_by_executor={CHECK_EXECUTOR_UUID: DEFAULT_JOB_OWNER_LIUM},
        ),
    )


async def _run_check_while_filler_runs(
    context_factory,
    monkeypatch: pytest.MonkeyPatch,
    redis: FakeRedis,
    records: list[GpuPowerRestoreRecord],
    live_watts_by_uuid: list[tuple[str, float]],
) -> None:
    monkeypatch.setattr(
        check_module,
        "read_gpu_power_restore_records",
        AsyncMock(return_value=GpuPowerRestoreReadResult(records=records, read_failed=False)),
    )
    ctx = context_factory(
        state=_running_filler_state(*live_watts_by_uuid),
        services=build_services(redis=redis),
    )
    result = await GpuPowerLimitCheck().run(ctx)

    # the check still passes: our own cap is why the limit is low, and a revert never uncaps
    assert result.passed is True


@pytest.mark.asyncio
async def test_a_live_revert_is_recorded(context_factory, monkeypatch) -> None:
    # The host raised our 280 W cap back to the card default while our job is still running.
    redis = FakeRedis()

    await _run_check_while_filler_runs(
        context_factory, monkeypatch, redis, [_record()], [("GPU-a", 400.0)]
    )

    history = _stored_history(redis)
    assert len(history.reverts) == 1
    assert history.reverts[0].gpu_uuid == "GPU-a"
    assert history.reverts[0].capped_to_watts == 280
    assert history.reverts[0].found_watts == 400
    assert redis.expiries[_revert_key(CHECK_EXECUTOR_UUID)] == CAP_REVERT_WINDOW_SECONDS


@pytest.mark.asyncio
async def test_a_cap_that_holds_records_nothing(context_factory, monkeypatch) -> None:
    redis = FakeRedis()

    await _run_check_while_filler_runs(
        context_factory, monkeypatch, redis, [_record()], [("GPU-a", 280.0)]
    )

    assert _revert_key(CHECK_EXECUTOR_UUID) not in redis.store


@pytest.mark.asyncio
async def test_a_reset_after_the_job_is_never_seen(context_factory, monkeypatch) -> None:
    # The whole reason detection lives here: a host that leaves our cap alone while the job runs
    # and only resets the limit once the container is gone has harmed nothing. With no Lium job
    # on the node the check takes the normal path and never looks at the revert history.
    redis = FakeRedis()
    monkeypatch.setattr(
        check_module,
        "read_gpu_power_restore_records",
        AsyncMock(return_value=GpuPowerRestoreReadResult(records=[_record()], read_failed=False)),
    )
    state = build_state(
        gpu_model="NVIDIA L40S",
        gpu_count=1,
        gpu_details=[{
            "name": "NVIDIA L40S", "uuid": "GPU-a",
            "power_limit": 400.0, "power_default_limit": 400.0, "power_max_limit": 450.0,
        }],
        rented_data=RentedExecutorsResponse(executors={}, default_job_owner_by_executor={}),
    )
    ctx = context_factory(state=state, services=build_services(redis=redis))

    await GpuPowerLimitCheck().run(ctx)

    assert _revert_key(CHECK_EXECUTOR_UUID) not in redis.store


@pytest.mark.asyncio
async def test_a_whole_node_revert_counts_once(context_factory, monkeypatch) -> None:
    # A host guard resets every GPU in the same tick. Counting eight of those as eight breaches
    # would zero an 8x node on a single event.
    redis = FakeRedis()

    await _run_check_while_filler_runs(
        context_factory, monkeypatch, redis,
        [_record(gpu_uuid="GPU-a"), _record(gpu_uuid="GPU-b")],
        [("GPU-a", 400.0), ("GPU-b", 400.0)],
    )

    assert len(_stored_history(redis).reverts) == 1


@pytest.mark.asyncio
async def test_the_same_job_is_not_counted_on_every_cycle(context_factory, monkeypatch) -> None:
    # The check runs every validation cycle for as long as the filler lives. One killed job is
    # one breach, however many cycles observe it.
    redis = FakeRedis()

    for _ in range(3):
        await _run_check_while_filler_runs(
            context_factory, monkeypatch, redis, [_record()], [("GPU-a", 400.0)]
        )

    assert len(_stored_history(redis).reverts) == 1


@pytest.mark.asyncio
async def test_another_executors_record_is_never_charged_here(context_factory, monkeypatch) -> None:
    # One host can carry several executors; a neighbour's record must not land on this one.
    redis = FakeRedis()

    await _run_check_while_filler_runs(
        context_factory, monkeypatch, redis,
        [_record(executor_id="executor-next-door")], [("GPU-a", 400.0)],
    )

    assert _revert_key(CHECK_EXECUTOR_UUID) not in redis.store


@pytest.mark.asyncio
async def test_a_failed_read_never_overwrites_the_stored_history(context_factory, monkeypatch) -> None:
    # A transient GET failure used to look like "this executor has no reverts", and the write
    # that followed replaced the real history with just this job's reverts.
    class ReadBlindRedis(FakeRedis):
        async def get(self, key: str) -> str | None:
            raise ConnectionError("redis down")

    earlier = GpuPowerCapRevertHistory(
        reverts=[
            GpuPowerCapRevert(
                observed_at=time.time() - 60, pod_id="pod-earlier", gpu_uuid="GPU-a",
                capped_to_watts=280, found_watts=400,
            )
        ]
    )
    redis = ReadBlindRedis({_revert_key(CHECK_EXECUTOR_UUID): earlier.model_dump_json()})

    await _run_check_while_filler_runs(
        context_factory, monkeypatch, redis, [_record(pod_id="pod-now")], [("GPU-a", 400.0)]
    )

    kept = GpuPowerCapRevertHistory.model_validate_json(redis.store[_revert_key(CHECK_EXECUTOR_UUID)])
    assert [revert.pod_id for revert in kept.reverts] == ["pod-earlier"]


@pytest.mark.asyncio
async def test_recording_survives_redis_being_down(context_factory, monkeypatch) -> None:
    # Redis resilience: the check must still pass, and the filler must keep running.
    await _run_check_while_filler_runs(
        context_factory, monkeypatch, BrokenRedis(), [_record()], [("GPU-a", 400.0)]
    )


@pytest.mark.asyncio
async def test_the_dry_run_pipeline_writes_nothing(context_factory, monkeypatch) -> None:
    redis = FakeRedis()
    monkeypatch.setattr(
        check_module,
        "read_gpu_power_restore_records",
        AsyncMock(return_value=GpuPowerRestoreReadResult(records=[_record()], read_failed=False)),
    )
    ctx = context_factory(
        state=_running_filler_state(("GPU-a", 400.0)), services=build_services(redis=redis)
    )

    await GpuPowerLimitCheck(restore_stale_caps=False).run(ctx)

    assert redis.store == {}


@pytest.mark.asyncio
async def test_old_reverts_drop_out_of_the_window(context_factory, monkeypatch) -> None:
    stale = GpuPowerCapRevert(
        observed_at=time.time() - CAP_REVERT_WINDOW_SECONDS - 60,
        pod_id="pod-stale", gpu_uuid="GPU-a", capped_to_watts=280, found_watts=400,
    )
    fresh = GpuPowerCapRevert(
        observed_at=time.time() - 60,
        pod_id="pod-fresh", gpu_uuid="GPU-a", capped_to_watts=280, found_watts=400,
    )
    redis = FakeRedis({
        _revert_key(CHECK_EXECUTOR_UUID): GpuPowerCapRevertHistory(reverts=[stale, fresh]).model_dump_json()
    })

    await _run_check_while_filler_runs(
        context_factory, monkeypatch, redis, [_record(pod_id="pod-now")], [("GPU-a", 400.0)]
    )

    assert [revert.pod_id for revert in _stored_history(redis).reverts] == ["pod-fresh", "pod-now"]


@pytest.mark.asyncio
async def test_history_is_bounded(context_factory, monkeypatch) -> None:
    now = time.time()
    redis = FakeRedis({
        _revert_key(CHECK_EXECUTOR_UUID): GpuPowerCapRevertHistory(
            reverts=[
                GpuPowerCapRevert(
                    observed_at=now - index, pod_id=f"seed-{index}", gpu_uuid="GPU-a",
                    capped_to_watts=280, found_watts=400,
                )
                for index in range(MAX_TRACKED_CAP_REVERTS + 10)
            ]
        ).model_dump_json()
    })

    await _run_check_while_filler_runs(
        context_factory, monkeypatch, redis, [_record(pod_id="pod-now")], [("GPU-a", 400.0)]
    )

    assert len(_stored_history(redis).reverts) == MAX_TRACKED_CAP_REVERTS


# ---------------------------- the applied-cap stamp ----------------------------


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


def _build_incentive(redis: FakeRedis) -> RentalPriceIncentive:
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
