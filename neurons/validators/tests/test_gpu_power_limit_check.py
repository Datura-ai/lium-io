from __future__ import annotations

import time
from unittest.mock import AsyncMock

import pytest
from helpers import build_state
from protocol.vc_protocol.compute_requests import RentedExecutorsResponse
from services.gpu_power_limit import (
    STALE_CAP_GRACE_SECONDS,
    GpuPowerRestoreReadResult,
    GpuPowerRestoreRecord,
)
from services.task.checks import gpu_power_limit as check_module
from services.task.checks.gpu_power_limit import GpuPowerLimitCheck
from services.task.pipeline import ContextState

# Matches default_executor().uuid in helpers.py — the executor the context_factory builds.
_EXECUTOR_UUID = "executor-123"

_OLD_CAP_AGE_SECONDS = STALE_CAP_GRACE_SECONDS + 60


def _restore_record(
    gpu_uuid: str = "GPU-abc",
    executor_id: str = _EXECUTOR_UUID,
    age_seconds: float = _OLD_CAP_AGE_SECONDS,
) -> GpuPowerRestoreRecord:
    return GpuPowerRestoreRecord(
        gpu_uuid=gpu_uuid,
        watts=350,
        pod_id="pod-1",
        executor_id=executor_id,
        capped_at=time.time() - age_seconds,
    )


def _read_result(
    *records: GpuPowerRestoreRecord, read_failed: bool = False
) -> GpuPowerRestoreReadResult:
    return GpuPowerRestoreReadResult(records=list(records), read_failed=read_failed)


@pytest.fixture
def read_records_mock(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    # DAH-2356 safety net: on a below-floor reading the check consults Redis for this validator's
    # own stale filler caps. Tests drive it through this mock (ctx.services.redis is None here).
    mock = AsyncMock(return_value=_read_result())
    monkeypatch.setattr(check_module, "read_gpu_power_restore_records", mock)
    return mock


@pytest.fixture
def restore_mock(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    mock = AsyncMock(return_value=1)  # records restored
    monkeypatch.setattr(check_module, "restore_tracked_gpu_power_limits", mock)
    return mock


def _state(
    current_limit: float | None,
    default_limit: float | None,
    max_limit: float = 450,
    count: int = 1,
    default_job_owner: str | None = None,
    rented_data_known: bool = True,
) -> ContextState:
    details = [
        {
            "name": "NVIDIA L40S",
            "uuid": "GPU-abc" if index == 0 else f"GPU-abc-{index}",
            "power_limit": current_limit,
            "power_default_limit": default_limit,
            "power_max_limit": max_limit,
        }
        for index in range(count)
    ]
    rented_data = None
    if rented_data_known:
        owner_map = {_EXECUTOR_UUID: default_job_owner} if default_job_owner is not None else {}
        rented_data = RentedExecutorsResponse(
            executors={},
            default_job_owner_by_executor=owner_map,
        )
    return build_state(
        gpu_model="NVIDIA L40S",
        gpu_count=count,
        gpu_details=details,
        rented_data=rented_data,
    )


@pytest.mark.asyncio
async def test_power_limit_passes_when_current_is_close_to_default(context_factory, read_records_mock) -> None:
    ctx = context_factory(state=_state(current_limit=320, default_limit=350))
    result = await GpuPowerLimitCheck().run(ctx)
    assert result.passed is True
    assert result.event.reason_code == "GPU_POWER_LIMIT_OK"
    read_records_mock.assert_not_awaited()  # healthy nodes cost zero Redis traffic


@pytest.mark.asyncio
async def test_power_limit_rejects_below_threshold(context_factory, read_records_mock) -> None:
    ctx = context_factory(state=_state(current_limit=105, default_limit=350))
    result = await GpuPowerLimitCheck().run(ctx)
    assert result.passed is False
    assert result.event.reason_code == "GPU_POWER_LIMIT_BELOW_DEFAULT"
    assert result.updates["score"] == 0.0
    assert result.updates["job_score"] == 0.0
    assert result.event.what_we_saw["rejected_gpus"][0]["power_limit_ratio"] == 0.3


@pytest.mark.asyncio
async def test_power_limit_allows_exact_threshold(context_factory, read_records_mock) -> None:
    ctx = context_factory(state=_state(current_limit=315, default_limit=350))
    result = await GpuPowerLimitCheck().run(ctx)
    assert result.passed is True
    assert result.event.reason_code == "GPU_POWER_LIMIT_OK"


@pytest.mark.asyncio
async def test_power_limit_rejects_between_old_and_new_floor(context_factory, read_records_mock) -> None:
    ctx = context_factory(state=_state(current_limit=300, default_limit=350))
    result = await GpuPowerLimitCheck().run(ctx)
    assert result.passed is False
    assert result.event.reason_code == "GPU_POWER_LIMIT_BELOW_DEFAULT"
    assert result.updates["score"] == 0.0


@pytest.mark.asyncio
async def test_power_limit_passes_when_default_limit_missing(context_factory, read_records_mock) -> None:
    ctx = context_factory(state=_state(current_limit=105, default_limit=None))
    result = await GpuPowerLimitCheck().run(ctx)
    assert result.passed is True
    assert result.event.reason_code == "GPU_POWER_LIMIT_BASELINE_INCOMPLETE"


@pytest.mark.asyncio
async def test_power_limit_skipped_when_lium_default_job_active(context_factory, read_records_mock) -> None:
    # DAH-2356: Lium lowered this node's power for its own idle filler (owner="lium"), so the
    # below-90% limit is expected — skip the penalty; the node keeps its score and stays rentable.
    ctx = context_factory(state=_state(current_limit=105, default_limit=350, default_job_owner="lium"))
    result = await GpuPowerLimitCheck().run(ctx)
    assert result.passed is True
    assert result.event.reason_code == "GPU_POWER_LIMIT_SKIPPED_LIUM_FILLER"
    # DAH-2786: the records ARE read here now — a live filler is the only moment we can catch
    # the host raising our cap back. Reading them never changes this verdict.
    read_records_mock.assert_awaited()


@pytest.mark.asyncio
async def test_power_limit_still_rejects_under_miner_default_job(context_factory, read_records_mock) -> None:
    # The exemption is Lium-only: a miner's own default job must NOT let a provider underpower.
    ctx = context_factory(state=_state(current_limit=105, default_limit=350, default_job_owner="miner"))
    result = await GpuPowerLimitCheck().run(ctx)
    assert result.passed is False
    assert result.event.reason_code == "GPU_POWER_LIMIT_BELOW_DEFAULT"
    assert result.updates["score"] == 0.0


@pytest.mark.asyncio
async def test_power_limit_restores_stale_lium_cap_and_skips_penalty(
    context_factory, read_records_mock, restore_mock
) -> None:
    # DAH-2356 safety net: the below-floor GPU carries this validator's own restore record older
    # than the grace period — an earlier post-filler restore failed. Restore it, no penalty.
    read_records_mock.return_value = _read_result(_restore_record())
    ctx = context_factory(state=_state(current_limit=105, default_limit=350))

    result = await GpuPowerLimitCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == "GPU_POWER_LIMIT_RESTORED_STALE_CAP"
    read_records_mock.assert_awaited_once()
    assert read_records_mock.await_args.args[1] == ["GPU-abc"]
    restore_mock.assert_awaited_once()
    assert restore_mock.await_args.args[2] == ["GPU-abc"]


@pytest.mark.asyncio
async def test_power_limit_skips_penalty_even_when_restore_fails(
    context_factory, read_records_mock, restore_mock
) -> None:
    # A record only exists because WE capped the GPU, so the miner is never penalized for it — even
    # while the restore keeps failing (the frozen record retries next cycle, alertable via warning).
    read_records_mock.return_value = _read_result(_restore_record())
    restore_mock.return_value = 0  # restore keeps failing
    ctx = context_factory(state=_state(current_limit=105, default_limit=350))

    result = await GpuPowerLimitCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == "GPU_POWER_LIMIT_RESTORED_STALE_CAP"


@pytest.mark.asyncio
async def test_power_limit_fresh_record_passes_without_restoring(
    context_factory, read_records_mock, restore_mock
) -> None:
    # A record younger than the grace period may belong to a filler the backend snapshot doesn't
    # report yet — skip the penalty but do NOT uncap what may be a live filler.
    read_records_mock.return_value = _read_result(_restore_record(age_seconds=60))
    ctx = context_factory(state=_state(current_limit=105, default_limit=350))

    result = await GpuPowerLimitCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == "GPU_POWER_LIMIT_RESTORED_STALE_CAP"
    restore_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_power_limit_missing_rented_data_passes_without_restoring(
    context_factory, read_records_mock, restore_mock
) -> None:
    # Without backend data we can't rule out an active Lium filler — skip the penalty (the record
    # is ours), but don't mutate anything.
    read_records_mock.return_value = _read_result(_restore_record())
    ctx = context_factory(state=_state(current_limit=105, default_limit=350, rented_data_known=False))

    result = await GpuPowerLimitCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == "GPU_POWER_LIMIT_RESTORED_STALE_CAP"
    restore_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_power_limit_ignores_record_from_another_executor(
    context_factory, read_records_mock, restore_mock
) -> None:
    # A record written for a different executor must grant no pass here (a miner replaying a capped
    # GPU's uuid in another node's spec gets no exemption) and must not be touched.
    read_records_mock.return_value = _read_result(_restore_record(executor_id="executor-other"))
    ctx = context_factory(state=_state(current_limit=105, default_limit=350))

    result = await GpuPowerLimitCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == "GPU_POWER_LIMIT_BELOW_DEFAULT"
    restore_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_power_limit_rejects_when_only_some_low_gpus_have_records(
    context_factory, read_records_mock, restore_mock
) -> None:
    # GPU-abc is our stale cap, but GPU-abc-1 is below floor with NO record — that one is the
    # miner's own doing, so the normal penalty applies.
    read_records_mock.return_value = _read_result(_restore_record())
    ctx = context_factory(state=_state(current_limit=105, default_limit=350, count=2))

    result = await GpuPowerLimitCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == "GPU_POWER_LIMIT_BELOW_DEFAULT"
    restore_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_power_limit_dry_run_never_mutates(context_factory, read_records_mock, restore_mock) -> None:
    # The dry-run pipeline constructs the check with restore_stale_caps=False: same verdict, but no
    # nvidia-smi -pl and no consumption of the shared restore records.
    read_records_mock.return_value = _read_result(_restore_record())
    ctx = context_factory(state=_state(current_limit=105, default_limit=350))

    result = await GpuPowerLimitCheck(restore_stale_caps=False).run(ctx)

    assert result.passed is True
    assert result.event.reason_code == "GPU_POWER_LIMIT_RESTORED_STALE_CAP"
    restore_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_power_limit_passes_when_redis_read_fails(
    context_factory, read_records_mock, restore_mock
) -> None:
    # A Redis outage makes "no record" indistinguishable from "our own cap whose record we can't
    # read" — never penalize over our own outage; the check simply re-runs next cycle.
    read_records_mock.return_value = _read_result(read_failed=True)
    ctx = context_factory(state=_state(current_limit=105, default_limit=350))

    result = await GpuPowerLimitCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == "GPU_POWER_LIMIT_STATE_UNAVAILABLE"
    assert not result.updates  # no score zeroing
    restore_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_power_limit_rescues_when_read_partially_fails_but_records_cover(
    context_factory, read_records_mock, restore_mock
) -> None:
    # Some keys errored, but the reads that DID succeed cover every below-floor GPU — the normal
    # stale-cap rescue applies as if the read were clean.
    read_records_mock.return_value = _read_result(_restore_record(), read_failed=True)
    ctx = context_factory(state=_state(current_limit=105, default_limit=350))

    result = await GpuPowerLimitCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == "GPU_POWER_LIMIT_RESTORED_STALE_CAP"
    restore_mock.assert_awaited_once()
