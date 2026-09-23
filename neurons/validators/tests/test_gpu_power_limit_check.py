from __future__ import annotations

import time
from unittest.mock import AsyncMock, Mock

import pytest
from core.config import settings
from helpers import build_state
from protocol.vc_protocol.compute_requests import RentedExecutorsResponse
from services.gpu_power_limit import (
    STALE_CAP_GRACE_SECONDS,
    GpuPowerRestoreReadResult,
    GpuPowerRestoreRecord,
    GpuPowerState,
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


def _live_state(
    current_watts: int, *gpu_uuids: str, default_watts: int = 350
) -> dict[str, GpuPowerState]:
    return {
        gpu_uuid: GpuPowerState(
            current_watts=current_watts, min_watts=100, max_watts=450, default_watts=default_watts
        )
        for gpu_uuid in gpu_uuids
    }


@pytest.fixture
def live_power_mock(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    # DAH-3630: the live nvidia-smi read the check takes before charging a GPU under a Lium filler.
    # By default the host still holds both test GPUs below the floor.
    mock = AsyncMock(return_value=_live_state(105, "GPU-abc", "GPU-abc-1"))
    monkeypatch.setattr(check_module, "query_gpu_power_state", mock)
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
    min_limit: float | None = None,
) -> ContextState:
    details = [
        {
            "name": "NVIDIA L40S",
            "uuid": "GPU-abc" if index == 0 else f"GPU-abc-{index}",
            "power_limit": current_limit,
            "power_default_limit": default_limit,
            "power_max_limit": max_limit,
        }
        | ({"power_min_limit": min_limit} if min_limit is not None else {})
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
async def test_power_limit_evidence_carries_the_host_floor(context_factory, read_records_mock) -> None:
    """B-113: the scrape's power_min_limit (the lowest cap the host accepts) travels into the
    check's measurements so a power-cap guard can tell a BIOS clamp from a refused cap; a scrape
    without the field yields None, never a failure."""
    ctx = context_factory(state=_state(current_limit=105, default_limit=350, min_limit=100))
    result = await GpuPowerLimitCheck().run(ctx)
    assert result.event.what_we_saw["rejected_gpus"][0]["power_min_limit"] == 100

    ctx = context_factory(state=_state(current_limit=320, default_limit=350))
    result = await GpuPowerLimitCheck().run(ctx)
    assert result.passed is True
    assert result.event.what_we_saw["measurements"][0]["power_min_limit"] is None


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
async def test_power_limit_skipped_when_lium_default_job_active(
    context_factory, read_records_mock, restore_mock, monkeypatch
) -> None:
    # DAH-2356: Lium lowered this node's power for its own idle filler (owner="lium"), so the
    # below-90% limit is expected — skip the penalty; the node keeps its score and stays rentable.
    # Holds with the DAH-3630 floor enforced: the record proves the cap is ours. Our live cap
    # (a fresh or an old record alike) is never restored under the running filler.
    monkeypatch.setattr(settings, "ENABLE_POWER_FLOOR_FOR_UNCAPPED_LIUM_FILLER_GPUS", True)
    read_records_mock.return_value = _read_result(_restore_record())
    ctx = context_factory(state=_state(current_limit=105, default_limit=350, default_job_owner="lium"))
    result = await GpuPowerLimitCheck().run(ctx)
    assert result.passed is True
    assert result.event.reason_code == "GPU_POWER_LIMIT_SKIPPED_LIUM_FILLER"
    assert not result.updates
    restore_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_host_that_reverted_the_cap_keeps_its_score_under_a_lium_filler(
    context_factory, read_records_mock, monkeypatch
) -> None:
    """DAH-3630 regression: a host that put the limit back to default while PEARL runs must keep PEARL
    and its unrented incentive. A revert gate here (the lium-io#1249 shape) would zero a node whose GPU
    Lium is using. A GPU at or above the floor costs no Redis read."""
    monkeypatch.setattr(settings, "ENABLE_POWER_FLOOR_FOR_UNCAPPED_LIUM_FILLER_GPUS", True)
    ctx = context_factory(state=_state(current_limit=350, default_limit=350, default_job_owner="lium"))
    result = await GpuPowerLimitCheck().run(ctx)
    assert result.passed is True
    assert result.event.reason_code == "GPU_POWER_LIMIT_SKIPPED_LIUM_FILLER"
    assert not result.updates
    read_records_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_host_held_limit_under_a_lium_filler_is_only_logged_while_the_flag_is_off(
    context_factory, read_records_mock, live_power_mock, monkeypatch
) -> None:
    """DAH-3630 regression: the floor for uncapped filler GPUs leaking out of its flag. Off, a below-floor
    GPU with no Lium record passes as before and the breach is one shadow log line."""
    monkeypatch.setattr(settings, "ENABLE_POWER_FLOOR_FOR_UNCAPPED_LIUM_FILLER_GPUS", False)
    mock_logger = Mock()
    monkeypatch.setattr(check_module, "logger", mock_logger)
    ctx = context_factory(state=_state(current_limit=105, default_limit=350, default_job_owner="lium"))

    result = await GpuPowerLimitCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == "GPU_POWER_LIMIT_SKIPPED_LIUM_FILLER"
    assert not result.updates
    (logged,) = mock_logger.info.call_args_list
    assert logged.args[0].extra["reason"] == "power_floor_uncapped_lium_filler_gpu"
    assert logged.args[0].extra["enforced"] is False
    assert [gpu["uuid"] for gpu in logged.args[0].extra["host_limited_gpus"]] == ["GPU-abc"]


@pytest.mark.asyncio
async def test_host_held_limit_under_a_lium_filler_earns_nothing_with_the_flag_on(
    context_factory, read_records_mock, live_power_mock, restore_mock, monkeypatch
) -> None:
    """DAH-3630 regression: the live-filler pass paying for a limit Lium never set. PEARL started uncapped
    (its cap failed) or a filler that never caps runs, and the host holds the GPU below the floor: that
    node is scored like any below-floor node, on the live reading it was charged for."""
    monkeypatch.setattr(settings, "ENABLE_POWER_FLOOR_FOR_UNCAPPED_LIUM_FILLER_GPUS", True)
    live_power_mock.return_value = _live_state(100, "GPU-abc")
    ctx = context_factory(state=_state(current_limit=105, default_limit=350, default_job_owner="lium"))

    result = await GpuPowerLimitCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == "GPU_POWER_LIMIT_BELOW_DEFAULT"
    assert result.updates["score"] == 0.0
    assert result.updates["job_score"] == 0.0
    assert result.event.what_we_saw["rejected_gpus"][0]["power_limit"] == 100
    assert read_records_mock.await_count == 2  # before and after the live read
    restore_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_filler_teardown_after_the_scrape_never_charges_lium_own_cap(
    context_factory, read_records_mock, live_power_mock, restore_mock, monkeypatch
) -> None:
    """DAH-3630 regression: a customer rental pre-empts PEARL after the scrape read GPU-abc at Lium's
    cap and before this check reads Redis. The teardown restored the limit and then deleted the record,
    so the check sees owner lium, a capped reading and no record. The live read shows the restored
    limit, so the node passes instead of scoring 0 for a cap Lium set."""
    monkeypatch.setattr(settings, "ENABLE_POWER_FLOOR_FOR_UNCAPPED_LIUM_FILLER_GPUS", True)
    mock_logger = Mock()
    monkeypatch.setattr(check_module, "logger", mock_logger)
    live_power_mock.return_value = _live_state(350, "GPU-abc")
    ctx = context_factory(state=_state(current_limit=105, default_limit=350, default_job_owner="lium"))

    result = await GpuPowerLimitCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == "GPU_POWER_LIMIT_SKIPPED_LIUM_FILLER"
    assert not result.updates
    live_power_mock.assert_awaited_once()
    mock_logger.info.assert_not_called()  # not counted in the shadow log either
    restore_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_filler_capped_after_the_first_record_read_is_not_charged(
    context_factory, read_records_mock, live_power_mock, monkeypatch
) -> None:
    """DAH-3630: a fresh PEARL apply writes its record, then caps, between the first Redis read and the
    live read. The live read sees Lium's cap; the second Redis read finds its record."""
    monkeypatch.setattr(settings, "ENABLE_POWER_FLOOR_FOR_UNCAPPED_LIUM_FILLER_GPUS", True)
    read_records_mock.side_effect = [_read_result(), _read_result(_restore_record(age_seconds=1))]
    ctx = context_factory(state=_state(current_limit=105, default_limit=350, default_job_owner="lium"))

    result = await GpuPowerLimitCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == "GPU_POWER_LIMIT_SKIPPED_LIUM_FILLER"
    assert not result.updates


@pytest.mark.asyncio
async def test_failed_live_read_never_zeroes_a_node_under_a_lium_filler(
    context_factory, read_records_mock, live_power_mock, monkeypatch
) -> None:
    """DAH-3630: without a live reading the scrape's may predate a teardown; pass this cycle."""
    monkeypatch.setattr(settings, "ENABLE_POWER_FLOOR_FOR_UNCAPPED_LIUM_FILLER_GPUS", True)
    live_power_mock.side_effect = RuntimeError("nvidia-smi power-state query failed")
    ctx = context_factory(state=_state(current_limit=105, default_limit=350, default_job_owner="lium"))

    result = await GpuPowerLimitCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == "GPU_POWER_LIMIT_SKIPPED_LIUM_FILLER"
    assert not result.updates
    read_records_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_frozen_record_from_an_earlier_executor_id_still_proves_the_cap_under_a_lium_filler(
    context_factory, read_records_mock, restore_mock, monkeypatch
) -> None:
    """DAH-3630 regression: a restore that failed (the GPU handle gone) freezes the record with the executor
    id of the job that first capped the GPU; the next PEARL run keeps that record. A provider whose
    executors re-register under new ids would be zeroed for Lium's own cap if only this executor's
    records counted."""
    monkeypatch.setattr(settings, "ENABLE_POWER_FLOOR_FOR_UNCAPPED_LIUM_FILLER_GPUS", True)
    read_records_mock.return_value = _read_result(_restore_record(executor_id="executor-before-reregister"))
    ctx = context_factory(state=_state(current_limit=105, default_limit=350, default_job_owner="lium"))

    result = await GpuPowerLimitCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == "GPU_POWER_LIMIT_SKIPPED_LIUM_FILLER"
    assert not result.updates
    restore_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_only_the_gpu_lium_did_not_cap_is_charged_under_a_lium_filler(
    context_factory, read_records_mock, live_power_mock, monkeypatch
) -> None:
    """DAH-3630: GPU-abc carries our cap record, GPU-abc-1 sits below the floor without one. The node
    fails on GPU-abc-1 alone, so the provider sees the GPU to fix, not the one Lium capped."""
    monkeypatch.setattr(settings, "ENABLE_POWER_FLOOR_FOR_UNCAPPED_LIUM_FILLER_GPUS", True)
    read_records_mock.return_value = _read_result(_restore_record(gpu_uuid="GPU-abc"))
    ctx = context_factory(
        state=_state(current_limit=105, default_limit=350, count=2, default_job_owner="lium")
    )

    result = await GpuPowerLimitCheck().run(ctx)

    assert result.passed is False
    assert [gpu["uuid"] for gpu in result.event.what_we_saw["rejected_gpus"]] == ["GPU-abc-1"]


@pytest.mark.asyncio
async def test_unreadable_records_never_zero_a_node_under_a_lium_filler(
    context_factory, read_records_mock, restore_mock, monkeypatch
) -> None:
    """DAH-3630 regression: a Redis outage read as "Lium did not cap this GPU" would zero every PEARL node
    at once. A failed read passes this cycle, even with the flag on."""
    monkeypatch.setattr(settings, "ENABLE_POWER_FLOOR_FOR_UNCAPPED_LIUM_FILLER_GPUS", True)
    read_records_mock.return_value = _read_result(read_failed=True)
    ctx = context_factory(state=_state(current_limit=105, default_limit=350, default_job_owner="lium"))

    result = await GpuPowerLimitCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == "GPU_POWER_LIMIT_STATE_UNAVAILABLE"
    assert not result.updates
    restore_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_unreadable_records_keep_main_skip_event_while_the_flag_is_off(
    context_factory, read_records_mock, live_power_mock, monkeypatch
) -> None:
    """DAH-3630 regression: with the flag off the event stream stays main's. A Redis error under a live
    filler emits the skip, not GPU_POWER_LIMIT_STATE_UNAVAILABLE."""
    monkeypatch.setattr(settings, "ENABLE_POWER_FLOOR_FOR_UNCAPPED_LIUM_FILLER_GPUS", False)
    read_records_mock.return_value = _read_result(read_failed=True)
    ctx = context_factory(state=_state(current_limit=105, default_limit=350, default_job_owner="lium"))

    result = await GpuPowerLimitCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == "GPU_POWER_LIMIT_SKIPPED_LIUM_FILLER"
    assert not result.updates
    live_power_mock.assert_not_awaited()


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
