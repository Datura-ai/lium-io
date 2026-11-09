from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, Mock

import pytest
from helpers import FakeRedis, build_services, build_state
from protocol.vc_protocol.compute_requests import RentedExecutor, RentedExecutorsResponse, RentedPod
from services.gpu_power_limit import (
    STALE_CAP_GRACE_SECONDS,
    GpuPowerRestoreReadResult,
    GpuPowerRestoreRecord,
    rental_legacy_pod_key,
    rental_start_ratio_key,
    scored_tier_ratio_key,
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
async def test_power_limit_80_percent_tier_passes_on_an_unrented_node(context_factory, read_records_mock) -> None:
    ctx = context_factory(state=_state(current_limit=280, default_limit=350))  # exactly 80 %
    result = await GpuPowerLimitCheck().run(ctx)
    assert result.passed is True
    assert result.event.reason_code == "GPU_POWER_LIMIT_OK"


@pytest.mark.asyncio
async def test_a_passing_80_percent_tier_reading_is_kept_for_the_rental_start_and_cleared_at_stock(
    context_factory, read_records_mock
) -> None:
    # F-36: a GPU the rental-start read misses starts at the tier the renter's badge showed
    redis = FakeRedis()
    key = scored_tier_ratio_key(_EXECUTOR_UUID, "GPU-abc")

    def ctx(watts: int):
        return context_factory(state=_state(current_limit=watts, default_limit=350), services=build_services(redis=redis))

    await GpuPowerLimitCheck().run(ctx(280))
    assert redis.store[key] == "0.8"
    await GpuPowerLimitCheck(restore_stale_caps=False).run(ctx(350))  # the dry run writes nothing
    assert key in redis.store
    await GpuPowerLimitCheck().run(ctx(350))
    assert key not in redis.store


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kept", "read_fails", "writes_lost", "outcome"),
    [
        # a clear lost once is tried again and lands
        ("0.8", False, 1, "GPU_POWER_LIMIT_OK"),
        # F-43: 0.8 kept and both clears lost, or the read lost too: no verdict, the previous result stands
        ("0.8", False, 2, "GPU_POWER_TIER_STATE_UNKNOWN"),
        ("0.8", True, 2, "GPU_POWER_TIER_STATE_UNKNOWN"),
        # F-50: nothing kept, or the clear landed: our outage never fails a stock node
        (None, False, 2, "GPU_POWER_LIMIT_OK"),
        ("0.8", True, 0, "GPU_POWER_LIMIT_OK"),
    ],
)
async def test_a_stock_reading_never_passes_while_a_tier_reading_may_still_be_kept(
    context_factory, read_records_mock, kept, read_fails, writes_lost, outcome
) -> None:
    redis = FakeRedis()
    key = scored_tier_ratio_key(_EXECUTOR_UUID, "GPU-abc")
    if kept:
        redis.store[key] = kept
    write, calls = redis.write_atomically, []

    async def lossy(writes):
        calls.append(writes)
        if len(calls) <= writes_lost:
            raise ConnectionError("down")
        return await write(writes)

    redis.write_atomically = lossy
    if read_fails:
        redis.mget = AsyncMock(side_effect=ConnectionError("down"))
    ctx = context_factory(state=_state(current_limit=350, default_limit=350), services=build_services(redis=redis))
    result = await GpuPowerLimitCheck().run(ctx)
    assert result.event.reason_code == outcome
    assert result.passed is (outcome == "GPU_POWER_LIMIT_OK")
    assert "score" not in result.updates  # never a zero score from our Redis
    assert (key in redis.store) is (outcome != "GPU_POWER_LIMIT_OK" or (kept is not None and writes_lost >= 2))


def test_an_unknown_power_tier_state_is_held_like_a_gpu_reset() -> None:
    from services.executor_rollout import withhold_renter_gpu_reset_verdicts

    held = Mock(failure_reason_code="GPU_POWER_TIER_STATE_UNKNOWN", executor_info=Mock(uuid="e"), job_batch_id="b")
    standing, withheld = withhold_renter_gpu_reset_verdicts({"hk": [held]})
    assert standing == {"hk": []} and [w.result for w in withheld] == [held]


@pytest.mark.asyncio
async def test_power_limit_rejects_below_the_80_percent_tier(context_factory, read_records_mock) -> None:
    ctx = context_factory(state=_state(current_limit=276, default_limit=350))
    result = await GpuPowerLimitCheck().run(ctx)
    assert result.passed is False
    assert result.event.reason_code == "GPU_POWER_LIMIT_BELOW_DEFAULT"
    assert result.updates["score"] == 0.0


def _rented_ctx(
    context_factory,
    current_limit: float,
    records: dict[str, dict],
    count: int = 1,
    redis: FakeRedis | None = None,
    gpu_count: int | None = None,
    container_name: str = "pod_pod-1",
):
    state = _state(current_limit=current_limit, default_limit=350, count=count)
    state.rented_data.executors[_EXECUTOR_UUID] = RentedExecutor(
        miner_hotkey="hk",
        executor_ip_address="1.2.3.4",
        executor_ip_port="22",
        pods=[RentedPod(pod_id="pod-1", container_name=container_name, gpu_count=gpu_count)],
    )
    redis = redis or FakeRedis()
    redis.store.update(
        {rental_start_ratio_key(_EXECUTOR_UUID, u): json.dumps(r) for u, r in records.items()}
    )
    return context_factory(state=state, services=build_services(redis=redis))


_AT_STOCK = {"pod_id": "pod-1", "ratio": 1.0}
_IN_TIER = {"pod_id": "pod-1", "ratio": 0.8514}


@pytest.mark.parametrize(
    ("current_limit", "records", "passed"),
    [
        # rented at stock power, cut to 85 % mid-rental: the renter never saw a badge
        (298, {"GPU-abc": _AT_STOCK}, False),
        (315, {"GPU-abc": _AT_STOCK}, True),
        # rented in the 80 % tier: keeps its own cap, never lower
        (280, {"GPU-abc": {"pod_id": "pod-1", "ratio": 0.8}}, True),
        (298, {"GPU-abc": _IN_TIER}, True),
        (285, {"GPU-abc": _IN_TIER}, False),
        # a rental from before the records, or a record of a pod no longer rented: 90 %
        (298, {}, False),
        (298, {"GPU-abc": {"pod_id": "old-pod", "ratio": 0.8}}, False),
    ],
)
@pytest.mark.asyncio
async def test_power_limit_rented_gpu_keeps_its_rental_start_cap(
    context_factory, read_records_mock, current_limit, records, passed
) -> None:
    result = await GpuPowerLimitCheck().run(_rented_ctx(context_factory, current_limit, records))
    assert result.passed is passed


@pytest.mark.parametrize(
    ("records", "gpu_count", "passed"),
    [
        # pod-1 holds GPU-abc only (recorded): GPU-abc-1 is the free bundle and stays in the tier
        ({"GPU-abc": _IN_TIER}, 1, True),
        # pod-1 holds both GPUs but only GPU-abc's record survived: GPU-abc-1 is held at 90 %
        ({"GPU-abc": _IN_TIER}, 2, False),
        ({"GPU-abc": _IN_TIER, "GPU-abc-1": _IN_TIER}, 2, True),
        # the backend predates gpu_count: which GPUs are rented is unknown, so 90 %
        ({"GPU-abc": _IN_TIER}, None, False),
    ],
)
@pytest.mark.asyncio
async def test_power_limit_split_node_free_gpu_floor_needs_every_rented_gpu_recorded(
    context_factory, read_records_mock, records, gpu_count, passed
) -> None:
    ctx = _rented_ctx(context_factory, 298, records, count=2, gpu_count=gpu_count)
    assert (await GpuPowerLimitCheck().run(ctx)).passed is passed


@pytest.mark.asyncio
async def test_power_limit_partial_record_holds_the_unrecorded_rented_gpu(context_factory, read_records_mock) -> None:
    # a 2-GPU rental at stock power whose GPU-abc-1 record was lost: GPU-abc-1 cut to 85 % fails
    ctx = _rented_ctx(context_factory, 298, {"GPU-abc": _AT_STOCK}, count=2, gpu_count=2)
    ctx.state.gpu_details[0]["power_limit"] = 350
    assert (await GpuPowerLimitCheck().run(ctx)).passed is False


@pytest.mark.asyncio
async def test_power_limit_pod_still_being_created_is_not_judged(context_factory, read_records_mock) -> None:
    # rented in the tier; the backend lists the pod before its record is written
    redis = FakeRedis()
    ctx = _rented_ctx(context_factory, 298, {}, gpu_count=1, container_name="", redis=redis)
    assert (await GpuPowerLimitCheck().run(ctx)).passed is True
    assert rental_legacy_pod_key("pod-1") not in redis.store


@pytest.mark.asyncio
async def test_power_limit_marks_a_rental_from_before_the_records(context_factory, read_records_mock) -> None:
    redis = FakeRedis()
    ctx = _rented_ctx(context_factory, 350, {}, gpu_count=1, redis=redis)
    assert (await GpuPowerLimitCheck().run(ctx)).passed is True
    assert rental_legacy_pod_key("pod-1") in redis.store
    dry = _rented_ctx(context_factory, 350, {}, gpu_count=1, redis=FakeRedis())
    await GpuPowerLimitCheck(restore_stale_caps=False).run(dry)
    assert rental_legacy_pod_key("pod-1") not in dry.services.redis.store


@pytest.mark.asyncio
async def test_power_limit_rented_node_redis_outage_falls_back_to_the_80_percent_floor(
    context_factory, read_records_mock
) -> None:
    ctx = _rented_ctx(context_factory, 298, {}, redis=FakeRedis(failing=True))
    assert (await GpuPowerLimitCheck().run(ctx)).passed is True


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


@pytest.mark.asyncio
async def test_an_unreadable_power_limit_fails(context_factory) -> None:
    ctx = context_factory(state=_state(current_limit=None, default_limit=350))

    result = await GpuPowerLimitCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == "GPU_POWER_LIMIT_UNREADABLE"
    assert result.updates["score"] == 0.0


@pytest.mark.asyncio
async def test_a_gb10_without_a_power_limit_passes_and_other_models_still_fail(context_factory) -> None:
    gb10 = _state(current_limit=None, default_limit=None)
    gb10.gpu_details[0]["name"] = "NVIDIA GB10"
    mixed = _state(current_limit=None, default_limit=None, count=2)
    mixed.gpu_details[0]["name"] = "NVIDIA GB10"

    gb10_result = await GpuPowerLimitCheck().run(context_factory(state=gb10))
    mixed_result = await GpuPowerLimitCheck().run(context_factory(state=mixed))

    assert gb10_result.passed is True
    assert gb10_result.event.reason_code == "GPU_POWER_LIMIT_OK"
    assert mixed_result.passed is False
    assert mixed_result.event.reason_code == "GPU_POWER_LIMIT_UNREADABLE"


@pytest.mark.asyncio
async def test_an_unreadable_power_limit_under_a_lium_filler_passes(context_factory) -> None:
    ctx = context_factory(
        state=_state(current_limit=None, default_limit=350, default_job_owner="lium")
    )

    result = await GpuPowerLimitCheck().run(ctx)

    assert result.passed is True


@pytest.mark.parametrize(
    ("owner", "recorded_watts", "expected_host"),
    [
        ("lium", 350, 350),  # Pearl over a stock host limit
        ("lium", 280, 280),  # Pearl over an 80 % tier host limit
        ("lium", None, None),  # no restore record: no field, the backend keeps today's verdict
        (None, 280, None),  # no filler: the live limit is the host's, no field
    ],
)
@pytest.mark.asyncio
async def test_the_host_limit_under_our_pearl_cap_reaches_the_specs(
    context_factory, read_records_mock, owner, recorded_watts, expected_host
) -> None:
    # lio#1924 r4477687011: the badge judges the limit the renter gets back, not Pearl's live cap
    live = 245 if owner else 280
    state = _state(current_limit=live, default_limit=350, default_job_owner=owner)
    state = ContextState(**{**state.__dict__, "specs": {"gpu": {"details": [dict(d) for d in state.gpu_details]}}})
    if recorded_watts is not None:
        read_records_mock.return_value = _read_result(
            _restore_record().model_copy(update={"watts": recorded_watts}),
            _restore_record(gpu_uuid="GPU-abc", executor_id="other-executor"),
        )
    result = await GpuPowerLimitCheck().run(context_factory(state=state))
    assert result.passed is True
    specs = result.updates["state"].specs if "state" in (result.updates or {}) else state.specs
    (detail,) = specs["gpu"]["details"]
    assert detail.get("power_host_limit") == expected_host
    assert detail["power_limit"] == live  # Pearl's guard still reads the live cap


@pytest.mark.asyncio
async def test_a_low_power_limit_names_each_gpu_measured_and_required(context_factory, read_records_mock) -> None:
    ctx = context_factory(state=_state(current_limit=105, default_limit=350))

    result = await GpuPowerLimitCheck().run(ctx)

    # the requirement is the message's own wording, never a second copy of the threshold
    assert result.event.remediation.startswith(
        "GPU 0: 105 W limit, default 350 W (30%). Restore the GPU power limit to at least 80% of the default"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("current_limit", [279.9, 279.99, 279.9999999])
async def test_a_power_limit_just_under_the_floor_never_shows_as_meeting_it(
    context_factory, read_records_mock, current_limit
) -> None:
    ctx = context_factory(state=_state(current_limit=current_limit, default_limit=350))

    result = await GpuPowerLimitCheck().run(ctx)

    # neither the watts nor the ratio round up onto the 280 W / 80% floor the GPU failed
    assert result.event.remediation.startswith(
        "GPU 0: 279 W limit, default 350 W (79%). Restore the GPU power limit to at least 80% of the default"
    )
