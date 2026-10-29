"""GpuThermalThrottleCheck and the scrape readings it needs (support ticket-0492).

Two rented H100s ran at 345-650 MHz under 92-100% load for two days (healthy ~1980 MHz) with SW thermal
slowdown at 86-87 C. The scrape kept only the clock, and no check looked at temperature or the clocks-event
reasons. The check is non-fatal and observe-only: a warning event, the score unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import json

import pytest
from neurons.validators.tests.helpers import (
    FakeRedis,
    build_scrape_namespace,
    build_services,
    build_state,
)
from protocol.vc_protocol.compute_requests import RentedExecutor, RentedExecutorsResponse
from services.task.availability import AvailabilityErrorCode, availability_errors
from services.task.checks import gpu_thermal_throttle
from services.task.checks.gpu_thermal_throttle import GpuThermalThrottleCheck
from services.task.checks.rented_gpu_drop import RentedGpuDropCheck
from services.task.messages import GpuThermalThrottleMessages as Msg
from services.task.pipeline_factory import PipelineFactory

SRC = Path(__file__).resolve().parents[1] / "src"

SW_THERMAL = 0x20
HW_THERMAL = 0x40
HW_SLOWDOWN = 0x8
SW_POWER_CAP = 0x4
GPU_IDLE = 0x1


def h100(**overrides: Any) -> dict[str, Any]:
    """One gpu_details row as MachineSpecScrapeCheck leaves it, healthy and busy."""
    row: dict[str, Any] = {
        "name": "NVIDIA H100 80GB HBM3",
        "uuid": "GPU-aaaa",
        "gpu_utilization": 98,
        "graphics_speed": 1980,
        "max_graphics_speed": 1980,
        "temperature": 62,
        "clocks_event_reasons": 0,
    }
    row.update(overrides)
    return row


async def run(context_factory, *rows: dict[str, Any], rented: bool = False, redis=None):
    rented_data = None
    if rented:
        pod = {"pod_id": "pod-1", "container_name": "c", "gpu_count": 1}
        executor = RentedExecutor.model_validate(
            {
                "miner_hotkey": "hk",
                "executor_ip_address": "1.2.3.4",
                "executor_ip_port": "8080",
                "pods": [pod],
            }
        )
        rented_data = RentedExecutorsResponse(executors={context_factory().executor.uuid: executor})
    ctx = context_factory(
        state=build_state(gpu_details=list(rows), rented_data=rented_data),
        services=build_services(redis=redis or FakeRedis()),
    )
    return await GpuThermalThrottleCheck().run(ctx)


# --------------------------------------------------------------------------------------------------
# flagged
# --------------------------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_the_ticket_reading_is_flagged(context_factory):
    # Arrange: the incident as the renter's nvidia-smi showed it
    row = h100(
        graphics_speed=345, temperature=87, gpu_utilization=100, clocks_event_reasons=SW_THERMAL
    )

    # Act
    result = await run(context_factory, row)

    # Assert: an unrented node fails with score 0 until the cooling is fixed
    assert result.passed is False
    assert result.event.reason_code == Msg.COLLAPSED.reason
    assert result.updates["score"] == 0.0
    gpu = result.event.what_we_saw["gpus"][0]
    assert gpu["uuid"] == "GPU-aaaa"
    assert gpu["temperature_c"] == 87
    assert gpu["graphics_clock_mhz"] == 345
    assert gpu["clocks_event_reasons"] == "0x20"
    assert gpu["reasons"] == ["sw_thermal_slowdown", "clock_below_half_of_max"]
    assert gpu["collapsed"] is True


@pytest.mark.asyncio
async def test_a_collapsed_gpu_on_a_rented_node_only_warns(context_factory):
    row = h100(graphics_speed=345, temperature=95, clocks_event_reasons=SW_THERMAL | HW_THERMAL)

    result = await run(context_factory, row, rented=True)

    assert result.passed is True
    assert result.event.reason_code == Msg.THROTTLED.reason


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("bits", "name"),
    [
        (SW_THERMAL, "sw_thermal_slowdown"),
        (HW_THERMAL, "hw_thermal_slowdown"),
        (HW_SLOWDOWN, "hw_slowdown"),
    ],
)
async def test_a_thermal_or_slowdown_bit_alone_is_flagged(context_factory, bits, name):
    # a mild throttle the matmul tolerance lets through: clock still above half of max
    result = await run(context_factory, h100(graphics_speed=1700, clocks_event_reasons=bits))

    assert result.event.reason_code == Msg.THROTTLED.reason
    assert result.event.what_we_saw["gpus"][0]["reasons"] == [name]


@pytest.mark.asyncio
async def test_a_collapsed_clock_is_flagged_when_the_driver_reports_no_reasons(context_factory):
    result = await run(context_factory, h100(graphics_speed=650, clocks_event_reasons=None))

    assert result.event.reason_code == Msg.THROTTLED.reason
    assert result.event.what_we_saw["gpus"][0]["reasons"] == ["clock_below_half_of_max"]


@pytest.mark.asyncio
async def test_only_the_throttled_card_of_a_multi_gpu_node_is_listed(context_factory):
    result = await run(
        context_factory,
        h100(uuid="GPU-ok"),
        h100(uuid="GPU-hot", graphics_speed=500, clocks_event_reasons=SW_THERMAL | HW_SLOWDOWN),
    )

    assert result.event.reason_code == Msg.COLLAPSED.reason
    assert result.event.what_we_saw["collapsed_count"] == 1
    assert [gpu["uuid"] for gpu in result.event.what_we_saw["gpus"]] == ["GPU-hot"]


# --------------------------------------------------------------------------------------------------
# pass
# --------------------------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_a_healthy_busy_gpu_passes(context_factory):
    result = await run(context_factory, h100())

    assert result.passed is True
    assert result.event.reason_code == Msg.OK.reason


@pytest.mark.asyncio
async def test_an_idle_gpu_is_not_judged(context_factory):
    # an idle card drops its clock on purpose
    result = await run(
        context_factory,
        h100(gpu_utilization=0, graphics_speed=345, clocks_event_reasons=GPU_IDLE | SW_THERMAL),
    )

    assert result.event.reason_code == Msg.OK.reason


@pytest.mark.asyncio
async def test_a_low_clock_explained_by_a_power_cap_is_left_to_the_power_limit_check(
    context_factory,
):
    result = await run(context_factory, h100(graphics_speed=800, clocks_event_reasons=SW_POWER_CAP))

    assert result.event.reason_code == Msg.OK.reason


@pytest.mark.asyncio
async def test_an_old_scrape_without_the_new_fields_passes(context_factory):
    # the rows a scrape from before this change leaves: no temperature, max clock or reasons
    row = h100(graphics_speed=345)
    for key in ("temperature", "max_graphics_speed", "clocks_event_reasons"):
        del row[key]

    result = await run(context_factory, row)

    assert result.passed is True
    assert result.event.reason_code == Msg.OK.reason


@pytest.mark.asyncio
async def test_no_gpu_details_passes(context_factory):
    result = await run(context_factory)

    assert result.event.reason_code == Msg.OK.reason


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "row",
    [
        h100(
            graphics_speed=1965, temperature=74, clocks_event_reasons=SW_THERMAL
        ),  # a thermal bit at full clock
        h100(
            graphics_speed=650, clocks_event_reasons=None
        ),  # a low clock the driver gives no reason for
    ],
)
async def test_only_a_collapse_with_a_slowdown_bit_fails(context_factory, row):
    result = await run(context_factory, row)

    assert result.passed is True
    assert result.event.reason_code == Msg.THROTTLED.reason


# --------------------------------------------------------------------------------------------------
# hold: a collapse is remembered until a busy clean reading (28 Oct, 8x H200 c63f453b)
# --------------------------------------------------------------------------------------------------
COLLAPSED_ROW = h100(graphics_speed=345, temperature=88, clocks_event_reasons=SW_THERMAL)
IDLE_ROW = h100(gpu_utilization=0, graphics_speed=345, clocks_event_reasons=GPU_IDLE)


def _hold_key(context_factory) -> str:
    ctx = context_factory()
    return f"{gpu_thermal_throttle.THERMAL_HOLD_KEY_PREFIX}:{ctx.miner_hotkey}:{ctx.executor.uuid}"


@pytest.mark.asyncio
async def test_a_rented_collapse_is_marked_and_only_warns(context_factory):
    redis = FakeRedis()

    result = await run(context_factory, COLLAPSED_ROW, rented=True, redis=redis)

    assert result.passed is True
    assert result.event.reason_code == Msg.THROTTLED.reason
    key = _hold_key(context_factory)
    assert json.loads(redis.store[key])["gpus"] == ["GPU-aaaa"]
    assert redis.ttl[key] == gpu_thermal_throttle.THERMAL_HOLD_SECONDS


@pytest.mark.asyncio
@pytest.mark.parametrize("first_rented", [True, False])
async def test_an_idle_cycle_after_a_collapse_is_held(context_factory, first_rented):
    redis = FakeRedis()
    await run(context_factory, COLLAPSED_ROW, rented=first_rented, redis=redis)

    result = await run(context_factory, IDLE_ROW, redis=redis)

    assert result.passed is False
    assert result.event.reason_code == Msg.COLLAPSED.reason
    assert result.updates["score"] == 0.0
    assert result.event.what_we_saw["held_from_rental"] is True
    assert result.event.what_we_saw["held_gpu_uuids"] == ["GPU-aaaa"]
    assert result.event.what_we_saw["held_since"]


@pytest.mark.asyncio
async def test_a_held_node_under_rental_still_only_warns(context_factory):
    redis = FakeRedis()
    await run(context_factory, COLLAPSED_ROW, redis=redis)

    result = await run(context_factory, IDLE_ROW, rented=True, redis=redis)

    assert result.passed is True
    assert _hold_key(context_factory) in redis.store


@pytest.mark.asyncio
async def test_a_busy_clean_reading_clears_the_hold(context_factory):
    redis = FakeRedis()
    await run(context_factory, COLLAPSED_ROW, rented=True, redis=redis)

    cleared = await run(context_factory, h100(), redis=redis)
    idle_after = await run(context_factory, IDLE_ROW, redis=redis)

    assert cleared.passed is True
    assert cleared.event.reason_code == Msg.OK.reason
    assert redis.store == {}
    assert idle_after.passed is True


@pytest.mark.asyncio
async def test_only_the_marked_gpu_that_read_clean_is_cleared(context_factory):
    redis = FakeRedis()
    await run(
        context_factory,
        {**COLLAPSED_ROW, "uuid": "GPU-a"},
        {**COLLAPSED_ROW, "uuid": "GPU-b"},
        rented=True,
        redis=redis,
    )

    result = await run(
        context_factory, h100(uuid="GPU-a"), h100(uuid="GPU-b", gpu_utilization=0), redis=redis
    )

    assert result.passed is False
    assert result.event.what_we_saw["held_gpu_uuids"] == ["GPU-b"]


@pytest.mark.asyncio
async def test_a_replaced_marked_gpu_is_cleared(context_factory):
    redis = FakeRedis()
    await run(context_factory, {**COLLAPSED_ROW, "uuid": "GPU-a"}, rented=True, redis=redis)

    result = await run(context_factory, h100(uuid="GPU-new", gpu_utilization=0), redis=redis)

    assert result.passed is True
    assert await redis.get(_hold_key(context_factory)) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ["graphics_speed", "max_graphics_speed", "clocks_event_reasons"])
async def test_a_busy_reading_missing_a_field_keeps_the_hold(context_factory, missing):
    # review: a reading judge_gpu cannot judge is not a clean one
    redis = FakeRedis()
    await run(context_factory, COLLAPSED_ROW, rented=True, redis=redis)
    row = h100()
    del row[missing]

    result = await run(context_factory, row, redis=redis)

    assert result.passed is False
    assert result.event.what_we_saw["held_gpu_uuids"] == ["GPU-aaaa"]


@pytest.mark.asyncio
async def test_a_row_without_a_uuid_drops_no_mark_as_replaced(context_factory):
    # review: one row with no uuid does not make the whole marked set look replaced
    redis = FakeRedis()
    await run(
        context_factory,
        {**COLLAPSED_ROW, "uuid": "GPU-a"},
        {**COLLAPSED_ROW, "uuid": "GPU-b"},
        rented=True,
        redis=redis,
    )
    no_uuid = h100(gpu_utilization=0)
    del no_uuid["uuid"]

    result = await run(context_factory, no_uuid, h100(uuid="GPU-a", gpu_utilization=0), redis=redis)

    assert result.passed is False
    assert result.event.what_we_saw["held_gpu_uuids"] == ["GPU-a", "GPU-b"]


@pytest.mark.asyncio
@pytest.mark.parametrize("held", [False, True])
async def test_a_collapse_fail_hides_the_node_at_once(context_factory, held):
    # review: an availability entry as GpuLossHoldCheck's, on a fresh collapse and on a held cycle
    redis = FakeRedis()
    if held:
        await run(context_factory, COLLAPSED_ROW, rented=True, redis=redis)

    result = await run(context_factory, IDLE_ROW if held else COLLAPSED_ROW, redis=redis)

    assert result.passed is False
    [entry] = availability_errors([result.event])
    assert entry.reason_code == AvailabilityErrorCode.GPU_THERMAL_COLLAPSE
    assert (entry.reach_source, entry.reach_target) == ("validator", "gpu")
    assert entry.what_we_saw.get("held_from_rental", False) is held


@pytest.mark.asyncio
async def test_a_rented_collapse_warning_is_no_availability_entry(context_factory):
    result = await run(context_factory, COLLAPSED_ROW, rented=True)

    assert availability_errors([result.event]) == []


@pytest.mark.asyncio
async def test_a_redis_outage_is_no_hold(context_factory):
    # read fails: no hold; write fails: the collapse verdict stands; neither raises
    redis = FakeRedis(failing=True)

    idle = await run(context_factory, IDLE_ROW, redis=redis)
    collapsed = await run(context_factory, COLLAPSED_ROW, redis=redis)
    rented = await run(context_factory, COLLAPSED_ROW, rented=True, redis=redis)

    assert idle.passed is True
    assert idle.event.reason_code == Msg.OK.reason
    assert collapsed.event.reason_code == Msg.COLLAPSED.reason
    assert rented.event.reason_code == Msg.THROTTLED.reason


@pytest.mark.asyncio
async def test_a_dry_run_writes_no_hold(context_factory, monkeypatch):
    monkeypatch.setattr(gpu_thermal_throttle.settings, "DRY_RUN", True)
    redis = FakeRedis()

    await run(context_factory, COLLAPSED_ROW, rented=True, redis=redis)

    assert redis.store == {}


# --------------------------------------------------------------------------------------------------
# pipelines
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "build",
    [
        PipelineFactory.build_checks,
        PipelineFactory.build_fast_path_checks,
        PipelineFactory.build_dry_run_checks,
    ],
)
def test_the_check_runs_right_after_the_rented_gpu_drop_check(build):
    # before the rented short-circuit, so rented nodes are judged too (the GPU loss hold sits between them)
    ids = [check.check_id for check in build()]

    assert ids.index(GpuThermalThrottleCheck.check_id) == ids.index(RentedGpuDropCheck.check_id) + 2


# --------------------------------------------------------------------------------------------------
# scrape: the clocks-event reasons fall back to the pre-535 name
# --------------------------------------------------------------------------------------------------
class _FakeNvmlError(Exception):
    pass


def test_clocks_event_reasons_fall_back_to_the_throttle_reasons_call():
    calls: list[str] = []

    def fake_pointer(name):
        calls.append(name)
        if name == "nvmlDeviceGetCurrentClocksEventReasons":
            raise _FakeNvmlError(name)

        def fn(handle, reasons_ref):
            reasons_ref._obj.value = SW_THERMAL
            return 0

        return fn

    import ctypes

    scrape = build_scrape_namespace(
        SRC / "miner_jobs" / "machine_scrape.py",
        {"nvmlDeviceGetClocksEventReasons"},
        {
            "c_ulonglong": ctypes.c_ulonglong,
            "byref": ctypes.byref,
            "NVMLError": _FakeNvmlError,
            "_nvmlGetFunctionPointer": fake_pointer,
            "_nvmlCheckReturn": lambda ret: None,
        },
    )

    assert scrape["nvmlDeviceGetClocksEventReasons"](object()) == SW_THERMAL
    assert calls == [
        "nvmlDeviceGetCurrentClocksEventReasons",
        "nvmlDeviceGetCurrentClocksThrottleReasons",
    ]
