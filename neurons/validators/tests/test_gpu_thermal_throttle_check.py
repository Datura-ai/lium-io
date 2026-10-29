"""GpuThermalThrottleCheck and the scrape readings it needs (support ticket-0492).

Two rented H100s ran at 345-650 MHz under 92-100% load for two days (healthy ~1980 MHz) with SW thermal
slowdown at 86-87 C. The scrape kept only the clock, and no check looked at temperature or the clocks-event
reasons. The check is non-fatal and observe-only: a warning event, the score unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from neurons.validators.tests.helpers import build_scrape_namespace, build_state
from protocol.vc_protocol.compute_requests import RentedExecutor, RentedExecutorsResponse
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


async def run(context_factory, *rows: dict[str, Any], rented: bool = False):
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
    ctx = context_factory(state=build_state(gpu_details=list(rows), rented_data=rented_data))
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
