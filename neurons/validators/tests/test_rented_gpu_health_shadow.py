"""RENTED_GPU_HEALTH shadow: a rented node's per-card health is read and logged, never scored."""

import json
import logging
import re
import sys
import types
from unittest.mock import patch

import pytest
from neurons.validators.src.services.task.checks import (
    RentedGpuDropCheck,
    RentedGpuHealthShadowCheck,
    TenantEnforcementCheck,
)
from neurons.validators.src.services.task.checks import rented_gpu_health as module
from neurons.validators.src.services.task.checks.gpu_fault_probe import PROBE_JSON_MARKER, PROBE_SOURCE
from neurons.validators.src.services.task.checks.rented_gpu_health import card_flags, judge_card, judge_health
from neurons.validators.src.services.task.messages import RentedGpuHealthMessages as Msg
from neurons.validators.src.services.task.pipeline_factory import PipelineFactory
from protocol.vc_protocol.compute_requests import RentedExecutor, RentedExecutorsResponse, RentedPod

from tests.helpers import build_context_config, build_services, build_state, default_executor
from tests.test_gpu_fault_probe_check import FakeRunner, _probe_namespace

UUIDS = [f"GPU-a0a{i}-0000-0000-0000-00000000000{i}" for i in range(8)]


def _healthy(index: int) -> dict:
    return {
        "index": index,
        "uuid": UUIDS[index],
        "pci_bus_id": f"00000000:{index + 1:02x}:00.0",
        "ecc_uncorrected": 0,
        "remapped_rows": [0, 0, 0, 0],
        "recovery_action": 0,
        "retired_pages_pending": False,
        "hardware_xids": [],
    }


def _stdout(gpus: list[dict], *, count: int | None = None, available: bool = True) -> str:
    report = {
        "status": "health",
        "nvml": {"available": available, "count": len(gpus) if count is None else count, "gpus": gpus},
        "xid": {"available": False, "hardware": []},
    }
    return PROBE_JSON_MARKER + " " + json.dumps(report) + "\n"


def _rented(executor_uuid: str, gpu_count: int = 8, owner: str = "miner-hotkey") -> RentedExecutorsResponse:
    pod = RentedPod(pod_id="p1", container_name="container_p1", gpu_count=gpu_count, status="RUNNING")
    return RentedExecutorsResponse(
        executors={
            executor_uuid: RentedExecutor(
                miner_hotkey=owner, executor_ip_address="127.0.0.1", executor_ip_port="8001", pods=[pod]
            )
        }
    )


def _ctx(context_factory, runner, *, rented=True, owner="miner-hotkey"):
    executor = default_executor()
    state = build_state(
        specs={"gpu": {"count": 8}},
        rented_data=_rented(executor.uuid, owner=owner) if rented else RentedExecutorsResponse(executors={}),
    )
    return context_factory(
        services=build_services(),
        config=build_context_config(),
        state=state,
        executor=executor,
        runner=runner,
        verified={"uuids": ",".join(UUIDS)},
    )


@pytest.fixture
def enabled():
    with patch.object(module.settings, "RENTED_GPU_HEALTH_SHADOW_ENABLED", True):
        yield


@pytest.mark.asyncio
async def test_switched_off_runs_nothing(context_factory, monkeypatch):
    monkeypatch.setenv("RENTED_GPU_HEALTH_SHADOW_ENABLED", "false")
    runner = FakeRunner()
    with patch.object(module, "settings", type(module.settings)(_env_file=None)):
        result = await RentedGpuHealthShadowCheck().run(_ctx(context_factory, runner))

    assert result.passed and result.event.reason_code == Msg.DISABLED.reason
    assert runner.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("kwargs", [{"rented": False}, {"owner": "another-hotkey"}])
async def test_an_unrented_node_is_skipped(context_factory, enabled, kwargs):
    runner = FakeRunner()
    result = await RentedGpuHealthShadowCheck().run(_ctx(context_factory, runner, **kwargs))

    assert result.passed and result.event.reason_code == Msg.NOT_RENTED.reason
    assert runner.calls == []


@pytest.mark.asyncio
async def test_the_health_read_runs_the_probe_in_health_mode_over_stdin(context_factory, enabled):
    runner = FakeRunner(_stdout([_healthy(i) for i in range(8)]))

    result = await RentedGpuHealthShadowCheck().run(_ctx(context_factory, runner))

    (call,) = runner.calls
    assert call["cmd"] == f"{default_executor().python_path} -I - --health"
    assert call["stdin_text"] == PROBE_SOURCE and call["retryable"] is False
    assert result.passed and result.event.reason_code == Msg.SHADOW_OK.reason
    assert result.event.what_we_saw["verdict"] == "ok"


@pytest.mark.asyncio
async def test_a_lost_card_and_a_pending_retirement_are_logged_but_never_fail(context_factory, enabled, caplog):
    gpus = [_healthy(i) for i in range(7)]
    gpus[2]["retired_pages_pending"] = True
    gpus[3] = {"index": 3, "error": "GPU is lost"}
    runner = FakeRunner(_stdout(gpus, count=7))

    with caplog.at_level(logging.WARNING, logger=module.__name__):
        result = await RentedGpuHealthShadowCheck().run(_ctx(context_factory, runner))

    assert result.passed is True
    assert RentedGpuHealthShadowCheck.fatal is False
    assert "score" not in result.updates
    assert result.event.reason_code == Msg.SHADOW_FAULT.reason
    seen = result.event.what_we_saw
    assert seen["verdict"] == "fault" and seen["shadow"] is True
    assert seen["expected_gpu_count"] == 8 and seen["nvml_gpu_count"] == 7
    verdicts = {card["index"]: card for card in seen["cards"]}
    assert verdicts[2]["reasons"] == ["retired_pages_pending"]
    assert verdicts[3]["reasons"] == ["nvml_error"] and verdicts[3]["error"] == "GPU is lost"
    assert verdicts[None]["verdict"] == "missing"
    assert seen["faulty_card_count"] == 3
    assert any("rented_gpu_health_shadow" in record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
async def test_no_report_is_unknown_not_a_fault(context_factory, enabled):
    runner = FakeRunner("python: not found", exit_code=127)

    result = await RentedGpuHealthShadowCheck().run(_ctx(context_factory, runner))

    assert result.passed and result.event.reason_code == Msg.SHADOW_UNKNOWN.reason
    assert result.event.what_we_saw["verdict"] == "unknown"


def test_each_fault_counter_names_its_reason():
    card = _healthy(0) | {
        "ecc_uncorrected": 2,
        "remapped_rows": [1, 1, 1, 1],
        "recovery_action": 2,
        "hardware_xids": [79],
        "read_errors": ["uuid: Unknown Error"],
    }

    assert judge_card(_healthy(0)) == []
    assert judge_card(card) == [
        "read_error",
        "ecc_uncorrected",
        "remapped_rows_pending",
        "remapped_rows_failed",
        "recovery_action",
        "xid_79",
    ]


def test_card_flags_name_a_reset_a_hung_card_and_a_narrowed_link():
    # one card broke under a rental: NVML asks for a reset, the clocks sit at max with nothing running
    # and the link trained down; GeForce cards have no ECC counters to show it
    broken = _healthy(0) | {
        "recovery_action": 1,
        "gfx_clock_mhz": 2820,
        "gfx_clock_max_mhz": 3105,
        "util_pct": 0,
        "pcie_width": 8,
        "pcie_width_max": 16,
    }

    assert card_flags(broken) == ["recovery_action", "clocks_high_at_idle", "pcie_width_drop"]


@pytest.mark.parametrize(
    "signals",
    [
        {"gfx_clock_mhz": 2820, "gfx_clock_max_mhz": 3105, "util_pct": 97},  # busy at high clocks
        {"gfx_clock_mhz": 210, "gfx_clock_max_mhz": 3105, "util_pct": 0},  # idle and clocked down
        {"gfx_clock_mhz": 2820, "gfx_clock_max_mhz": None, "util_pct": 0},  # max clock unknown
        {"pcie_width": 16, "pcie_width_max": 16},  # an idle Gen1 link at full width is normal
        {"pcie_width": None, "pcie_width_max": 16},
        {"recovery_action": 0},
    ],
)
def test_normal_or_unknown_signals_raise_no_flag(signals):
    assert card_flags(_healthy(0) | signals) == []


@pytest.mark.asyncio
async def test_flags_are_logged_per_card_and_never_fail(context_factory, enabled, caplog):
    gpus = [_healthy(i) for i in range(8)]
    gpus[1] |= {"gfx_clock_mhz": 2820, "gfx_clock_max_mhz": 3105, "util_pct": 0, "pcie_width": 16}
    runner = FakeRunner(_stdout(gpus))

    with caplog.at_level(logging.WARNING, logger=module.__name__):
        result = await RentedGpuHealthShadowCheck().run(_ctx(context_factory, runner))

    assert result.passed is True and "score" not in result.updates
    seen = result.event.what_we_saw
    # a flag alone does not make the card a fault: an idle rented card at high clocks can be legitimate
    assert seen["verdict"] == "ok" and seen["flagged_card_count"] == 1
    card = seen["cards"][1]
    assert card["flags"] == ["clocks_high_at_idle"] and card["gfx_clock_mhz"] == 2820
    assert seen["cards"][0]["flags"] == []
    (record,) = [r for r in caplog.records if "rented_gpu_health_shadow" in r.getMessage()]
    assert record.levelno == logging.WARNING
    assert "clocks_high_at_idle" in record.msg.to_full_string()


def test_extra_cards_beyond_the_rental_are_not_missing():
    verdict = judge_health({"available": True, "count": 8, "gpus": [_healthy(i) for i in range(8)]}, 4)

    assert verdict["verdict"] == "ok" and verdict["faulty_card_count"] == 0


@pytest.mark.parametrize(
    "build",
    [PipelineFactory.build_checks, PipelineFactory.build_dry_run_checks, PipelineFactory.build_fast_path_checks],
)
def test_the_check_runs_beside_the_drop_check_before_the_rented_halt(build):
    kinds = [type(check) for check in build()]

    assert kinds.index(RentedGpuDropCheck) < kinds.index(RentedGpuHealthShadowCheck) < kinds.index(TenantEnforcementCheck)


# --- the probe's --health mode -----------------------------------------------------------------------------------


class _NvmlError(Exception):
    def __init__(self, value: int, text: str):
        super().__init__(text)
        self.value = value


def _fake_pynvml(lost_index: int, pending_index: int) -> types.ModuleType:
    nvml = types.ModuleType("pynvml")
    nvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED = 1
    nvml.NVML_VOLATILE_ECC = 0
    nvml.nvmlInit = lambda: None
    nvml.nvmlShutdown = lambda: None
    nvml.nvmlDeviceGetCount = lambda: 3

    def handle(i):
        if i == lost_index:
            raise _NvmlError(15, "GPU is lost")
        return i

    def not_supported(h):
        raise _NvmlError(3, "Not Supported")

    nvml.nvmlDeviceGetHandleByIndex = handle
    nvml.nvmlDeviceGetUUID = lambda h: f"GPU-{h}".encode()
    nvml.nvmlDeviceGetPciInfo = lambda h: types.SimpleNamespace(busId=f"00000000:0{h + 1}:00.0")
    nvml.nvmlDeviceGetTotalEccErrors = lambda h, kind, counter: 0
    nvml.nvmlDeviceGetRemappedRows = lambda h: (0, 0, 0, 0)
    nvml.nvmlDeviceGetGpuRecoveryAction = not_supported
    nvml.nvmlDeviceGetRetiredPagesPendingStatus = lambda h: 1 if h == pending_index else 0
    nvml.asked_fields = []

    def field_values(h, ids):
        nvml.asked_fields.append(ids)
        # card 0: an old driver without field 230; card 2: "Reset"
        code = 3 if h == 0 else 0
        return [types.SimpleNamespace(nvmlReturn=code, value=types.SimpleNamespace(uiVal=1))]

    nvml.nvmlDeviceGetFieldValues = field_values
    nvml.NVML_CLOCK_GRAPHICS = 0
    nvml.nvmlDeviceGetClockInfo = lambda h, kind: 2700
    nvml.nvmlDeviceGetMaxClockInfo = lambda h, kind: 3105
    nvml.nvmlDeviceGetUtilizationRates = lambda h: types.SimpleNamespace(gpu=0)
    nvml.nvmlDeviceGetCurrPcieLinkWidth = not_supported if pending_index == 0 else (lambda h: 8)
    nvml.nvmlDeviceGetMaxPcieLinkWidth = lambda h: 16
    return nvml


def test_health_mode_lists_a_lost_card_and_keeps_reading_the_rest(monkeypatch):
    nvml = _fake_pynvml(lost_index=1, pending_index=2)
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    nvml_health = _probe_namespace(
        "nvml_health", "_read_counters", "nvml_recovery_action", "FI_GPU_RECOVERY_ACTION"
    )["nvml_health"]

    snapshot = nvml_health()

    assert snapshot["available"] is True and snapshot["count"] == 3
    lost, pending = snapshot["gpus"][1], snapshot["gpus"][2]
    assert lost == {"index": 1, "error": "GPU is lost"}
    assert pending["retired_pages_pending"] is True and pending["uuid"] == "GPU-2"
    # NOT_SUPPORTED is not a read error
    assert "read_errors" not in snapshot["gpus"][0]
    # the recovery action is field 230 (the pinned binding has no getter); a per-field error is None
    assert snapshot["gpus"][0]["recovery_action"] is None and pending["recovery_action"] == 1
    assert nvml.asked_fields == [[230], [230]]
    assert pending["gfx_clock_mhz"] == 2700 and pending["gfx_clock_max_mhz"] == 3105
    assert pending["util_pct"] == 0 and pending["pcie_width"] == 8 and pending["pcie_width_max"] == 16


def test_a_failed_signal_read_is_unknown_not_a_read_error(monkeypatch):
    nvml = _fake_pynvml(lost_index=1, pending_index=0)
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    nvml_health = _probe_namespace(
        "nvml_health", "_read_counters", "nvml_recovery_action", "FI_GPU_RECOVERY_ACTION"
    )["nvml_health"]

    card = nvml_health()["gpus"][0]

    assert card["pcie_width"] is None and "read_errors" not in card


def test_health_report_gives_each_card_only_its_own_hardware_xids():
    namespace = _probe_namespace("health_report", "parse_xid", "pci_key", "HEALTH_XIDS")
    namespace["re"] = re
    namespace["nvml_health"] = None
    namespace["xid_lines"] = lambda: {
        "available": True,
        "count": 3,
        "last": [],
        "lines": [
            "NVRM: Xid (PCI:0000:01:00): 79, pid=0, GPU has fallen off the bus.",
            "NVRM: Xid (PCI:0000:02:00): 31, pid=42, application fault",
            "NVRM: Xid (PCI:0000:02:00): 94, pid=0, Contained ECC error",
        ],
    }
    namespace["nvml_snapshot_forked"] = lambda mp, read: {
        "available": True,
        "count": 3,
        "gpus": [
            {"index": 0, "pci_bus_id": "00000000:01:00.0"},
            {"index": 1, "pci_bus_id": "00000000:02:00.0"},
            {"index": 2, "pci_bus_id": "00000000:03:00.0"},
        ],
    }

    report = namespace["health_report"](None)

    assert report["status"] == "health"
    assert [gpu["hardware_xids"] for gpu in report["nvml"]["gpus"]] == [[79], [94], []]
    assert [line["xid"] for line in report["xid"]["hardware"]] == [79, 94]
    assert "lines" not in report["xid"]
