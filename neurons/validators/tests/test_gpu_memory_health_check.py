"""DAH-3110: a card NVML reports broken (uncorrected ECC, pending/failed row remap, recovery
action) fails the node like a GPU ban, on rented and idle nodes alike; a live rental is kept."""

import json
import subprocess
import sys
from unittest.mock import AsyncMock

import pytest
from neurons.validators.src.services.task import pipeline_factory
from neurons.validators.src.services.task.checks.gpu_fault_probe import (
    PROBE_JSON_MARKER,
    PROBE_SOURCE,
)
from neurons.validators.src.services.task.checks.gpu_memory_health import GpuMemoryHealthCheck
from neurons.validators.src.services.task.messages import GpuMemoryHealthMessages as Msg
from neurons.validators.src.services.task.pipeline import Pipeline
from protocol.vc_protocol.compute_requests import RentedExecutor, RentedExecutorsResponse, RentedPod

from core.config import settings
from tests.helpers import default_executor
from tests.test_gpu_fault_probe_check import FakeRunner, _probe_namespace, make_ctx

# a card seen on a rented 8x RTX PRO 6000 node, 14 Oct: 1 uncorrectable DRAM ECC error, a pending
# row remap and "Drain and Reset" as the recovery action
BROKEN = {
    "index": 2,
    "uuid": "GPU-2",
    "ecc_uncorrected": 1,
    "remapped_rows": [0, 1, 1, 0],
    "recovery_action": 4,
}
HEALTHY = {
    "index": 0,
    "uuid": "GPU-0",
    "ecc_uncorrected": 0,
    "remapped_rows": [0, 0, 0, 0],
    "recovery_action": 0,
}


def _report(status: str, **extra) -> str:
    return PROBE_JSON_MARKER + " " + json.dumps({"status": status, **extra}) + "\n"


def test_the_standing_faults_of_a_broken_card_are_each_flagged():
    nvml_health = _probe_namespace("nvml_health", "RECOVERY_ACTIONS")["nvml_health"]

    flags = nvml_health({"available": True, "gpus": [HEALTHY, BROKEN]})

    assert flags == [
        "gpu 2 (GPU-2): 1 uncorrected ECC error(s) since the driver loaded",
        "gpu 2 (GPU-2): row remap pending (needs a GPU reset)",
        "gpu 2 (GPU-2): NVML recovery action DRAIN_AND_RESET required",
    ]


def test_a_card_whose_reads_are_not_supported_is_not_flagged():
    nvml_health = _probe_namespace("nvml_health", "RECOVERY_ACTIONS")["nvml_health"]
    consumer = {"index": 0, "ecc_uncorrected": None, "remapped_rows": None, "recovery_action": None}

    assert nvml_health({"available": True, "gpus": [consumer]}) == []


def test_the_snapshot_reads_the_recovery_action_through_field_value_230(monkeypatch):
    # released bindings have no nvmlDeviceGetGpuRecoveryAction; the action is field 230 (review)
    from types import ModuleType, SimpleNamespace

    namespace = _probe_namespace("nvml_snapshot", "nvml_recovery_action", "FI_GPU_RECOVERY_ACTION")
    asked = []

    def field_values(handle, ids):
        asked.append(ids)
        code = 0 if handle == 0 else 3  # the second card does not support the field
        return [SimpleNamespace(nvmlReturn=code, value=SimpleNamespace(uiVal=4))]

    class NVMLError(Exception):
        pass

    fake = ModuleType("pynvml")
    fake.__dict__.update(
        NVML_SUCCESS=0,
        NVMLError=NVMLError,
        NVML_MEMORY_ERROR_TYPE_UNCORRECTED=1,
        NVML_VOLATILE_ECC=0,
        nvmlInit=lambda: None,
        nvmlShutdown=lambda: None,
        nvmlDeviceGetCount=lambda: 2,
        nvmlDeviceGetHandleByIndex=lambda i: i,
        nvmlDeviceGetUUID=lambda h: f"GPU-{h}",
        nvmlDeviceGetPciInfo=lambda h: SimpleNamespace(busId=b"0000:01:00.0"),
        nvmlDeviceGetTotalEccErrors=lambda h, kind, counter: 0,
        nvmlDeviceGetRemappedRows=lambda h: (0, 0, 0, 0),
        nvmlDeviceGetFieldValues=field_values,
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake)

    gpus = namespace["nvml_snapshot"]()["gpus"]

    assert [gpu["recovery_action"] for gpu in gpus] == [4, None]
    assert asked == [[230], [230]]


def test_nvml_only_runs_no_cuda_and_always_prints_a_report():
    # no NVML binding here: the mode still prints its marker line and exits 0
    run = subprocess.run(
        [sys.executable, "-I", "-", "--nvml-only"],
        input=PROBE_SOURCE,
        capture_output=True,
        text=True,
        timeout=60,
    )

    line = next(line for line in run.stdout.splitlines() if line.startswith(PROBE_JSON_MARKER))
    report = json.loads(line[len(PROBE_JSON_MARKER) :])
    assert run.returncode == 0
    assert report["status"] in ("ok", "flagged", "error")
    assert "devices" not in report


FLAGS = ["gpu 2 (GPU-2): 1 uncorrected ECC error(s) since the driver loaded"]


def _live_rental() -> RentedExecutorsResponse:
    executor = default_executor()
    return RentedExecutorsResponse(
        executors={
            executor.uuid: RentedExecutor(
                miner_hotkey="miner-hotkey",
                executor_ip_address=executor.address,
                executor_ip_port=str(executor.port),
                pods=[RentedPod(pod_id="pod-1", container_name="pod_p1")],
            )
        },
        banned_guids=[],
    )


@pytest.mark.asyncio
async def test_a_broken_card_fails_an_idle_node_and_stops_its_pipeline(context_factory):
    runner = FakeRunner(_report("flagged", flags=FLAGS))
    ctx = make_ctx(context_factory, runner)

    result = await GpuMemoryHealthCheck().run(ctx)
    success, _, _ = await Pipeline(checks=[GpuMemoryHealthCheck()], sink=AsyncMock()).run(ctx)

    assert (result.passed, result.fatal, result.event.reason_code) == (
        False,
        True,
        Msg.FAILED.reason,
    )
    assert result.updates == {"clear_verified_job_info": True}
    assert result.event.what_we_saw["flags"] == FLAGS
    assert success is False
    assert runner.calls[0]["cmd"].endswith("-I - --nvml-only")
    assert runner.calls[0]["stdin_text"] == PROBE_SOURCE


@pytest.mark.asyncio
async def test_a_broken_card_under_a_live_rental_fails_without_stopping_verification(
    context_factory,
):
    ctx = make_ctx(
        context_factory, FakeRunner(_report("flagged", flags=FLAGS)), rented_data=_live_rental()
    )

    result = await GpuMemoryHealthCheck().run(ctx)

    assert (result.passed, result.fatal) == (False, False)
    assert result.updates == {"clear_verified_job_info": True}


@pytest.mark.asyncio
async def test_with_enforcement_off_a_broken_card_is_only_flagged(context_factory, monkeypatch):
    monkeypatch.setattr(settings, "GPU_MEMORY_HEALTH_ENFORCEMENT_ENABLED", False)
    runner = FakeRunner(_report("flagged", flags=FLAGS))

    result = await GpuMemoryHealthCheck().run(make_ctx(context_factory, runner))

    assert (result.passed, result.event.reason_code) == (True, Msg.FLAGGED.reason)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stdout, reason",
    [
        (_report("ok", flags=[]), Msg.OK.reason),
        (_report("error", flags=[], nvml={"available": False}), Msg.UNKNOWN.reason),
        ("no marker", Msg.UNKNOWN.reason),
    ],
)
async def test_a_healthy_or_unread_node_passes(context_factory, stdout, reason):
    result = await GpuMemoryHealthCheck().run(make_ctx(context_factory, FakeRunner(stdout)))

    assert (result.passed, result.event.reason_code) == (True, reason)


def test_the_read_runs_before_the_rented_short_circuit_in_every_pipeline():
    for build in (
        pipeline_factory.PipelineFactory.build_checks,
        pipeline_factory.PipelineFactory.build_fast_path_checks,
        pipeline_factory.PipelineFactory.build_dry_run_checks,
    ):
        ids = [check.check_id for check in build()]
        assert ids.index("gpu.validate.memory_health") < ids.index("executor.validate.rented_state")
