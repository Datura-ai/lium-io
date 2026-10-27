import json
from datetime import UTC, datetime
from unittest.mock import patch

import pytest
from neurons.validators.src.protocol.vc_protocol.compute_requests import RentedExecutorsResponse
from neurons.validators.src.services.const import FILLER_CONTAINER_PREFIX
from neurons.validators.src.services.task import pipeline_factory
from neurons.validators.src.services.task.checks import nvlink_fault
from neurons.validators.src.services.task.checks.nvlink_fault import (
    PROBE_JSON_MARKER,
    PROBE_SOURCE,
    NvlinkFaultCheck,
)
from neurons.validators.src.services.task.messages import NvlinkFaultMessages as Msg
from neurons.validators.src.services.task.runner import SSHCommandResult

from tests.helpers import build_context_config, build_services, build_state, default_executor

NVLINK_BOARD = {"interconnect": {"nvlink": True, "nvlink_links": 18, "nvlink_active_links": 18}}
PCIE_BOARD = {"interconnect": {"nvlink": False, "pcie_class": "SYS"}}


class FakeRunner:
    def __init__(self, stdout: str = ""):
        self.stdout = stdout
        self.calls: list[dict] = []

    async def run(self, cmd, *, timeout=60, check=False, retryable=True, stdin_text=None):
        self.calls.append({"cmd": cmd, "stdin_text": stdin_text})
        now = datetime.now(UTC)
        return SSHCommandResult(
            command=cmd,
            command_id="cid",
            exit_code=0,
            stdout=self.stdout,
            stderr="",
            duration_ms=900,
            started_at=now,
            finished_at=now,
            success=True,
            error_type=None,
            error_message=None,
        )


def report(**fields) -> str:
    return "noise\n" + PROBE_JSON_MARKER + " " + json.dumps(fields) + "\n"


async def run_check(context_factory, specs, stdout="", rented_data=None):
    runner = FakeRunner(stdout)
    ctx = context_factory(
        services=build_services(),
        config=build_context_config(),
        state=build_state(specs=specs, rented_data=rented_data),
        runner=runner,
    )
    return await NvlinkFaultCheck().run(ctx), runner


@pytest.mark.asyncio
async def test_a_healthy_nvlink_board_passes(context_factory):
    result, runner = await run_check(context_factory, NVLINK_BOARD, report(status="ok", gpus=8))

    assert result.passed is True
    assert result.event.reason_code == Msg.PROBE_OK.reason
    assert runner.calls[0]["stdin_text"] == PROBE_SOURCE


FAULT = report(status="fault", error="gpu 0: cuMulticastBindMem -> CUresult 401")


@pytest.mark.asyncio
async def test_a_board_whose_nvlink_multicast_cannot_bind_fails_when_enforced(context_factory):
    with patch.object(nvlink_fault.settings, "NVLINK_FAULT_ENFORCEMENT_ENABLED", True):
        result, _ = await run_check(context_factory, NVLINK_BOARD, FAULT)

    assert result.passed is False
    assert NvlinkFaultCheck.fatal is True
    assert result.event.reason_code == "NVLINK_FAULT"
    assert result.event.severity == "error"
    assert "401" in result.event.what_we_saw["probe"]["error"]


@pytest.mark.asyncio
async def test_shadow_mode_logs_the_fault_and_the_node_passes(context_factory):
    # shadow first (taiberium, 23 Oct): ~100 all-NVLink nodes, false-fail count unknown until the logs show it
    assert (
        type(nvlink_fault.settings).model_fields["NVLINK_FAULT_ENFORCEMENT_ENABLED"].default
        is False
    )
    with patch.object(nvlink_fault.settings, "NVLINK_FAULT_ENFORCEMENT_ENABLED", False):
        result, _ = await run_check(context_factory, NVLINK_BOARD, FAULT)

    assert result.passed is True
    assert result.event.reason_code == "NVLINK_FAULT"
    assert result.event.severity == "warning"
    assert "NOT changed" in result.event.impact


@pytest.mark.asyncio
@pytest.mark.parametrize("specs", [PCIE_BOARD, {}, {"interconnect": None}])
async def test_a_pcie_or_unknown_board_is_not_probed(context_factory, specs):
    result, runner = await run_check(context_factory, specs)

    assert result.passed is True
    assert result.event.reason_code == Msg.SKIPPED.reason
    assert runner.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stdout",
    [
        report(status="unsupported", error="driver has no cuMulticastCreate"),
        report(status="error", error="cuMulticastCreate -> CUresult 1"),
        "no marker at all",
    ],
)
async def test_unsupported_or_unmeasured_passes(context_factory, stdout):
    result, _ = await run_check(context_factory, NVLINK_BOARD, stdout)

    assert result.passed is True
    assert result.event.reason_code == Msg.UNKNOWN.reason


def test_runs_unrented_after_the_gpu_probes_in_every_pipeline():
    factory = pipeline_factory.PipelineFactory
    for build in (factory.build_checks, factory.build_dry_run_checks):
        ids = [check.check_id for check in build()]
        assert ids.index("gpu.validate.nvlink_fault") > ids.index("gpu.validate.pinned_memory")


# --- the probe script itself, against a fake libcuda ------------------------------------------------


def load_probe() -> dict:
    namespace: dict = {"__name__": "probe"}
    exec(compile(PROBE_SOURCE, "nvlink_multicast_probe.py", "exec"), namespace)
    return namespace


class FakeCuda:
    """libcuda stand-in: every call succeeds unless `codes` names it."""

    def __init__(self, gpus=8, multicast=1, codes=None, missing=()):
        self.codes = codes or {}
        names = [
            "cuInit",
            "cuDeviceGetCount",
            "cuDeviceGet",
            "cuDeviceGetAttribute",
            "cuMulticastGetGranularity",
            "cuMulticastCreate",
            "cuMulticastAddDevice",
            "cuDevicePrimaryCtxRetain",
            "cuCtxSetCurrent",
            "cuMemCreate",
            "cuMulticastBindMem",
        ]
        outputs = {
            "cuDeviceGetCount": gpus,
            "cuDeviceGetAttribute": multicast,
            "cuMulticastGetGranularity": 2 * 1024 * 1024,
        }
        for name in names:
            if name not in missing:
                setattr(self, name, self._fn(name, outputs.get(name)))

    def _fn(self, name, output):
        def fn(*args):
            if output is not None:
                args[0]._obj.value = output
            if name == "cuDeviceGet":
                args[0]._obj.value = args[1]
            return self.codes.get(name, 0)

        return fn


def test_probe_binds_on_a_healthy_board():
    probe = load_probe()
    assert probe["probe"](FakeCuda()) == {"status": "ok", "gpus": 8, "size_mb": 2}


def test_probe_reports_a_bind_in_illegal_state_as_a_fault():
    probe = load_probe()
    with pytest.raises(probe["Fault"], match="cuMulticastBindMem -> CUresult 401"):
        probe["probe"](FakeCuda(codes={"cuMulticastBindMem": 401}))


@pytest.mark.parametrize(
    "lib",
    [
        FakeCuda(missing=("cuMulticastCreate",)),  # driver before the multicast API
        FakeCuda(multicast=0),  # PCIe card / no NVSwitch
        FakeCuda(gpus=1),
    ],
)
def test_probe_calls_a_board_it_cannot_judge_unsupported(lib):
    probe = load_probe()
    with pytest.raises(probe["Unsupported"]):
        probe["probe"](lib)


def test_probe_reports_any_other_error_as_could_not_measure():
    probe = load_probe()
    with pytest.raises(probe["ProbeError"]):
        probe["probe"](FakeCuda(codes={"cuMulticastCreate": 1}))


@pytest.mark.asyncio
async def test_active_filler_skips_the_probe(context_factory):
    rented_data = RentedExecutorsResponse(
        executors={},
        banned_guids=[],
        filler_containers_by_executor={default_executor().uuid: f"{FILLER_CONTAINER_PREFIX}active"},
    )

    result, runner = await run_check(context_factory, NVLINK_BOARD, FAULT, rented_data=rented_data)

    assert result.passed is True
    assert result.event.reason_code == Msg.FILLER_SKIPPED.reason
    assert runner.calls == []
