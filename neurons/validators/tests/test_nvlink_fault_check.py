import json
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

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


async def run_check(context_factory, specs, stdout="", rented_data=None, services=None, ssh=None):
    runner = FakeRunner(stdout)
    ctx = context_factory(
        services=services or build_services(),
        config=build_context_config(),
        state=build_state(specs=specs, rented_data=rented_data),
        runner=runner,
        ssh=ssh,
    )
    return await NvlinkFaultCheck().run(ctx), runner


FAULT_KEY = nvlink_fault.fault_key("miner-hotkey", default_executor().uuid)
CHECKED_KEY = nvlink_fault.checked_key("miner-hotkey", default_executor().uuid)
FILLER = f"{FILLER_CONTAINER_PREFIX}active"


def filler_node(owner: str = "lium") -> RentedExecutorsResponse:
    uuid = default_executor().uuid
    return RentedExecutorsResponse(
        executors={},
        banned_guids=[],
        all_filler_containers_by_executor={uuid: [FILLER]},
        default_job_owner_by_executor={uuid: owner},
    )


def removing_ssh() -> AsyncMock:
    ssh = AsyncMock()
    ssh.run.return_value = SimpleNamespace(exit_status=0, stdout="", stderr="")
    return ssh


@pytest.mark.asyncio
async def test_a_healthy_nvlink_board_passes(context_factory):
    result, runner = await run_check(context_factory, NVLINK_BOARD, report(status="ok", gpus=8))

    assert result.passed is True
    assert result.event.reason_code == Msg.PROBE_OK.reason
    assert runner.calls[0]["stdin_text"] == PROBE_SOURCE


FAULT = report(status="fault", error="gpu 0: cuMulticastBindMem -> CUresult 401")


@pytest.mark.asyncio
async def test_a_board_whose_nvlink_multicast_cannot_bind_fails(context_factory):
    services = build_services()
    services.redis.store[CHECKED_KEY] = "1"

    result, _ = await run_check(context_factory, NVLINK_BOARD, FAULT, services=services)

    assert result.passed is False
    assert NvlinkFaultCheck.fatal is True
    assert result.event.reason_code == "NVLINK_FAULT"
    assert result.event.severity == "error"
    assert "401" in result.event.what_we_saw["probe"]["error"]
    assert services.redis.ttl[FAULT_KEY] == nvlink_fault.FAULT_TTL_SECONDS
    assert CHECKED_KEY not in services.redis.store


@pytest.mark.asyncio
async def test_a_pass_records_a_fresh_check_and_forgets_an_old_fault(context_factory):
    services = build_services()
    services.redis.store[FAULT_KEY] = "1"

    result, _ = await run_check(
        context_factory, NVLINK_BOARD, report(status="ok", gpus=8), services=services
    )

    assert result.passed is True
    assert services.redis.ttl[CHECKED_KEY] == nvlink_fault.CHECKED_TTL_SECONDS
    assert FAULT_KEY not in services.redis.store


@pytest.mark.asyncio
async def test_a_probe_without_a_verdict_still_counts_as_a_fresh_check(context_factory):
    # a board that cannot be measured must not lose its filler on every cycle
    services = build_services()
    services.redis.store[FAULT_KEY] = "1"

    result, _ = await run_check(
        context_factory, NVLINK_BOARD, report(status="unsupported"), services=services
    )

    assert result.passed is True
    assert services.redis.ttl[CHECKED_KEY] == nvlink_fault.CHECKED_TTL_SECONDS
    # only a pass clears a fault: a host can make the probe report nothing at will
    assert FAULT_KEY in services.redis.store


@pytest.mark.asyncio
async def test_redis_down_does_not_change_the_verdict_on_an_idle_node(context_factory):
    services = build_services()
    services.redis.failing = True

    result, _ = await run_check(context_factory, NVLINK_BOARD, FAULT, services=services)

    assert result.passed is False
    assert result.event.reason_code == "NVLINK_FAULT"


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
async def test_a_filler_on_a_node_with_a_fresh_check_skips_the_probe(context_factory):
    services = build_services()
    services.redis.store[CHECKED_KEY] = "1"
    ssh = removing_ssh()

    result, runner = await run_check(
        context_factory, NVLINK_BOARD, FAULT, rented_data=filler_node(), services=services, ssh=ssh
    )

    assert result.passed is True
    assert result.event.reason_code == Msg.FILLER_SKIPPED.reason
    assert runner.calls == []
    ssh.run.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_filler_does_not_hide_a_faulted_node(context_factory):
    # no fresh check on record: remove our filler and probe, so a filler never keeps a broken node listed
    ssh = removing_ssh()

    result, runner = await run_check(
        context_factory, NVLINK_BOARD, FAULT, rented_data=filler_node(), ssh=ssh
    )

    assert ssh.run.await_args.args[0] == f"/usr/bin/docker rm -fv {FILLER}"
    assert result.event.what_we_saw["removed_fillers"] == [FILLER]
    assert len(runner.calls) == 1
    assert result.passed is False
    assert result.event.reason_code == Msg.PROBE_FAILED.reason


@pytest.mark.asyncio
async def test_a_remembered_fault_fails_a_node_whose_filler_cannot_be_removed(context_factory):
    services = build_services()
    services.redis.store[FAULT_KEY] = "1"

    result, runner = await run_check(
        context_factory,
        NVLINK_BOARD,
        report(status="ok"),
        rented_data=filler_node(owner="miner"),
        services=services,
    )

    assert runner.calls == []
    assert result.passed is False
    assert result.event.reason_code == Msg.PROBE_FAILED.reason
    assert result.event.what_we_saw["remembered_fault"] is True


@pytest.mark.asyncio
async def test_a_filler_that_cannot_be_removed_and_no_fault_on_record_skips(context_factory):
    result, runner = await run_check(
        context_factory, NVLINK_BOARD, FAULT, rented_data=filler_node(owner="miner")
    )

    assert runner.calls == []
    assert result.passed is True
    assert result.event.reason_code == Msg.FILLER_SKIPPED.reason


@pytest.mark.asyncio
async def test_redis_down_keeps_the_filler_and_skips(context_factory):
    services = build_services()
    services.redis.failing = True
    ssh = removing_ssh()

    result, runner = await run_check(
        context_factory, NVLINK_BOARD, FAULT, rented_data=filler_node(), services=services, ssh=ssh
    )

    assert result.passed is True
    assert result.event.reason_code == Msg.FILLER_SKIPPED.reason
    assert runner.calls == []
    ssh.run.assert_not_awaited()
