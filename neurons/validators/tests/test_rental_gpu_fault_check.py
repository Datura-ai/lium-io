"""DAH-3490: the mid-rental check reads the host's Xid lines per rented pod and tells the backend once."""

from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from neurons.validators.src.protocol.vc_protocol.compute_requests import (
    RentedExecutor,
    RentedExecutorsResponse,
    RentedPod,
)
from neurons.validators.src.services.gpu_xid_attribution import (
    CONTAINER_PIDS_COMMAND,
    CONTAINER_STARTED_AT_COMMAND,
    ECC_QUERY_COMMAND,
    XID_LOG_COMMAND,
)
from neurons.validators.src.services.task import pipeline_factory
from neurons.validators.src.services.task.checks.rental_gpu_fault import (
    PHASE_MID_RENTAL,
    REPORTED_WORKLOAD_LINES_KEY,
    RentalGpuFaultCheck,
)
from neurons.validators.src.services.task.messages import RentalGpuFaultMessages as Msg
from neurons.validators.src.services.task.runner import SSHCommandResult

from tests.helpers import build_services, build_state, default_executor

POD = "11655dc5-53ba-4a8d-a341-fe6c9d12bda7"
CONTAINER = f"pod_{POD}"
STARTED = datetime.now(UTC) - timedelta(hours=1)


def xid(minutes_after_start: int, code: int, pid: int | None = 4242) -> str:
    stamp = (STARTED + timedelta(minutes=minutes_after_start)).strftime("%Y-%m-%dT%H:%M:%S")
    pid_part = f"pid={pid}, name=python, " if pid is not None else ""
    return f"{stamp},000000+00:00 NVRM: Xid (PCI:0000:81:00): {code}, {pid_part}Ch 00000008"


class ScriptedRunner:
    """Answers each host command from a script keyed by the command text; records the order."""

    def __init__(self, answers: dict[str, tuple[int, str]]):
        self.answers = answers
        self.calls: list[str] = []

    async def run(self, cmd, *, timeout=60, check=False, retryable=True, stdin_text=None):
        self.calls.append(cmd)
        exit_code, stdout = self.answers.get(cmd, (1, ""))
        now = datetime.now(UTC)
        return SSHCommandResult(
            command=cmd,
            command_id="cid",
            exit_code=exit_code,
            stdout=stdout,
            stderr="",
            duration_ms=1,
            started_at=now,
            finished_at=now,
            success=exit_code == 0,
        )


def host(xid_lines: str, *, ecc: str = "00000000:81:00.0, GPU-aaaa, 0\n", pids: str = "PID\n4242\n4243\n") -> ScriptedRunner:
    return ScriptedRunner(
        {
            XID_LOG_COMMAND: (0, xid_lines),
            ECC_QUERY_COMMAND: (0, ecc),
            CONTAINER_STARTED_AT_COMMAND.format(name=CONTAINER): (0, STARTED.strftime("%Y-%m-%dT%H:%M:%S.000000000Z") + "\n"),
            CONTAINER_PIDS_COMMAND.format(name=CONTAINER): (0, pids),
        }
    )


def rented() -> RentedExecutorsResponse:
    executor = default_executor()
    return RentedExecutorsResponse(
        executors={
            executor.uuid: RentedExecutor(
                miner_hotkey="miner-hotkey",
                executor_ip_address=executor.address,
                executor_ip_port=str(executor.port),
                pods=[RentedPod(pod_id=POD, container_name=CONTAINER)],
            )
        }
    )


@contextmanager
def probe_flag(enabled: bool = True):
    with patch("neurons.validators.src.services.task.checks.rental_gpu_fault.settings") as s:
        s.RENTAL_GPU_FAULT_PROBE_ENABLED = enabled
        yield s


def make_ctx(context_factory, runner, *, rented_data=rented(), redis=None):
    services = build_services(redis=redis)
    return context_factory(services=services, state=build_state(rented_data=rented_data), runner=runner), services


@pytest.mark.asyncio
async def test_a_renters_application_xid_is_reported_to_the_backend_once_per_rental(context_factory):
    """On origin/main nothing reads the kernel log during a rental, so the renter is never told their
    workload broke a card and the provider is never told they are still paid; the check does not exist."""
    runner = host(xid(10, 31))
    ctx, services = make_ctx(context_factory, runner)

    with probe_flag():
        result = await RentalGpuFaultCheck().run(ctx)

    assert result.passed is True  # the score is never touched mid-rental
    assert result.event.reason_code == Msg.WORKLOAD_FAULT.reason
    assert result.event.what_we_saw["pods"][POD]["attribution"] == "workload"
    assert result.event.what_we_saw["reported_pod_ids"] == [POD]
    services.backend.report_gpu_fault_probe.assert_awaited_once()
    executor_uuid, report = services.backend.report_gpu_fault_probe.await_args.args
    assert executor_uuid == ctx.executor.uuid
    assert report["pod_id"] == POD and report["phase"] == PHASE_MID_RENTAL
    assert len(report["workload_xids"]) == 1 and report["hardware_xids"] == []
    # the host is read, never the container: dmesg and nvidia-smi on the executor, docker inspect/top for the window and the PIDs
    assert runner.calls[0] == XID_LOG_COMMAND
    assert CONTAINER_PIDS_COMMAND.format(name=CONTAINER) in runner.calls


@pytest.mark.asyncio
async def test_a_hardware_xid_is_the_providers_and_is_not_reported_as_the_workloads(context_factory):
    runner = host(xid(10, 31) + "\n" + xid(11, 79, pid=None))
    ctx, services = make_ctx(context_factory, runner)

    with probe_flag():
        result = await RentalGpuFaultCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.HARDWARE_FAULT.reason
    services.backend.report_gpu_fault_probe.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_quiet_kernel_log_reports_nothing(context_factory):
    ctx, services = make_ctx(context_factory, host(""))
    with probe_flag():
        result = await RentalGpuFaultCheck().run(ctx)
    assert result.event.reason_code == Msg.NO_FAULT.reason
    services.backend.report_gpu_fault_probe.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_unreadable_kernel_log_is_an_error_not_a_quiet_log(context_factory):
    # the pipeline's exit code is tail's; the marker line is how a dmesg that cannot be read is told apart
    runner = host("DMESG_UNAVAILABLE\n")
    ctx, services = make_ctx(context_factory, runner)
    with probe_flag():
        result = await RentalGpuFaultCheck().run(ctx)
    assert result.passed is True and result.event.reason_code == Msg.PROBE_ERROR.reason
    services.backend.report_gpu_fault_probe.assert_not_awaited()


@pytest.mark.asyncio
async def test_host_text_that_does_not_parse_never_costs_the_cycle(context_factory):
    # a stamp the regex accepts but strptime refuses, and a started_at that is not a date
    runner = host("2026-13-45T25:61:61,000000+00:00 NVRM: Xid (PCI:0000:81:00): 31, pid=4242, Ch 00000008\n")
    runner.answers[CONTAINER_STARTED_AT_COMMAND.format(name=CONTAINER)] = (0, "2026-99-99T00:00:00Z\n")
    ctx, _ = make_ctx(context_factory, runner)
    with probe_flag():
        result = await RentalGpuFaultCheck().run(ctx)
    assert result.passed is True and result.event.reason_code == Msg.NO_FAULT.reason


@pytest.mark.asyncio
async def test_an_idle_node_is_not_read(context_factory):
    runner = host(xid(10, 31))
    ctx, services = make_ctx(context_factory, runner, rented_data=None)
    with probe_flag():
        result = await RentalGpuFaultCheck().run(ctx)
    assert result.event.reason_code == Msg.NOT_RENTED.reason
    assert runner.calls == []


@pytest.mark.asyncio
async def test_the_flag_turns_the_check_off(context_factory):
    runner = host(xid(10, 31))
    ctx, _ = make_ctx(context_factory, runner)
    with probe_flag(enabled=False):
        result = await RentalGpuFaultCheck().run(ctx)
    assert result.event.reason_code == Msg.DISABLED.reason
    assert runner.calls == []


@pytest.mark.asyncio
async def test_the_same_fault_is_not_reported_again_on_the_next_cycle(context_factory):
    redis = AsyncMock()
    redis.hget.return_value = "1"  # one workload line was already reported for this pod
    ctx, services = make_ctx(context_factory, host(xid(10, 31)), redis=redis)
    with probe_flag():
        result = await RentalGpuFaultCheck().run(ctx)
    assert result.event.reason_code == Msg.WORKLOAD_FAULT.reason
    assert result.event.what_we_saw["reported_pod_ids"] == []
    services.backend.report_gpu_fault_probe.assert_not_awaited()

    # a second line is new evidence: reported, and the count moves on
    redis.hget.return_value = "1"
    ctx, services = make_ctx(context_factory, host(xid(10, 31) + "\n" + xid(20, 43)), redis=redis)
    with probe_flag():
        await RentalGpuFaultCheck().run(ctx)
    services.backend.report_gpu_fault_probe.assert_awaited_once()
    redis.hset.assert_awaited_with(REPORTED_WORKLOAD_LINES_KEY, POD, "2")


def test_the_check_runs_before_every_fatal_gpu_check():
    # a card that stopped being listed mid-rental halts the pipeline at the count / model / fingerprint / spec-change
    # check; the attribution has to have run by then or the renter and the provider are never told
    ids = [check.check_id for check in pipeline_factory.PipelineFactory.build_checks()]
    ours = ids.index("gpu.validate.rental_fault")
    assert ids.index("gpu.scrape.machine_spec") < ours
    for fatal in ("gpu.validate.count", "gpu.validate.model", "gpu.validate.fingerprint", "gpu.validate.spec_change"):
        assert fatal in ids and ids.index(fatal) > ours, fatal
    assert ids.index("executor.validate.rented_state") > ours
    assert "gpu.validate.rental_fault" not in [check.check_id for check in pipeline_factory.PipelineFactory.build_dry_run_checks()]
