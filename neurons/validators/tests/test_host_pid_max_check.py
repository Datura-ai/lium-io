"""DAH-7041: validation lists only a node whose host kernel.pid_max the rental create accepts."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
from services.docker_service import DockerService
from services.host_pid_max import pid_max_refused
from services.task.availability import AvailabilityErrorCode
from services.task.checks.host_pid_max import HostPidMaxCheck
from services.task.pipeline_factory import PipelineFactory
from services.task.runner import SSHCommandRunner
from ssh_channel_double import create_process_answering_through

from tests.helpers import build_state, make_context
from tests.test_sysbox_required_check import _rented_data

# what a host answered to `cat /proc/sys/kernel/pid_max`: (exit status, stdout)
UNUSABLE_ANSWERS = [
    (0, ""),  # answered with no number (24 Nov 2026 node 0e25f768: "unparsable output")
    (0, "Welcome to the node\n4194304\n"),  # a profile banner on a non-interactive command
    (0, "0\n"),
    (0, "4096\n"),  # below the host PID margin
    (1, "cat: /proc/sys/kernel/pid_max: Permission denied\n"),
]
USABLE_ANSWERS = [(0, "4194304\n"), (0, "32768\n")]


def _answer(exit_status: int, stdout: str):
    return Mock(exit_status=exit_status, stdout=stdout, stderr="")


def _ctx(ssh, rented: bool = False):
    state = build_state(rented_data=_rented_data() if rented else None)
    return make_context(ssh=ssh, runner=SSHCommandRunner(ssh), state=state)


def _ssh(exit_status: int, stdout: str):
    ssh = AsyncMock()
    ssh.run = AsyncMock(return_value=_answer(exit_status, stdout))
    return ssh


@pytest.mark.asyncio
@pytest.mark.parametrize(("exit_status", "stdout"), UNUSABLE_ANSWERS)
async def test_an_unrented_node_with_an_unusable_pid_max_is_hidden(exit_status, stdout):
    result = await HostPidMaxCheck().run(_ctx(_ssh(exit_status, stdout)))

    assert result.passed is False
    assert result.fatal is True
    assert result.event.reason_code == AvailabilityErrorCode.HOST_PID_MAX_UNUSABLE
    assert result.event.is_availability_error


@pytest.mark.asyncio
@pytest.mark.parametrize(("exit_status", "stdout"), UNUSABLE_ANSWERS[:1])
async def test_a_rented_node_is_hidden_and_its_rental_runs_on(exit_status, stdout):
    result = await HostPidMaxCheck().run(_ctx(_ssh(exit_status, stdout), rented=True))

    assert result.passed is False
    assert result.fatal is False
    assert result.event.is_availability_error


@pytest.mark.asyncio
@pytest.mark.parametrize(("exit_status", "stdout"), USABLE_ANSWERS)
async def test_a_usable_pid_max_passes(exit_status, stdout):
    result = await HostPidMaxCheck().run(_ctx(_ssh(exit_status, stdout)))

    assert result.passed is True
    assert result.event.reason_code == "HOST_PID_MAX_OK"


@pytest.mark.asyncio
async def test_a_read_with_no_answer_says_nothing_about_the_host():
    ssh = AsyncMock()
    ssh.run = AsyncMock(side_effect=asyncio.TimeoutError)

    result = await HostPidMaxCheck().run(_ctx(ssh))

    assert result.passed is True
    assert result.event.reason_code == "HOST_PID_MAX_NOT_READ"


@pytest.mark.asyncio
async def test_nothing_is_read_while_the_pids_limit_is_off(monkeypatch):
    monkeypatch.setattr("core.config.settings.RENTAL_PIDS_LIMIT_PER_CPU", 0)
    ssh = _ssh(0, "")

    result = await HostPidMaxCheck().run(_ctx(ssh))

    assert result.passed is True
    ssh.run.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(("exit_status", "stdout"), UNUSABLE_ANSWERS + USABLE_ANSWERS)
async def test_validation_and_the_create_agree_on_every_answer(exit_status, stdout):
    """The create's own read: a node validation lists is one whose rental the create does not refuse."""
    svc = DockerService(ssh_service=Mock(), redis_service=Mock(), attestation_service=Mock())
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(return_value=_answer(exit_status, stdout))
    ssh_client.create_process = create_process_answering_through(lambda command: ssh_client.run(command))
    create_refuses = pid_max_refused(await svc._read_host_pid_max(ssh_client))

    listed = (await HostPidMaxCheck().run(_ctx(_ssh(exit_status, stdout)))).passed

    assert listed is not create_refuses


def _names(checks):
    return [type(c).__name__ for c in checks]


@pytest.mark.parametrize(
    "build",
    [PipelineFactory.build_checks, PipelineFactory.build_fast_path_checks, PipelineFactory.build_dry_run_checks],
)
def test_every_pipeline_reads_it_after_the_non_fatal_checks(build):
    """It halts an idle node's cycle: the non-fatal checks before the fatal GPU checks (the GPU dropout record among
    them) must still run on that cycle, or a dropout seen with it is never recorded."""
    names = _names(build())
    at = names.index("HostPidMaxCheck")
    assert at > names.index("RentedGpuDropCheck") and at > names.index("GpuLossHoldCheck")
    assert names[at + 1] == "GpuCountCheck"
