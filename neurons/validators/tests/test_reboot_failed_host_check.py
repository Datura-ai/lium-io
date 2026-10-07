"""After a failed pod reboot the validator checks whether the host can start a GPU container.

A rented pod in REBOOT_FAILED whose container is not running gets POD_NOT_RUNNING, which the backend
skips for a pod under a platform operation, so a host that can no longer start a container stayed
active and billed. With the flag on, the host is asked to start the executor's own image with
`nvidia-smi -L`; a refusal clears the verified job without the POD_NOT_RUNNING reason.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from neurons.validators.src.services.task.checks import rented_machine
from neurons.validators.src.services.task.checks.rented_machine import TenantEnforcementCheck
from neurons.validators.src.services.task.messages import TenantEnforcementMessages as Msg
from neurons.validators.src.services.task.runner import SSHCommandResult
from protocol.vc_protocol.compute_requests import RentedExecutor, RentedExecutorsResponse, RentedPod
from protocol.vc_protocol.validator_requests import ResetVerifiedJobReason
from test_rented_machine_check import (
    DummyBackendClient,
    DummyScoreCalculator,
    DummySSHClient,
    MockContainerCleanup,
)

from helpers import build_context_config, build_services, build_state

EXECUTOR_UUID = "executor-123"
MINER = "miner-hotkey"
CONTAINER_ID = "executor-container-id"
IMAGE_ID = "sha256:" + "a" * 64
NVIDIA_HOOK_ERROR = (
    "docker: Error response from daemon: failed to create task for container: OCI runtime create failed:"
    " nvidia-container-cli: initialization error: nvml error: driver not loaded"
)


class FakeRunner:
    def __init__(self, *, image: SSHCommandResult | None = None, gpu_run: SSHCommandResult | None = None):
        self.image = image or _result(0, stdout=IMAGE_ID)
        self.gpu_run = gpu_run or _result(0, stdout="GPU 0: NVIDIA H100 (UUID: GPU-1)")
        self.commands: list[str] = []

    async def run(self, cmd: str, **kwargs) -> SSHCommandResult:
        self.commands.append(cmd)
        return self.image if "docker inspect" in cmd else self.gpu_run


def _result(exit_code: int, *, stdout: str = "", stderr: str = "", error_type: str | None = None) -> SSHCommandResult:
    now = datetime.now(UTC)
    return SSHCommandResult(
        command="",
        command_id="id",
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        duration_ms=1,
        started_at=now,
        finished_at=now,
        success=exit_code == 0,
        error_type=error_type,
    )


def _context(context_factory, runner: FakeRunner, *, pod_status: str | None = "REBOOT_FAILED", owner: str = MINER):
    rented_data = RentedExecutorsResponse(
        executors={
            EXECUTOR_UUID: RentedExecutor(
                miner_hotkey=owner,
                executor_ip_address="127.0.0.1",
                executor_ip_port="22",
                pods=[RentedPod(pod_id="pod-1", container_name="pod_pod-1", status=pod_status)],
            )
        },
        banned_guids=[],
    )
    recovery = AsyncMock()
    recovery.recover_pod_after_stale_vloopback_mount.return_value = False
    return context_factory(
        services=build_services(
            score_calculator=DummyScoreCalculator(),
            container_cleanup=MockContainerCleanup(),
            backend=DummyBackendClient(active=True),
            pod_recovery=recovery,
        ),
        config=build_context_config(),
        state=build_state(
            specs={"docker": {"container_id": CONTAINER_ID}},
            gpu_processes=[],
            gpu_details=[],
            rented_data=rented_data,
        ),
        ssh=DummySSHClient(pod_running=False),
        runner=runner,
        miner_hotkey=MINER,
        executor_ssh_private_key="ssh-key",
    )


async def _run(ctx, *, enabled: bool = True):
    with patch.object(rented_machine.settings, "REBOOT_FAILED_HOST_CHECK_ENABLED", enabled):
        return await TenantEnforcementCheck().run(ctx)


@pytest.mark.asyncio
async def test_a_host_that_cannot_start_a_gpu_container_after_a_failed_reboot_is_reset(context_factory):
    runner = FakeRunner(gpu_run=_result(125, stderr=NVIDIA_HOOK_ERROR))

    result = await _run(_context(context_factory, runner))

    assert result.passed is False
    assert result.event.reason_code == Msg.REBOOT_FAILED_HOST_FAULT.reason
    assert result.updates["clear_verified_job_info"] is True
    assert "clear_verified_job_reason" not in result.updates
    evidence = result.updates["clear_verified_job_evidence"]
    assert evidence["reason_code"] == "REBOOT_FAILED_HOST_FAULT"
    assert evidence["pod_id"] == "pod-1"
    assert evidence["host_gpu_container"]["exit_code"] == 125
    assert "nvidia-container-cli" in evidence["host_gpu_container"]["stderr_tail"]


@pytest.mark.asyncio
async def test_the_host_check_runs_the_executors_own_image_with_gpus(context_factory):
    runner = FakeRunner(gpu_run=_result(125, stderr=NVIDIA_HOOK_ERROR))

    await _run(_context(context_factory, runner))

    assert CONTAINER_ID in runner.commands[0]
    assert "--gpus all" in runner.commands[1]
    assert f"--entrypoint nvidia-smi {IMAGE_ID} -L" in runner.commands[1]


@pytest.mark.asyncio
async def test_a_healthy_host_keeps_pod_not_running(context_factory):
    runner = FakeRunner()

    result = await _run(_context(context_factory, runner))

    assert result.event.reason_code == Msg.POD_NOT_RUNNING.reason
    assert result.updates["clear_verified_job_reason"] == ResetVerifiedJobReason.POD_NOT_RUNNING.value


@pytest.mark.asyncio
async def test_a_timeout_decides_nothing(context_factory):
    runner = FakeRunner(gpu_run=_result(-1, error_type="timeout"))

    result = await _run(_context(context_factory, runner))

    assert result.event.reason_code == Msg.POD_NOT_RUNNING.reason


@pytest.mark.asyncio
async def test_an_image_id_that_is_not_a_digest_skips_the_check(context_factory):
    runner = FakeRunner(image=_result(0, stdout="busybox; reboot"))

    result = await _run(_context(context_factory, runner))

    assert result.event.reason_code == Msg.POD_NOT_RUNNING.reason
    assert len(runner.commands) == 1


@pytest.mark.asyncio
async def test_the_flag_off_runs_no_host_check(context_factory):
    runner = FakeRunner(gpu_run=_result(125, stderr=NVIDIA_HOOK_ERROR))

    result = await _run(_context(context_factory, runner), enabled=False)

    assert result.event.reason_code == Msg.POD_NOT_RUNNING.reason
    assert runner.commands == []


@pytest.mark.asyncio
async def test_a_running_pod_status_runs_no_host_check(context_factory):
    runner = FakeRunner(gpu_run=_result(125, stderr=NVIDIA_HOOK_ERROR))

    result = await _run(_context(context_factory, runner, pod_status="RUNNING"))

    assert result.event.reason_code == Msg.POD_NOT_RUNNING.reason
    assert runner.commands == []


@pytest.mark.asyncio
async def test_a_rental_another_miner_owns_runs_no_host_check(context_factory):
    runner = FakeRunner(gpu_run=_result(125, stderr=NVIDIA_HOOK_ERROR))

    result = await _run(_context(context_factory, runner, owner="other-miner"))

    assert result.event.reason_code == Msg.POD_NOT_RUNNING.reason
    assert runner.commands == []


def test_the_flag_ships_off():
    assert type(rented_machine.settings).model_fields["REBOOT_FAILED_HOST_CHECK_ENABLED"].default is False
