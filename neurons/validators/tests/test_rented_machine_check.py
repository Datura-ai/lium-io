from dataclasses import replace
import asyncio
import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import asyncssh
import pytest
from helpers import FakeRedis, build_context_config, build_services, build_state
import neurons.validators.src.services.task.checks.rented_machine as rented_machine_module
from neurons.validators.src.core.docker_utils import DockerCommand, _collect_host_context
from neurons.validators.src.services.task.checks.rented_machine import (
    TenantEnforcementCheck,
    _collect_pod_diagnostics,
)
from neurons.validators.src.services.const import POD_CONTAINER_NAME_GRACE_MINUTES
from neurons.validators.src.services.task.messages import TenantEnforcementMessages as Msg
from neurons.validators.src.services.docker_service import DockerService
from neurons.validators.src.services.task.pipeline import Context
from neurons.validators.src.services.task.pipeline_factory import PipelineFactory

from protocol.vc_protocol.compute_requests import RentedExecutor, RentedExecutorsResponse, RentedPod
from protocol.vc_protocol.validator_requests import ResetVerifiedJobReason


class MockContainerCleanup:
    """Mock container cleanup service for tests."""
    async def cleanup(self, ssh_client, rented_data, executor_uuid):
        return 0, []


def convert_rented_machine_to_rented_data(
    rented_machine: dict | None,
    executor_uuid: str = "executor-123",
) -> RentedExecutorsResponse | None:
    """Convert old rented_machine dict format to new RentedExecutorsResponse.

    Args:
        rented_machine: Old format dict with "containers" and "owner_flag" keys
        executor_uuid: The executor UUID to use as key

    Returns:
        RentedExecutorsResponse or None if rented_machine is None or empty
    """
    if not rented_machine:
        return None

    containers = rented_machine.get("containers", [])
    if not containers:
        return None

    pods = [
        RentedPod(pod_id=c.get("pod_id", ""), container_name=c.get("name", ""))
        for c in containers
    ]

    return RentedExecutorsResponse(
        executors={
            executor_uuid: RentedExecutor(
                miner_hotkey="miner-hotkey",
                executor_ip_address="127.0.0.1",
                executor_ip_port="8080",
                pods=pods,
                owner_flag=rented_machine.get("owner_flag", False),
            )
        },
        banned_guids=[],
    )


def build_rented_data(
    executor_uuid: str,
    rented_machine: dict | None,
) -> RentedExecutorsResponse | None:
    """Convert old rented_machine dict format to RentedExecutorsResponse."""
    if not rented_machine:
        return None

    containers = rented_machine.get("containers", [])
    if not containers:
        return RentedExecutorsResponse(executors={}, banned_guids=[])

    pods = [
        RentedPod(
            pod_id=c.get("pod_id", "pod-123"),
            container_name=c.get("name", "container"),
            rented_ports=[],
            created_at=c.get("created_at"),
            status=c.get("status"),
        )
        for c in containers
    ]

    executor = RentedExecutor(
        miner_hotkey="miner-hotkey",
        executor_ip_address="127.0.0.1",
        executor_ip_port="22",
        pods=pods,
        owner_flag=rented_machine.get("owner_flag", False),
    )

    return RentedExecutorsResponse(
        executors={executor_uuid: executor},
        banned_guids=[],
    )


def build_filler_data(executor_uuid: str, filler_container: str) -> RentedExecutorsResponse:
    return RentedExecutorsResponse(
        executors={},
        filler_containers_by_executor={executor_uuid: filler_container},
        banned_guids=[],
    )


class DummySSHClient:
    """Mock SSH client for pod health and SSH keys checks."""

    def __init__(
        self,
        *,
        pod_running: bool = True,
        ssh_keys: list[str] | None = None,
        should_raise: bool = False,
        raise_on_run: BaseException | None = None,
    ):
        """
        Args:
            pod_running: Whether the container is running
            ssh_keys: SSH keys to return from authorized_keys
            should_raise: Whether to raise a generic RuntimeError
            raise_on_run: Specific exception to raise on every .run() call
        """
        self.pod_running = pod_running
        self.ssh_keys = ssh_keys or []
        self.should_raise = should_raise
        self.raise_on_run = raise_on_run
        self.commands_called: list[str] = []
        self.container_finished_at = "2026-08-03T16:59:21.514042646Z"

    async def run(self, command: str):
        """Mock SSH run command."""
        self.commands_called.append(command)

        if self.raise_on_run is not None:
            raise self.raise_on_run

        if self.should_raise:
            raise RuntimeError("SSH command failed")

        result = Mock()
        if "docker ps" in command:
            result.stdout = "container_id_123" if self.pod_running else ""
        elif "authorized_keys" in command:
            result.stdout = "\n".join(self.ssh_keys) if self.ssh_keys else ""
        elif "{{.State.Running}}" in command:
            result.stdout = f"{str(self.pod_running).lower()}\n"
        elif "docker inspect" in command:
            result.stdout = json.dumps(
                {
                    "Status": "exited",
                    "ExitCode": 255,
                    "FinishedAt": self.container_finished_at,
                }
            )
        else:
            result.stdout = ""

        return result


class DummyScoreCalculator:
    """Mock score calculator."""

    def __init__(self, *, actual_score: float = 1.0, job_score: float = 1.0, warning: str = ""):
        self.actual_score = actual_score
        self.job_score = job_score
        self.warning = warning
        self.called_with: dict | None = None

    def __call__(self, ctx, rented: bool):
        self.called_with = {
            "ctx": ctx,
            "rented": rented,
        }
        return self.actual_score, self.job_score, self.warning


class DummyBackendClient:
    def __init__(self, *, active: bool | None = True, local_volume_path: str | None = None):
        self.active = active
        self.local_volume_path = local_volume_path
        self.called_with: list[str] = []
        self.host_reboot_recoveries: list[tuple[str, str]] = []

    async def get_pod_rental_active(self, pod_id: str):
        self.called_with.append(pod_id)
        if self.active is None:
            return None
        return Mock(
            active=self.active,
            rental_closed_at=None,
            local_volume_path=self.local_volume_path,
        )

    async def report_pod_host_reboot_recovered(self, pod_id: str, container_finished_at: str):
        self.host_reboot_recoveries.append((pod_id, container_finished_at))
        return Mock(recorded=True)


@pytest.mark.parametrize(
    "rented_machine,pod_running,ssh_keys,owner_flag,gpu_processes,gpu_details,port_count_db,port_maps,expected_pass,expected_reason,expect_halt",
    [
        # Not rented - should pass and continue
        (None, True, [], False, [], [], 0, [], True, Msg.NOT_RENTED.reason, False),
        # Not rented (empty containers) - should pass and continue
        ({"containers": []}, True, [], False, [], [], 0, [], True, Msg.NOT_RENTED.reason, False),

        # Rented but pod not running - should fail
        (
            {"containers": [{"name": "tenant-123", "pod_id": "pod-1"}]},
            False,
            [],
            False,
            [],
            [],
            0,
            [],
            False,
            Msg.POD_NOT_RUNNING.reason,
            False,
        ),

        # Rented, pod running, no GPU processes outside - should pass with halt
        (
            {"containers": [{"name": "tenant-123", "pod_id": "pod-1"}], "owner_flag": False},
            True,
            ["ssh-rsa AAA..."],
            False,
            [{"container_name": "tenant-123", "pid": 1234}],
            [{"gpu_utilization": 50, "memory_utilization": 60}],
            10,
            [],
            True,
            Msg.ALREADY_RENTED.reason,
            True,
        ),

        # Rented, pod running, GPU process outside but owner_flag=True - should pass with halt
        (
            {"containers": [{"name": "tenant-123", "pod_id": "pod-1"}], "owner_flag": True},
            True,
            [],
            True,
            [{"container_name": "other-container", "pid": 1234}],
            [{"gpu_utilization": 50, "memory_utilization": 60}],
            5,
            [],
            True,
            Msg.ALREADY_RENTED.reason,
            True,
        ),

        # Rented, pod running, GPU process outside, high utilization - should fail
        (
            {"containers": [{"name": "tenant-123", "pod_id": "pod-1"}], "owner_flag": False},
            True,
            ["ssh-rsa AAA..."],
            False,
            [{"container_name": "other-container", "pid": 1234}],
            [{"gpu_utilization": 95, "memory_utilization": 80}],
            10,
            [],
            False,
            Msg.GPU_OUTSIDE_TENANT.reason,
            False,
        ),

        # Rented, pod running, GPU process outside but usage within limits - should pass with halt
        (
            {"containers": [{"name": "tenant-123", "pod_id": "pod-1"}], "owner_flag": False},
            True,
            [],
            False,
            [{"container_name": "other-container", "pid": 1234}],
            [{"gpu_utilization": 3, "memory_utilization": 4}],
            10,
            [],
            True,
            Msg.ALREADY_RENTED.reason,
            True,
        ),

        # Rented, fallback to Redis port maps
        (
            {"containers": [{"name": "tenant-123", "pod_id": "pod-1"}], "owner_flag": False},
            True,
            [],
            False,
            [],
            [],
            0,  # Port count from DB is 0, will fallback to Redis
            [b"8080,9080", b"8081,9081"],  # 2 port mappings
            True,
            Msg.ALREADY_RENTED.reason,
            True,
        ),
    ],
)
@pytest.mark.asyncio
async def test_tenant_enforcement_check(
    rented_machine,
    pod_running,
    ssh_keys,
    owner_flag,
    gpu_processes,
    gpu_details,
    port_count_db,
    port_maps,
    expected_pass,
    expected_reason,
    expect_halt,
    context_factory,
):
    executor_uuid = "executor-123"

    # Build rented_data from old rented_machine format
    rented_data = build_rented_data(executor_uuid, rented_machine)

    # Create mock SSH client
    ssh_client = DummySSHClient(
        pod_running=pod_running,
        ssh_keys=ssh_keys,
    )

    # Create mock score calculator
    score_calculator = DummyScoreCalculator(actual_score=1.0, job_score=1.0, warning="")

    # Setup services
    services = build_services(
        score_calculator=score_calculator,
        container_cleanup=MockContainerCleanup(),
        backend=DummyBackendClient(active=True),
    )

    # Setup config
    config = build_context_config()

    # Setup state with GPU info and rented_data
    state = build_state(
        gpu_processes=gpu_processes,
        gpu_details=gpu_details,
        gpu_model="NVIDIA RTX 4090",
        rented_data=rented_data,
    )

    # Create context
    ctx = context_factory(
        services=services,
        config=config,
        state=state,
        ssh=ssh_client,
        collateral_deposited=True,
        is_rental_succeed=True,
        contract_version="v1.0.0",
    )

    # Run the check
    result = await TenantEnforcementCheck().run(ctx)

    # Verify result
    assert result.passed is expected_pass
    assert result.event.reason_code == expected_reason
    assert result.halt is expect_halt

    # Verify SSH interactions for rented machines
    containers = rented_machine.get("containers", []) if rented_machine else []
    if containers:
        # Should check if pod is running
        assert any("docker ps" in cmd for cmd in ssh_client.commands_called)

        if pod_running:
            # Should check SSH keys
            assert any(
                "authorized_keys" in cmd and "-u 0" in cmd
                for cmd in ssh_client.commands_called
            )

            # If passed and halted, verify score was calculated
            if expected_pass and expect_halt:
                assert score_calculator.called_with is not None
                assert score_calculator.called_with["rented"] is True

                # Verify updates
                assert "rented" in result.updates
                assert result.updates["rented"] is True
                assert "score" in result.updates
                assert "job_score" in result.updates
                assert "success" in result.updates
                assert result.updates["success"] is True

    # Verify updates for not rented case
    if not containers:
        assert "rented" in result.updates
        assert result.updates["rented"] is False
        assert result.updates["ssh_pub_keys"] is None

    # Verify failure cases
    if not expected_pass:
        if expected_reason == Msg.POD_NOT_RUNNING.reason:
            assert "clear_verified_job_info" in result.updates
            assert result.updates["clear_verified_job_info"] is True
            assert "clear_verified_job_reason" in result.updates
            assert result.updates["clear_verified_job_reason"] == ResetVerifiedJobReason.POD_NOT_RUNNING.value
        elif expected_reason == Msg.GPU_OUTSIDE_TENANT.reason:
            # Verify GPU usage details in what_we_saw
            assert "gpu_utilization" in result.event.what_we_saw
            assert "vram_utilization" in result.event.what_we_saw
            assert "process_count" in result.event.what_we_saw


@pytest.mark.asyncio
async def test_tenant_enforcement_skips_pod_not_running_when_backend_says_inactive(context_factory):
    executor_uuid = "executor-123"
    rented_data = build_rented_data(
        executor_uuid,
        {"containers": [{"name": "tenant-123", "pod_id": "pod-1"}], "owner_flag": False},
    )
    backend = DummyBackendClient(active=False)
    services = build_services(
        score_calculator=DummyScoreCalculator(actual_score=1.0, job_score=1.0, warning=""),
        container_cleanup=MockContainerCleanup(),
        backend=backend,
    )
    state = build_state(
        gpu_processes=[],
        gpu_details=[],
        gpu_model="NVIDIA RTX 4090",
        rented_data=rented_data,
    )
    ctx = context_factory(
        services=services,
        config=build_context_config(),
        state=state,
        ssh=DummySSHClient(pod_running=False),
        collateral_deposited=True,
        is_rental_succeed=True,
        contract_version="v1.0.0",
    )

    result = await TenantEnforcementCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.STALE_POD_NOT_RUNNING.reason
    assert backend.called_with == ["pod-1"]
    assert "clear_verified_job_info" not in result.updates
    assert "clear_verified_job_reason" not in result.updates


@pytest.mark.asyncio
async def test_tenant_enforcement_withholds_the_verdict_when_the_backend_state_is_unknown(context_factory):
    executor_uuid = "executor-123"
    rented_data = build_rented_data(
        executor_uuid,
        {"containers": [{"name": "tenant-123", "pod_id": "pod-1"}], "owner_flag": False},
    )
    backend = DummyBackendClient(active=None)
    services = build_services(
        score_calculator=DummyScoreCalculator(actual_score=1.0, job_score=1.0, warning=""),
        container_cleanup=MockContainerCleanup(),
        backend=backend,
    )
    state = build_state(
        gpu_processes=[],
        gpu_details=[],
        gpu_model="NVIDIA RTX 4090",
        rented_data=rented_data,
    )
    ctx = context_factory(
        services=services,
        config=build_context_config(),
        state=state,
        ssh=DummySSHClient(pod_running=False),
        collateral_deposited=True,
        is_rental_succeed=True,
        contract_version="v1.0.0",
    )

    result = await TenantEnforcementCheck().run(ctx)

    # a renter who ended the pod since the batch-start list is not the provider's fault
    assert result.passed is True
    assert backend.called_with == ["pod-1"]
    assert result.updates["default_extra"]["pods_rental_state_unknown"] == ["pod-1"]
    assert "clear_verified_job_info" not in result.updates


def _unknown_rental_ctx(context_factory, backend, redis):
    rented_data = build_rented_data(
        "executor-123", {"containers": [{"name": "tenant-123", "pod_id": "pod-1"}], "owner_flag": False}
    )
    services = build_services(
        score_calculator=DummyScoreCalculator(actual_score=1.0, job_score=1.0, warning=""),
        container_cleanup=MockContainerCleanup(),
        backend=backend,
        redis=redis,
    )
    state = build_state(gpu_processes=[], gpu_details=[], gpu_model="NVIDIA RTX 4090", rented_data=rented_data)
    return context_factory(
        services=services,
        config=build_context_config(),
        state=state,
        ssh=DummySSHClient(pod_running=False),
        collateral_deposited=True,
        is_rental_succeed=True,
        contract_version="v1.0.0",
    )


@pytest.mark.asyncio
async def test_tenant_enforcement_judges_a_down_pod_on_the_third_unanswered_rental_check(context_factory):
    # a long backend outage must not keep a really-down pod unpenalised: two cycles held, the third judged
    backend, redis = DummyBackendClient(active=None), FakeRedis()

    def cycle():
        return TenantEnforcementCheck().run(_unknown_rental_ctx(context_factory, backend, redis))

    assert [(await cycle()).passed for _ in range(2)] == [True, True]
    third = await cycle()

    assert third.event.reason_code == Msg.POD_NOT_RUNNING.reason
    assert third.updates["clear_verified_job_info"] is True
    # the same outage goes on: judged every cycle until an answer, never held again
    assert (await cycle()).event.reason_code == Msg.POD_NOT_RUNNING.reason


@pytest.mark.asyncio
async def test_tenant_enforcement_an_answered_rental_check_resets_the_unanswered_count(context_factory):
    backend, redis = DummyBackendClient(active=None), FakeRedis()

    def cycle():
        return TenantEnforcementCheck().run(_unknown_rental_ctx(context_factory, backend, redis))

    await cycle()
    await cycle()
    backend.active = False  # the stale-pod answer
    assert (await cycle()).event.reason_code == Msg.STALE_POD_NOT_RUNNING.reason
    backend.active = None

    assert (await cycle()).passed is True
    assert redis.store["rental_state_unknown:pod-1"] == "1"


@pytest.mark.asyncio
async def test_tenant_enforcement_judges_now_when_redis_cannot_count(context_factory):
    # a backend and Redis outage together must not hide a down pod forever: without a count, the pod is judged
    result = await TenantEnforcementCheck().run(
        _unknown_rental_ctx(context_factory, DummyBackendClient(active=None), FakeRedis(failing=True))
    )

    assert result.event.reason_code == Msg.POD_NOT_RUNNING.reason


@pytest.mark.asyncio
async def test_tenant_enforcement_a_running_pod_resets_the_unanswered_count(context_factory):
    backend, redis = DummyBackendClient(active=None), FakeRedis()
    ctx = _unknown_rental_ctx(context_factory, backend, redis)

    await TenantEnforcementCheck().run(ctx)
    await TenantEnforcementCheck().run(ctx)
    ctx.ssh.pod_running = True
    assert (await TenantEnforcementCheck().run(ctx)).passed is True
    ctx.ssh.pod_running = False

    assert (await TenantEnforcementCheck().run(ctx)).passed is True  # a fresh unanswered check is held again
    assert redis.store["rental_state_unknown:pod-1"] == "1"


@pytest.mark.asyncio
async def test_tenant_enforcement_a_dry_run_never_changes_the_unanswered_count(context_factory):
    backend, redis = DummyBackendClient(active=None), FakeRedis()
    ctx = _unknown_rental_ctx(context_factory, backend, redis)

    for _ in range(2):
        await TenantEnforcementCheck(recover_stale_pods=False).run(ctx)
    backend.active = True
    await TenantEnforcementCheck(recover_stale_pods=False).run(ctx)

    assert "rental_state_unknown:pod-1" not in redis.store
    backend.active = None
    assert (await TenantEnforcementCheck().run(ctx)).passed is True  # production still has both holds


class DiagnosticsSSHClient:
    """SSH mock returning scripted stdout per docker/shell subcommand for diagnostics tests."""

    def __init__(
        self,
        *,
        inspect_stdout: str = "",
        inspect_stderr: str = "",
        logs_stdout: str = "",
        uptime_stdout: str = "",
        ps_executor_stdout: str = "",
        executor_inspect_stdout: str = "",
        raise_on: str | None = None,
    ):
        self.inspect_stdout = inspect_stdout
        self.inspect_stderr = inspect_stderr
        self.logs_stdout = logs_stdout
        self.uptime_stdout = uptime_stdout
        self.ps_executor_stdout = ps_executor_stdout
        self.executor_inspect_stdout = executor_inspect_stdout
        self.raise_on = raise_on
        self.commands_called: list[str] = []

    async def run(self, command: str):
        self.commands_called.append(command)
        if self.raise_on and self.raise_on in command:
            raise RuntimeError(f"ssh failed on {self.raise_on}")
        result = Mock()
        # Order matters — more specific matches first.
        if command.startswith("uptime"):
            result.stdout = self.uptime_stdout
            result.stderr = ""
        elif "--filter label=com.docker.compose.service=executor" in command:
            result.stdout = self.ps_executor_stdout
            result.stderr = ""
        elif "{{.State.StartedAt}}" in command:
            result.stdout = self.executor_inspect_stdout
            result.stderr = ""
        elif "docker inspect" in command:
            result.stdout = self.inspect_stdout
            result.stderr = self.inspect_stderr
        elif "docker logs" in command:
            result.stdout = self.logs_stdout
            result.stderr = ""
        else:
            result.stdout = ""
            result.stderr = ""
        return result


@pytest.mark.asyncio
async def test_collect_pod_diagnostics_reports_exited_container():
    state = {
        "Status": "exited",
        "ExitCode": 137,
        "OOMKilled": False,
        "Error": "",
        "StartedAt": "2026-04-20T00:00:00Z",
        "FinishedAt": "2026-04-20T00:05:00Z",
    }
    ssh = DiagnosticsSSHClient(
        inspect_stdout=json.dumps(state),
        logs_stdout="traceback line 1\ntraceback line 2\n",
    )

    diagnostics = await _collect_pod_diagnostics(ssh, "pod_abc")

    assert diagnostics["container_status"] == "exited"
    assert diagnostics["container_exit_code"] == 137
    assert diagnostics["container_oom_killed"] is False
    assert diagnostics["container_logs_tail"].endswith("traceback line 2\n")
    assert any("docker inspect" in cmd for cmd in ssh.commands_called)
    assert any("docker logs" in cmd for cmd in ssh.commands_called)
    # Pod-scoped commands all carry the pod name; host-context probes do not.
    pod_scoped = [
        cmd for cmd in ssh.commands_called
        if ("docker inspect" in cmd or "docker logs" in cmd)
        and "{{.State.StartedAt}}" not in cmd
    ]
    assert pod_scoped
    assert all("pod_abc" in cmd for cmd in pod_scoped)


@pytest.mark.asyncio
async def test_collect_pod_diagnostics_handles_removed_container():
    ssh = DiagnosticsSSHClient(
        inspect_stdout="",
        inspect_stderr="Error: No such object: pod_abc",
        logs_stdout="",
    )

    diagnostics = await _collect_pod_diagnostics(ssh, "pod_abc")

    assert diagnostics["container_status"] is None
    assert "Error: No such object: pod_abc" in diagnostics["diagnostics_capture_error"]
    assert diagnostics["container_logs_tail"] is None


@pytest.mark.asyncio
async def test_collect_pod_diagnostics_never_raises_on_ssh_failure():
    ssh = DiagnosticsSSHClient(raise_on="docker inspect")

    diagnostics = await _collect_pod_diagnostics(ssh, "pod_abc")

    assert diagnostics["diagnostics_capture_error"] is not None
    assert "ssh failed" in diagnostics["diagnostics_capture_error"]


@pytest.mark.asyncio
async def test_collect_pod_diagnostics_is_bounded_by_its_timeout():
    # lio#1827 review: a stalled docker read must not hold the stop-refused failure past the task deadline
    async def hang(*args, **kwargs):
        await asyncio.sleep(3600)

    ssh = Mock(run=hang)

    diagnostics = await _collect_pod_diagnostics(ssh, "pod_abc", timeout=0.01)

    assert diagnostics["diagnostics_capture_error"] == "diagnostics timed out"


@pytest.mark.asyncio
async def test_collect_pod_diagnostics_includes_host_context_when_available():
    state = {
        "Status": "exited",
        "ExitCode": 137,
        "OOMKilled": True,
        "Error": "",
        "StartedAt": "2026-04-20T00:00:00Z",
        "FinishedAt": "2026-04-20T00:05:00Z",
    }
    ssh = DiagnosticsSSHClient(
        inspect_stdout=json.dumps(state),
        logs_stdout="oom line\n",
        uptime_stdout="2026-04-03 10:07:20\n",
        ps_executor_stdout="executor-executor-1\n",
        executor_inspect_stdout="2026-04-20T03:23:42.619916501Z\n",
    )

    diagnostics = await _collect_pod_diagnostics(ssh, "pod_abc")

    assert diagnostics["container_oom_killed"] is True
    assert diagnostics["container_host_context"] == {
        "host_boot_time": "2026-04-03 10:07:20",
        "executor_container_name": "executor-executor-1",
        "executor_container_started_at": "2026-04-20T03:23:42.619916501Z",
    }
    # executor lookup uses the compose label filter — not a hardcoded container name
    assert any(
        "--filter label=com.docker.compose.service=executor" in cmd
        for cmd in ssh.commands_called
    )


@pytest.mark.asyncio
async def test_collect_host_context_handles_missing_executor_container():
    ssh = DiagnosticsSSHClient(
        uptime_stdout="2026-04-03 10:07:20\n",
        ps_executor_stdout="",  # no container matching the compose label
    )

    host_context = await _collect_host_context(ssh)

    assert host_context == {"host_boot_time": "2026-04-03 10:07:20"}
    assert "executor_container_name" not in host_context
    assert "executor_container_started_at" not in host_context


@pytest.mark.asyncio
async def test_collect_host_context_never_raises_on_ssh_failure():
    ssh = DiagnosticsSSHClient(raise_on="uptime")

    host_context = await _collect_host_context(ssh)

    assert "host_boot_error" in host_context
    assert "ssh failed" in host_context["host_boot_error"]


@pytest.mark.parametrize(
    "transport_exc",
    [
        pytest.param(asyncssh.ConnectionLost("conntrack flush"), id="connection_lost"),
        pytest.param(asyncssh.DisconnectError(11, "network", "en"), id="disconnect_error"),
        pytest.param(OSError("socket closed"), id="os_error"),
    ],
)
@pytest.mark.asyncio
async def test_tenant_enforcement_emits_transport_unreachable_on_ssh_failure(
    transport_exc, context_factory
):
    """DAH-2055: SSH transport failure mid-cycle must NOT trigger immediate
    executor flip. Validator emits EXECUTOR_TRANSPORT_UNREACHABLE without any
    clear_verified_job_* in updates so compute-app's reset_verified_job is bypassed."""
    executor_uuid = "executor-123"
    rented_machine = {
        "containers": [{"name": "tenant-123", "pod_id": "pod-1"}],
        "owner_flag": False,
    }
    rented_data = build_rented_data(executor_uuid, rented_machine)

    ssh_client = DummySSHClient(raise_on_run=transport_exc)
    services = build_services(
        score_calculator=DummyScoreCalculator(actual_score=1.0, job_score=1.0, warning=""),
        container_cleanup=MockContainerCleanup(),
        backend=DummyBackendClient(active=True),
    )
    state = build_state(
        gpu_processes=[],
        gpu_details=[],
        gpu_model="NVIDIA RTX 4090",
        rented_data=rented_data,
    )
    ctx = context_factory(
        services=services,
        config=build_context_config(),
        state=state,
        ssh=ssh_client,
        collateral_deposited=True,
        is_rental_succeed=True,
        contract_version="v1.0.0",
    )

    result = await TenantEnforcementCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.EXECUTOR_TRANSPORT_UNREACHABLE.reason
    assert "clear_verified_job_info" not in result.updates
    assert "clear_verified_job_reason" not in result.updates


@pytest.mark.asyncio
async def test_tenant_enforcement_reports_unreachable_when_a_pod_read_outlasts_the_cycle(context_factory, monkeypatch):
    """5bad1f5a 4 Nov: a pod read hung on a dead shell until the cycle was dropped without a result. The read is
    bounded by the cycle time budget and the hang ends as the transport-unreachable result."""
    from neurons.validators.src.services.task.checks import rented_machine

    class HangingSSH(DummySSHClient):
        async def run(self, command: str):
            await asyncio.sleep(3600)

    monkeypatch.setattr(rented_machine, "task_seconds_left", lambda _ctx: rented_machine.POD_READ_RESERVE_SECONDS)
    rented_data = build_rented_data(
        "executor-123", {"containers": [{"name": "tenant-123", "pod_id": "pod-1"}], "owner_flag": False}
    )
    ctx = context_factory(
        services=build_services(
            score_calculator=DummyScoreCalculator(actual_score=1.0, job_score=1.0, warning=""),
            container_cleanup=MockContainerCleanup(),
            backend=DummyBackendClient(active=True),
        ),
        config=build_context_config(),
        state=build_state(gpu_processes=[], gpu_details=[], gpu_model="NVIDIA RTX 4090", rented_data=rented_data),
        ssh=HangingSSH(),
        collateral_deposited=True,
        is_rental_succeed=True,
        contract_version="v1.0.0",
    )

    result = await asyncio.wait_for(TenantEnforcementCheck().run(ctx), 5)

    assert result.passed is False
    assert result.event.reason_code == Msg.EXECUTOR_TRANSPORT_UNREACHABLE.reason


@pytest.mark.asyncio
async def test_tenant_enforcement_keeps_pod_not_running_for_non_transport_errors(context_factory):
    """Regression guard: a non-transport exception (e.g. RuntimeError from a
    misbehaving docker daemon) must keep its legacy POD_NOT_RUNNING classification
    and continue to set clear_verified_job_reason."""
    executor_uuid = "executor-123"
    rented_machine = {
        "containers": [{"name": "tenant-123", "pod_id": "pod-1"}],
        "owner_flag": False,
    }
    rented_data = build_rented_data(executor_uuid, rented_machine)

    ssh_client = DummySSHClient(should_raise=True)  # raises RuntimeError
    services = build_services(
        score_calculator=DummyScoreCalculator(actual_score=1.0, job_score=1.0, warning=""),
        container_cleanup=MockContainerCleanup(),
        backend=DummyBackendClient(active=True),
    )
    state = build_state(
        gpu_processes=[],
        gpu_details=[],
        gpu_model="NVIDIA RTX 4090",
        rented_data=rented_data,
    )
    ctx = context_factory(
        services=services,
        config=build_context_config(),
        state=state,
        ssh=ssh_client,
        collateral_deposited=True,
        is_rental_succeed=True,
        contract_version="v1.0.0",
    )

    result = await TenantEnforcementCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.POD_NOT_RUNNING.reason
    assert result.updates.get("clear_verified_job_info") is True
    assert result.updates.get("clear_verified_job_reason") == ResetVerifiedJobReason.POD_NOT_RUNNING.value


@pytest.mark.asyncio
async def test_tenant_enforcement_marks_filler_only_as_not_rented_without_halting(context_factory):
    executor_uuid = "executor-123"
    filler_container = "filler_5703f4c9-c2f4-4fae-a652-3dee4753030a"
    ssh_client = DummySSHClient()
    score_calculator = DummyScoreCalculator(actual_score=0.7, job_score=0.7)
    services = build_services(
        score_calculator=score_calculator,
        container_cleanup=MockContainerCleanup(),
    )
    state = build_state(
        gpu_processes=[{"container_name": filler_container, "pid": 1234}],
        gpu_details=[{"gpu_utilization": 95, "memory_utilization": 80}],
        rented_data=RentedExecutorsResponse(
            executors={},
            filler_containers_by_executor={executor_uuid: filler_container},
        ),
    )
    ctx = context_factory(
        services=services,
        state=state,
        ssh=ssh_client,
        collateral_deposited=True,
        contract_version="v1.0.0",
    )

    result = await TenantEnforcementCheck().run(ctx)

    assert result.passed is True
    assert result.halt is False
    assert result.event.reason_code == Msg.NOT_RENTED.reason
    assert result.event.what_we_saw["filler_container"] == filler_container
    assert result.updates["rented"] is False
    assert "success" not in result.updates
    assert score_calculator.called_with is None
    assert not any("docker ps" in cmd for cmd in ssh_client.commands_called)


@pytest.mark.asyncio
async def test_tenant_enforcement_keeps_filler_only_not_rented_for_unmapped_process(context_factory):
    executor_uuid = "executor-123"
    filler_container = "filler_active"
    services = build_services(
        score_calculator=DummyScoreCalculator(),
        container_cleanup=MockContainerCleanup(),
    )
    state = build_state(
        gpu_processes=[{"container_name": "filler_other", "pid": 1234}],
        gpu_details=[{"gpu_utilization": 95, "memory_utilization": 80}],
        rented_data=build_filler_data(executor_uuid, filler_container),
    )
    ctx = context_factory(
        services=services,
        state=state,
        ssh=DummySSHClient(),
        collateral_deposited=True,
        contract_version="v1.0.0",
    )

    result = await TenantEnforcementCheck().run(ctx)

    assert result.passed is True
    assert result.halt is False
    assert result.event.reason_code == Msg.NOT_RENTED.reason
    assert result.event.what_we_saw["filler_container"] == filler_container


@pytest.mark.asyncio
async def test_tenant_enforcement_allows_mapped_filler_with_customer_rental(context_factory):
    executor_uuid = "executor-123"
    pod_container = "tenant-123"
    filler_container = "filler_active"
    rented_data = build_rented_data(
        executor_uuid,
        {"containers": [{"name": pod_container, "pod_id": "pod-1"}], "owner_flag": False},
    )
    rented_data.filler_containers_by_executor[executor_uuid] = filler_container
    score_calculator = DummyScoreCalculator(actual_score=1.0, job_score=1.0)
    services = build_services(
        score_calculator=score_calculator,
        container_cleanup=MockContainerCleanup(),
    )
    state = build_state(
        gpu_processes=[{"container_name": filler_container, "pid": 1234}],
        gpu_details=[{"gpu_utilization": 95, "memory_utilization": 80}],
        rented_data=rented_data,
    )
    ctx = context_factory(
        services=services,
        state=state,
        ssh=DummySSHClient(pod_running=True, ssh_keys=["ssh-rsa AAA..."]),
        collateral_deposited=True,
        contract_version="v1.0.0",
    )

    result = await TenantEnforcementCheck().run(ctx)

    assert result.passed is True
    assert result.halt is True
    assert result.event.reason_code == Msg.ALREADY_RENTED.reason
    assert result.updates["rented"] is True
    assert score_calculator.called_with["rented"] is True


@pytest.mark.asyncio
async def test_tenant_enforcement_allows_every_filler_bundle_beside_customer_pod(context_factory):
    # DAH-2472: a partially rented GPU-split node keeps one filler per free VRAM bundle. Every
    # bundle's GPU process must count as in-tenant, not only the first filler's.
    executor_uuid = "executor-123"
    pod_container = "tenant-123"
    fillers = ["filler_bundle_a", "filler_bundle_b"]
    rented_data = build_rented_data(
        executor_uuid,
        {"containers": [{"name": pod_container, "pod_id": "pod-1"}], "owner_flag": False},
    )
    rented_data.all_filler_containers_by_executor[executor_uuid] = fillers
    score_calculator = DummyScoreCalculator(actual_score=1.0, job_score=1.0)
    services = build_services(
        score_calculator=score_calculator,
        container_cleanup=MockContainerCleanup(),
    )
    state = build_state(
        gpu_processes=[
            {"container_name": pod_container, "pid": 1001},
            {"container_name": fillers[0], "pid": 1002},
            {"container_name": fillers[1], "pid": 1003},
        ],
        gpu_details=[{"gpu_utilization": 95, "memory_utilization": 80}],
        rented_data=rented_data,
    )
    ctx = context_factory(
        services=services,
        state=state,
        ssh=DummySSHClient(pod_running=True, ssh_keys=["ssh-rsa AAA..."]),
        collateral_deposited=True,
        contract_version="v1.0.0",
    )

    result = await TenantEnforcementCheck().run(ctx)

    assert result.passed is True, result.event
    assert result.event.reason_code == Msg.ALREADY_RENTED.reason
    assert result.updates["rented"] is True


def build_recovery_context(
    context_factory,
    ssh: DummySSHClient,
    docker: AsyncMock,
    *,
    private_key: str | None = "ssh-key",
    backend: DummyBackendClient | None = None,
    redis: Mock | None = None,
) -> Context:
    rented_data = build_rented_data(
        "executor-123",
        {"containers": [{"name": "pod_pod-1", "pod_id": "pod-1"}], "owner_flag": False},
    )
    services = build_services(
        score_calculator=DummyScoreCalculator(actual_score=1.0, job_score=1.0, warning=""),
        container_cleanup=MockContainerCleanup(),
        backend=backend if backend is not None else DummyBackendClient(active=True),
        pod_recovery=docker,
        **({"redis": redis} if redis is not None else {}),
    )
    return context_factory(
        services=services,
        config=build_context_config(),
        state=build_state(
            gpu_processes=[],
            gpu_details=[],
            gpu_model="NVIDIA RTX 4090",
            rented_data=rented_data,
        ),
        ssh=ssh,
        executor_ssh_private_key=private_key,
        collateral_deposited=True,
        is_rental_succeed=True,
        contract_version="v1.0.0",
    )


@pytest.mark.asyncio
async def test_tenant_enforcement_passes_when_pod_recovered_from_stale_mount(context_factory):
    ssh = DummySSHClient(pod_running=False, ssh_keys=["ssh-rsa recovered"])
    docker = AsyncMock()

    async def bring_pod_back_up(**kwargs):
        ssh.pod_running = True
        return True

    docker.recover_pod_after_stale_vloopback_mount.side_effect = bring_pod_back_up
    ctx = build_recovery_context(context_factory, ssh, docker)

    result = await TenantEnforcementCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.ALREADY_RENTED.reason
    assert "clear_verified_job_info" not in result.updates
    assert result.updates["ssh_pub_keys"] == ["ssh-rsa recovered"]
    assert result.updates["default_extra"]["recovered_pods"] == ["pod_pod-1"]
    assert docker.recover_pod_after_stale_vloopback_mount.await_args.kwargs["pod_id"] == "pod-1"


@pytest.mark.asyncio
async def test_tenant_enforcement_hands_recovery_the_backend_plaintext_path(context_factory):
    # DAH-2545: an encrypted pod can only be revived at the path the backend recorded at create
    # time, so whatever rental-active returns has to reach the recovery untouched
    ssh = DummySSHClient(pod_running=False, ssh_keys=["ssh-rsa recovered"])
    docker = AsyncMock()

    async def bring_pod_back_up(**kwargs):
        ssh.pod_running = True
        return True

    docker.recover_pod_after_stale_vloopback_mount.side_effect = bring_pod_back_up
    backend = DummyBackendClient(active=True, local_volume_path="/workspace")
    ctx = build_recovery_context(context_factory, ssh, docker, backend=backend)

    result = await TenantEnforcementCheck().run(ctx)

    assert result.passed is True
    recovery_kwargs = docker.recover_pod_after_stale_vloopback_mount.await_args.kwargs
    assert recovery_kwargs["local_volume_path"] == "/workspace"


@pytest.mark.asyncio
async def test_tenant_enforcement_reports_recovery_to_the_backend(context_factory):
    # the renter has to learn their pod went down because the provider's host restarted
    ssh = DummySSHClient(pod_running=False, ssh_keys=["ssh-rsa recovered"])
    docker = AsyncMock()

    async def bring_pod_back_up(**kwargs):
        ssh.pod_running = True
        return True

    docker.recover_pod_after_stale_vloopback_mount.side_effect = bring_pod_back_up
    backend = DummyBackendClient(active=True)
    ctx = build_recovery_context(context_factory, ssh, docker, backend=backend)

    result = await TenantEnforcementCheck().run(ctx)

    assert result.passed is True
    assert backend.host_reboot_recoveries == [("pod-1", ssh.container_finished_at)]


@pytest.mark.asyncio
async def test_tenant_enforcement_survives_a_failed_recovery_report(context_factory):
    # the rental is already saved by then; a backend that is down must not undo that
    ssh = DummySSHClient(pod_running=False, ssh_keys=["ssh-rsa recovered"])
    docker = AsyncMock()

    async def bring_pod_back_up(**kwargs):
        ssh.pod_running = True
        return True

    docker.recover_pod_after_stale_vloopback_mount.side_effect = bring_pod_back_up
    backend = DummyBackendClient(active=True)
    backend.report_pod_host_reboot_recovered = AsyncMock(side_effect=RuntimeError("backend down"))
    ctx = build_recovery_context(context_factory, ssh, docker, backend=backend)

    result = await TenantEnforcementCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.ALREADY_RENTED.reason


@pytest.mark.asyncio
async def test_tenant_enforcement_penalises_when_recovery_declines(context_factory):
    docker = AsyncMock()
    docker.recover_pod_after_stale_vloopback_mount.return_value = False
    ctx = build_recovery_context(context_factory, DummySSHClient(pod_running=False), docker)

    result = await TenantEnforcementCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.POD_NOT_RUNNING.reason
    assert result.updates["clear_verified_job_reason"] == ResetVerifiedJobReason.POD_NOT_RUNNING.value


@pytest.mark.asyncio
async def test_tenant_enforcement_penalises_when_pod_stays_down_after_recovery(context_factory):
    docker = AsyncMock()
    docker.recover_pod_after_stale_vloopback_mount.return_value = True
    ctx = build_recovery_context(context_factory, DummySSHClient(pod_running=False), docker)

    result = await TenantEnforcementCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.POD_NOT_RUNNING.reason


@pytest.mark.asyncio
async def test_tenant_enforcement_penalises_when_recovery_raises(context_factory):
    docker = AsyncMock()
    docker.recover_pod_after_stale_vloopback_mount.side_effect = RuntimeError("ssh died")
    ctx = build_recovery_context(context_factory, DummySSHClient(pod_running=False), docker)

    result = await TenantEnforcementCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.POD_NOT_RUNNING.reason


class MissingPodSSHClient(DummySSHClient):
    """The pod's container is gone: `docker inspect` answers No such object."""

    async def run(self, command: str):
        if "docker inspect" in command and not self.pod_running:
            self.commands_called.append(command)
            return Mock(stdout="", stderr="Error: No such object: pod_pod-1", exit_status=1)
        return await super().run(command)


@pytest.mark.parametrize("create_in_flight", [False, True])
@pytest.mark.asyncio
async def test_tenant_enforcement_renames_back_a_stranded_parked_container(
    context_factory, create_in_flight
):
    # 22 Oct: a failed reboot left only pod_<id>__prev; the cycle gives it its name back, but never
    # while the pod's create may still be between its park and `docker run`
    ssh = MissingPodSSHClient(pod_running=False, ssh_keys=["ssh-rsa recovered"])
    docker = AsyncMock()
    docker.recover_pod_after_stale_vloopback_mount.return_value = False

    async def rename_back_and_start(**kwargs):
        ssh.pod_running = True
        return True

    docker.recover_parked_pod_container.side_effect = rename_back_and_start
    lock = Mock(acquire=AsyncMock(return_value=True), release=AsyncMock())
    redis = Mock(
        backend_create_inflight=AsyncMock(return_value=False),
        renting_in_progress=AsyncMock(return_value=create_in_flight),
        executor_create_lock=Mock(return_value=lock),
    )
    ctx = build_recovery_context(
        context_factory,
        ssh,
        docker,
        backend=DummyBackendClient(active=True, local_volume_path="/workspace"),
        redis=redis,
    )

    result = await TenantEnforcementCheck().run(ctx)

    if create_in_flight:
        assert result.passed is False
        assert result.event.reason_code == Msg.POD_NOT_RUNNING.reason
        docker.recover_parked_pod_container.assert_not_awaited()
    else:
        assert result.passed is True, result.event
        assert result.updates["default_extra"]["recovered_pods"] == ["pod_pod-1"]
        kwargs = docker.recover_parked_pod_container.await_args.kwargs
        assert (kwargs["container_name"], kwargs["local_volume_path"]) == ("pod_pod-1", "/workspace")
        lock.release.assert_awaited_once()


def _parked_recovery_ctx(context_factory, docker):
    ssh = MissingPodSSHClient(pod_running=False, ssh_keys=["ssh-rsa recovered"])
    docker.recover_pod_after_stale_vloopback_mount.return_value = False
    lock = Mock(acquire=AsyncMock(return_value=True), release=AsyncMock())
    redis = Mock(
        backend_create_inflight=AsyncMock(return_value=False),
        renting_in_progress=AsyncMock(return_value=False),
        executor_create_lock=Mock(return_value=lock),
    )
    ctx = build_recovery_context(
        context_factory,
        ssh,
        docker,
        backend=DummyBackendClient(active=True, local_volume_path="/workspace"),
        redis=redis,
    )
    return ctx, lock


@pytest.mark.asyncio
async def test_tenant_enforcement_never_recovers_under_the_connectors_running_create(context_factory):
    # creates run in the connector process; the pending mark expires at 30 min, so a reboot still
    # running then is seen in the connector's shared in-flight record
    docker = AsyncMock()
    ctx, lock = _parked_recovery_ctx(context_factory, docker)
    ctx.services.redis.backend_create_inflight.return_value = True

    result = await TenantEnforcementCheck().run(ctx)

    assert result.event.reason_code == Msg.POD_NOT_RUNNING.reason
    docker.recover_parked_pod_container.assert_not_awaited()
    lock.acquire.assert_not_awaited()


@pytest.mark.asyncio
async def test_tenant_enforcement_a_recovery_past_its_deadline_fails_inside_the_lock(
    context_factory, monkeypatch
):
    # bounded well inside the create lock's TTL, and a missed deadline is a failed recovery, not a
    # dead SSH transport
    docker = AsyncMock()

    async def stall(**kwargs):
        await asyncio.sleep(10)

    docker.recover_parked_pod_container.side_effect = stall
    ctx, lock = _parked_recovery_ctx(context_factory, docker)
    # the real deadline plus its cleanup (a 30 s stop and a 30 s rename) ends before the lock's TTL
    assert rented_machine_module._RECOVERY_DEADLINE_SEC + 60 < rented_machine_module.EXECUTOR_CREATE_LOCK_TIMEOUT
    monkeypatch.setattr(rented_machine_module, "_RECOVERY_DEADLINE_SEC", 0.01)

    result = await TenantEnforcementCheck().run(ctx)

    assert result.event.reason_code == Msg.POD_NOT_RUNNING.reason
    lock.release.assert_awaited_once()


@pytest.mark.parametrize("budget_left", [False, True])
@pytest.mark.asyncio
async def test_tenant_enforcement_parked_recovery_comes_out_of_the_cycle_budget(
    context_factory, monkeypatch, budget_left
):
    # lio#1827 review 4490776354: a recovery outliving the task drops its verdict
    docker = AsyncMock()
    recovery_deadlines = []
    docker.recover_parked_pod_container.side_effect = lambda **_kw: recovery_deadlines.append(deadlines[-1]) or True
    ctx, _lock = _parked_recovery_ctx(context_factory, docker)
    floor = rented_machine_module.POD_READ_RESERVE_SECONDS + rented_machine_module._PARKED_RECOVERY_CLEANUP_RESERVE_SEC
    monkeypatch.setattr(rented_machine_module, "task_seconds_left", lambda _ctx: floor + (60 if budget_left else 0))
    deadlines = []
    real_timeout = asyncio.timeout
    monkeypatch.setattr(rented_machine_module.asyncio, "timeout", lambda d: deadlines.append(d) or real_timeout(d))

    await TenantEnforcementCheck().run(ctx)

    assert recovery_deadlines == ([60] if budget_left else [])


@pytest.mark.parametrize(
    "create_in_flight, local_volume_path, recover_enabled, expected_restored",
    [
        pytest.param(False, "/workspace", True, True, id="restored"),
        pytest.param(True, "/workspace", True, False, id="create_in_flight"),
        # restore_volume_mount_of_running_container stops a pod it has no path for
        pytest.param(False, None, True, True, id="plaintext_path_unknown"),
        pytest.param(False, "/workspace", False, False, id="dry_run_leaves_the_live_pod_alone"),
    ],
)
@pytest.mark.asyncio
async def test_tenant_enforcement_remounts_a_running_pods_missing_encrypted_volume(
    context_factory, create_in_flight, local_volume_path, recover_enabled, expected_restored
):
    # ticket-0729: the provider started the pod by hand, so it ran without its gocryptfs workspace
    ssh = DummySSHClient(pod_running=True, ssh_keys=["ssh-rsa renter"])
    docker = AsyncMock()
    docker.encrypted_volume_mount_missing.return_value = True
    lock = Mock(acquire=AsyncMock(return_value=True), release=AsyncMock())
    redis = Mock(
        backend_create_inflight=AsyncMock(return_value=False),
        renting_in_progress=AsyncMock(return_value=create_in_flight),
        executor_create_lock=Mock(return_value=lock),
    )
    ctx = build_recovery_context(
        context_factory,
        ssh,
        docker,
        backend=DummyBackendClient(active=True, local_volume_path=local_volume_path),
        redis=redis,
    )

    result = await TenantEnforcementCheck(recover_stale_pods=recover_enabled).run(ctx)

    assert result.passed is True, result.event
    if recover_enabled:  # the probe matches the rental's own mount, at its path
        assert docker.encrypted_volume_mount_missing.await_args.args[1:] == ("pod_pod-1", local_volume_path)
    if expected_restored:
        kwargs = docker.restore_volume_mount_of_running_container.await_args.kwargs
        assert (kwargs["container_name"], kwargs["local_volume_path"]) == ("pod_pod-1", local_volume_path)
        lock.release.assert_awaited_once()
    else:
        docker.restore_volume_mount_of_running_container.assert_not_awaited()


@pytest.mark.parametrize("probe_fits", [False, True])
@pytest.mark.asyncio
async def test_tenant_enforcement_mount_probe_comes_out_of_the_cycle_budget(context_factory, monkeypatch, probe_fits):
    # review 5785846085: the probe must not eat the result's reserve, or the task is dropped without a verdict
    from neurons.validators.src.services.task.checks import rented_machine

    floor = rented_machine.POD_READ_RESERVE_SECONDS + rented_machine._MOUNT_RESTORE_CLEANUP_RESERVE_SEC
    left = floor + rented_machine._MOUNT_PROBE_SECONDS + (1 if probe_fits else -1)
    monkeypatch.setattr(rented_machine, "task_seconds_left", lambda _ctx: left)
    ssh = DummySSHClient(pod_running=True, ssh_keys=["ssh-rsa renter"])
    docker = AsyncMock()
    docker.encrypted_volume_mount_missing.return_value = False
    ctx = build_recovery_context(context_factory, ssh, docker, backend=DummyBackendClient(active=True))

    result = await TenantEnforcementCheck().run(ctx)

    assert result.passed is True, result.event
    assert docker.encrypted_volume_mount_missing.await_count == (1 if probe_fits else 0)


@pytest.mark.asyncio
async def test_tenant_enforcement_defers_a_pod_its_own_restore_stopped_to_the_next_cycle(context_factory):
    # F-17: a cut restore's cleanup already spent the reserve; recovering the pod it just stopped in the same cycle
    # (a fresh 240 s deadline) would outlive the task, so the result would be dropped
    ssh = DummySSHClient(pod_running=True, ssh_keys=["ssh-rsa renter"])
    docker = AsyncMock()
    docker.encrypted_volume_mount_missing.return_value = True

    async def cut_restore_then_stop(**kwargs):
        ssh.pod_running = False
        return False

    docker.restore_volume_mount_of_running_container.side_effect = cut_restore_then_stop
    lock = Mock(acquire=AsyncMock(return_value=True), release=AsyncMock())
    ctx = build_recovery_context(
        context_factory,
        ssh,
        docker,
        backend=DummyBackendClient(active=True, local_volume_path="/workspace"),
        redis=_recovery_redis(lock, {"stopped_by_recovery:pod-1": "stopped"}),
    )

    result = await TenantEnforcementCheck().run(ctx)

    assert result.passed, result.event
    docker.recover_parked_pod_container.assert_not_awaited()
    assert result.updates["default_extra"]["pods_stopped_by_validator"] == ["pod-1"]
    assert _pod_states(result)["pod-1"] == "exited"


@pytest.mark.parametrize("stop_outcome", ["failed", "unknown"])
@pytest.mark.asyncio
async def test_tenant_enforcement_defers_only_a_confirmed_stop_of_an_unmounted_pod(context_factory, stop_outcome):
    # lio#1827 review: docker refusing the stop leaves the pod running without its mount, so the host fails the check;
    # a stop it could not confirm is judged next cycle, never counted as the validator's
    ssh = DummySSHClient(pod_running=True, ssh_keys=["ssh-rsa renter"])
    docker = AsyncMock()
    docker.encrypted_volume_mount_missing.return_value = True
    docker.restore_volume_mount_of_running_container.return_value = False
    docker.unmounted_stop_outcome.return_value = stop_outcome
    lock = Mock(acquire=AsyncMock(return_value=True), release=AsyncMock())
    ctx = build_recovery_context(
        context_factory,
        ssh,
        docker,
        backend=DummyBackendClient(active=True, local_volume_path="/workspace"),
        redis=_recovery_redis(lock, {}),
    )

    result = await TenantEnforcementCheck().run(ctx)

    docker.recover_parked_pod_container.assert_not_awaited()
    assert "pods_stopped_by_validator" not in result.updates["default_extra"]
    if stop_outcome == "failed":
        assert not result.passed
        assert result.event.reason_code == Msg.POD_NOT_RUNNING.reason
        assert result.updates["clear_verified_job_info"] is True
        assert result.updates["default_extra"]["pods_stop_refused"] == ["pod-1"]
    else:
        assert result.passed, result.event
        assert _pod_states(result)["pod-1"] == "unknown"


@pytest.mark.asyncio
@pytest.mark.parametrize("running_under_lock", [True, False])
async def test_tenant_enforcement_forgets_a_pending_recovery_stop_of_a_running_pod(context_factory, running_under_lock):
    # lio#1827 review: with no mark in Redis yet, a pending stop-mark retry of a pod running again is still dropped,
    # but only from a read under the create lock: a recovery that stopped the pod after the cycle's earlier read
    # keeps its retry and park record
    ssh = DummySSHClient(pod_running=True, ssh_keys=["ssh-rsa renter"])
    lock = Mock(acquire=AsyncMock(return_value=True), release=AsyncMock())
    docker = AsyncMock()
    docker.encrypted_volume_mount_missing.return_value = False
    docker.recovery_stop_pending.return_value = True
    redis = _recovery_redis(lock, {})
    if not running_under_lock:
        lock.acquire.side_effect = lambda *a, **k: setattr(ssh, "pod_running", False) or True

    await TenantEnforcementCheck().run(build_recovery_context(context_factory, ssh, docker, redis=redis))

    lock.acquire.assert_awaited()
    if running_under_lock:
        docker.forget_recovery_stop.assert_awaited_once_with("pod-1")
    else:
        docker.forget_recovery_stop.assert_not_awaited()


@pytest.mark.asyncio
async def test_tenant_enforcement_takes_no_lock_for_a_running_pod_without_a_recovery_stop(context_factory):
    ssh = DummySSHClient(pod_running=True, ssh_keys=["ssh-rsa renter"])
    lock = Mock(acquire=AsyncMock(return_value=True), release=AsyncMock())
    docker = AsyncMock()
    docker.encrypted_volume_mount_missing.return_value = False
    docker.recovery_stop_pending.return_value = False
    redis = _recovery_redis(lock, {})

    assert (await TenantEnforcementCheck().run(build_recovery_context(context_factory, ssh, docker, redis=redis))).passed

    lock.acquire.assert_not_awaited()
    docker.forget_recovery_stop.assert_not_awaited()


def _recovery_redis(lock, marks: dict[str, str]) -> Mock:
    return Mock(
        backend_create_inflight=AsyncMock(return_value=False),
        renting_in_progress=AsyncMock(return_value=False),
        executor_create_lock=Mock(return_value=lock),
        get=AsyncMock(side_effect=lambda key: marks.get(key)),
        set=AsyncMock(side_effect=lambda key, value, ex=None: marks.__setitem__(key, value)),
        delete=AsyncMock(side_effect=lambda key: marks.pop(key, None)),
    )


@pytest.mark.parametrize("stopped_before_lock", [False, True])
@pytest.mark.asyncio
async def test_tenant_enforcement_clears_a_leftover_recovery_mark_on_a_running_pod(
    context_factory, stopped_before_lock
):
    # a mark whose delete never landed (Redis down past every retry, then a validator restart) is cleared by the first
    # cycle that sees the pod running once Redis answers, so a later provider stop is not restarted
    ssh = DummySSHClient(pod_running=True, ssh_keys=["ssh-rsa renter"])
    marks = {"stopped_by_recovery:pod-1": "1"}
    lock = Mock(acquire=AsyncMock(return_value=True), release=AsyncMock())
    docker = AsyncMock()
    docker.encrypted_volume_mount_missing.return_value = False
    redis = _recovery_redis(lock, marks)

    def cycle():
        return TenantEnforcementCheck().run(build_recovery_context(context_factory, ssh, docker, redis=redis))

    redis.delete.side_effect = ConnectionError("Redis down")
    assert (await cycle()).passed  # a failed delete is harmless
    assert marks

    redis.delete.side_effect = lambda key: marks.pop(key, None)
    lock.acquire.return_value = False  # a recovery holds the create lock between its mark and its stop
    assert (await cycle()).passed
    assert marks

    async def acquire(*args, **kwargs):  # a competing recovery marks and stops the pod, then releases the lock
        if stopped_before_lock:
            ssh.pod_running = False
            marks["stopped_by_recovery:pod-1"] = "1"
        return True

    lock.acquire.side_effect = acquire
    assert (await cycle()).passed
    # only a pod still running under the lock loses its mark
    assert bool(marks) is stopped_before_lock
    lock.release.assert_awaited()
    if stopped_before_lock:
        return
    lock.acquire.side_effect = None

    ssh.pod_running = False  # the provider stops the pod
    result = await cycle()
    assert result.event.reason_code == Msg.POD_NOT_RUNNING.reason
    docker.recover_parked_pod_container.assert_not_awaited()


@pytest.mark.parametrize("stopped_by_us", [True, False])
@pytest.mark.asyncio
async def test_tenant_enforcement_starts_next_cycle_a_pod_it_stopped_while_the_backend_failed(
    context_factory, stopped_by_us
):
    # a transient backend failure (no plaintext path) stops the unmounted pod; that cycle reports it
    # down and the next one starts it with its mount. A pod stopped by anyone else is never started.
    ssh = DummySSHClient(pod_running=stopped_by_us, ssh_keys=["ssh-rsa renter"])
    marks: dict[str, str] = {}
    lock = Mock(acquire=AsyncMock(return_value=True), release=AsyncMock())
    docker = AsyncMock()
    docker.recover_pod_after_stale_vloopback_mount.return_value = False
    docker.encrypted_volume_mount_missing.return_value = True

    async def restore(**kwargs):  # as the real one: no path -> marked and stopped
        if kwargs["local_volume_path"]:
            return True
        marks[f"stopped_by_recovery:{kwargs['pod_id']}"] = "stopped"
        ssh.pod_running = False
        return False

    async def start_marked(**kwargs):  # as the real one: starts a marked pod once the path is known
        if not (kwargs["stopped_by_recovery"] and kwargs["local_volume_path"]):
            return False
        marks.clear()
        ssh.pod_running = True
        docker.encrypted_volume_mount_missing.return_value = False
        return True

    docker.restore_volume_mount_of_running_container.side_effect = restore
    docker.recover_parked_pod_container.side_effect = start_marked
    backend = DummyBackendClient(active=None)  # the backend client answers None on a failed request

    def cycle():
        ctx = build_recovery_context(context_factory, ssh, docker, backend=backend, redis=_recovery_redis(lock, marks))
        return TenantEnforcementCheck().run(ctx)

    first = await cycle()
    assert _pod_states(first)["pod-1"] != "running"
    assert not ssh.pod_running
    if stopped_by_us:
        # reported down through its pod state only: a renter breaking the mount from inside neither fails
        # the provider's validation nor gets the host reset
        assert first.passed and first.halt, first.event
        assert first.event.reason_code == Msg.ALREADY_RENTED.reason
        assert not any(key.startswith("clear_verified_job") for key in first.updates)
        assert first.updates["default_extra"]["pods_stopped_by_validator"] == ["pod-1"]
    else:
        # the backend did not answer whether the rental is still open, so the verdict waits for the next cycle
        assert not any(key.startswith("clear_verified_job") for key in first.updates)
        assert first.updates["default_extra"]["pods_rental_state_unknown"] == ["pod-1"]

    backend.active, backend.local_volume_path = True, "/workspace"
    second = await cycle()
    if stopped_by_us:
        assert second.passed, second.event
        assert ssh.pod_running
        assert _pod_states(second)["pod-1"] == "running"
    else:
        assert not second.passed
        assert second.event.reason_code == Msg.POD_NOT_RUNNING.reason
        assert not ssh.pod_running
        docker.recover_parked_pod_container.assert_not_awaited()


@pytest.mark.parametrize(
    "state",
    [
        "both_gone",
        "parked",
        "stopped_anchor",
        "anchor_without_mark",
        "forged_anchor",
        "stopped_canonical",
        "lock_busy",
        "create_in_flight",
        "delete_fails",
        "delete_hangs",
    ],
)
@pytest.mark.asyncio
async def test_tenant_enforcement_reports_a_pod_it_stopped_down_once_its_container_is_gone(
    context_factory, monkeypatch, state
):
    # a pod the validator stopped passes until, under the create lock, neither <name> nor its parked <name>__prev
    # exists: only then is the mark dropped and POD_NOT_RUNNING reported, or the renter is billed for a pod that no
    # longer exists; a parked original (failed recovery) or an unconfirmed check keeps the pass (lio#1827 review)
    ssh = DummySSHClient(pod_running=False)
    real_run = ssh.run
    present = {
        "parked": "pod_pod-1__prev",
        "stopped_anchor": "pod_pod-1__stopped",
        "anchor_without_mark": "pod_pod-1__stopped",
        "forged_anchor": "pod_pod-1__stopped",
        "stopped_canonical": "pod_pod-1",
    }.get(state)

    async def run(command):
        if "{{.State.Running}}" not in command:
            return await real_run(command)
        if present and command.endswith(f" {present}"):
            return Mock(stdout="false\n", stderr="", exit_status=0)
        return Mock(stdout="", stderr="Error: No such object", exit_status=1)

    ssh.run = run
    # anchor_without_mark (review 5785846085): the stop's mark write failed, so only the __stopped name attributes it
    # forged_anchor (lio#1827 review): the provider made the __stopped container; no record of our own park, no pass
    unmarked = state in ("anchor_without_mark", "forged_anchor")
    marks = {} if unmarked else {"stopped_by_recovery:pod-1": "stopped"}
    lock = Mock(acquire=AsyncMock(return_value=state != "lock_busy"), release=AsyncMock())
    redis = _recovery_redis(lock, marks)
    redis.backend_create_inflight = AsyncMock(return_value=state == "create_in_flight")
    if state == "delete_fails":
        redis.delete = AsyncMock(side_effect=ConnectionError("redis down"))
    if state == "delete_hangs":
        # a stalled delete past the deadline: the confirmed absence still stands, every cycle

        async def hang_delete(_key):
            await asyncio.Event().wait()

        redis.delete = hang_delete
        monkeypatch.setattr(rented_machine_module, "_RECOVERY_MARK_CLEAR_TIMEOUT_SEC", 0.01)
    docker = AsyncMock()
    docker.recover_parked_pod_container.return_value = False  # a failed recovery: nothing renamed back
    docker.recover_pod_after_stale_vloopback_mount.return_value = False
    docker.encrypted_volume_mount_missing.return_value = False
    docker.unmarked_stop_parked.return_value = state == "anchor_without_mark"

    async def missing(*args, **kwargs):
        return {"container_missing": True}

    monkeypatch.setattr(rented_machine_module, "_collect_pod_diagnostics", missing)
    ctx = build_recovery_context(context_factory, ssh, docker, redis=redis)
    result = await TenantEnforcementCheck().run(ctx)
    if state == "delete_hangs":
        result = await TenantEnforcementCheck().run(ctx)

    if state in ("both_gone", "delete_fails", "delete_hangs", "forged_anchor"):
        assert not result.passed
        assert result.event.reason_code == Msg.POD_NOT_RUNNING.reason
        assert result.updates["clear_verified_job_reason"] == ResetVerifiedJobReason.POD_NOT_RUNNING.value
        assert marks == ({} if state in ("both_gone", "forged_anchor") else {"stopped_by_recovery:pod-1": "stopped"})
    else:
        assert result.passed
        assert "clear_verified_job_reason" not in result.updates
        assert marks == ({} if unmarked else {"stopped_by_recovery:pod-1": "stopped"})
        assert result.updates["default_extra"]["pods_stopped_by_validator"] == ["pod-1"]


@pytest.mark.asyncio
async def test_tenant_enforcement_leaves_the_pod_untouched_when_cancelled_during_the_path_lookup(context_factory):
    # the path is looked up before the mount probe, so a cancel there has not judged or changed anything yet
    ssh = DummySSHClient(pod_running=True, ssh_keys=["ssh-rsa renter"])
    docker = AsyncMock()
    docker.encrypted_volume_mount_missing.return_value = True

    async def restore(**kwargs):
        if not kwargs["local_volume_path"]:
            ssh.pod_running = False
        return False

    docker.restore_volume_mount_of_running_container.side_effect = restore
    entered = asyncio.Event()

    async def lookup(_pod_id):
        entered.set()
        await asyncio.Event().wait()

    backend = DummyBackendClient(active=True, local_volume_path="/workspace")
    backend.get_pod_rental_active = lookup
    lock = Mock(acquire=AsyncMock(return_value=True), release=AsyncMock())
    ctx = build_recovery_context(context_factory, ssh, docker, backend=backend, redis=_recovery_redis(lock, {}))

    task = asyncio.create_task(TenantEnforcementCheck().run(ctx))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert ssh.pod_running
    docker.encrypted_volume_mount_missing.assert_not_awaited()
    lock.acquire.assert_not_awaited()


@pytest.mark.asyncio
async def test_tenant_enforcement_emits_transport_unreachable_when_recheck_loses_ssh(
    context_factory,
):
    """DAH-2055 still holds on the recovery path: if the SSH transport dies during the
    post-recovery re-check, pod state is unknown, so the miner must get
    EXECUTOR_TRANSPORT_UNREACHABLE rather than a penalty or an uncaught exception."""
    ssh = DummySSHClient(pod_running=False)
    docker = AsyncMock()

    async def start_pod_then_lose_transport(**kwargs):
        ssh.raise_on_run = asyncssh.ConnectionLost("host rebooted again")
        return True

    docker.recover_pod_after_stale_vloopback_mount.side_effect = start_pod_then_lose_transport
    ctx = build_recovery_context(context_factory, ssh, docker)

    result = await TenantEnforcementCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.EXECUTOR_TRANSPORT_UNREACHABLE.reason
    assert "clear_verified_job_info" not in result.updates
    assert "clear_verified_job_reason" not in result.updates


@pytest.mark.asyncio
async def test_tenant_enforcement_emits_transport_unreachable_when_repair_loses_ssh(
    context_factory,
):
    """The vloopback repair runs over the same ctx.ssh session, so a transport death there is
    DAH-2055 territory too: pod state is unknown and the miner must not be penalised."""
    docker = AsyncMock()
    docker.recover_pod_after_stale_vloopback_mount.side_effect = asyncssh.ConnectionLost(
        "host went away mid-repair"
    )
    ctx = build_recovery_context(context_factory, DummySSHClient(pod_running=False), docker)

    result = await TenantEnforcementCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.EXECUTOR_TRANSPORT_UNREACHABLE.reason
    assert "clear_verified_job_info" not in result.updates
    assert "clear_verified_job_reason" not in result.updates


@pytest.mark.asyncio
async def test_tenant_enforcement_skips_recovery_without_private_key(context_factory):
    docker = AsyncMock()
    docker.recover_pod_after_stale_vloopback_mount.return_value = True
    ctx = build_recovery_context(
        context_factory, DummySSHClient(pod_running=False), docker, private_key=None
    )

    result = await TenantEnforcementCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.POD_NOT_RUNNING.reason
    docker.recover_pod_after_stale_vloopback_mount.assert_not_awaited()


@pytest.mark.asyncio
async def test_tenant_enforcement_skips_recovery_in_dry_run(context_factory):
    # the dry run pipeline runs against live executors alongside production validation, so it must
    # report the pod as down without rmdir'ing a host path or starting a customer's container
    docker = AsyncMock()
    docker.recover_pod_after_stale_vloopback_mount.return_value = True
    ctx = build_recovery_context(context_factory, DummySSHClient(pod_running=False), docker)

    result = await TenantEnforcementCheck(recover_stale_pods=False).run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.POD_NOT_RUNNING.reason
    docker.recover_pod_after_stale_vloopback_mount.assert_not_awaited()


def test_dry_run_pipeline_does_not_recover_pods():
    recovery_flags = [
        check.recover_stale_pods
        for check in PipelineFactory.build_dry_run_checks()
        if type(check).__name__ == "TenantEnforcementCheck"
    ]

    assert recovery_flags == [False]


def test_context_annotations_resolve_at_runtime():
    # Every test above builds Context with model_construct, which skips validation and hides an
    # annotation pydantic cannot resolve. A ContextServices field typed only under TYPE_CHECKING
    # leaves Context unbuildable, and then every real validation cycle raises instead of running.
    assert Context.model_rebuild(force=True) is True


# ---------------------------------------------------------------------------
# DAH-3338: the container state of every rented pod rides ctx.state.pod_states
# ---------------------------------------------------------------------------


class AbsentContainerSSHClient(DummySSHClient):
    """`docker ps` lists nothing and `docker inspect` answers "No such container"."""

    def __init__(self):
        super().__init__(pod_running=False)

    async def run(self, command: str):
        if "docker inspect" in command:
            self.commands_called.append(command)
            return Mock(stdout="", stderr="Error: No such container: tenant-123")
        return await super().run(command)


def _pod_states(result) -> dict[str, str]:
    return {s.pod_id: s.container_state.value for s in result.updates["state"].pod_states}


def _tenant_ctx(context_factory, ssh_client, rented_machine, *, backend=None, prior_state=None):
    services = build_services(
        score_calculator=DummyScoreCalculator(actual_score=1.0, job_score=1.0, warning=""),
        container_cleanup=MockContainerCleanup(),
        backend=backend or DummyBackendClient(active=True),
    )
    state = build_state(
        gpu_processes=[],
        gpu_details=[],
        gpu_model="NVIDIA RTX 4090",
        rented_data=build_rented_data("executor-123", rented_machine),
        pod_states=prior_state or [],
    )
    return context_factory(
        services=services,
        config=build_context_config(),
        state=state,
        ssh=ssh_client,
        collateral_deposited=True,
        is_rental_succeed=True,
        contract_version="v1.0.0",
    )


@pytest.mark.asyncio
async def test_pod_states_reports_running_for_a_healthy_rented_pod(context_factory):
    ctx = _tenant_ctx(
        context_factory,
        DummySSHClient(pod_running=True, ssh_keys=["ssh-rsa AAA"]),
        {"containers": [{"name": "tenant-123", "pod_id": "pod-1"}]},
    )

    result = await TenantEnforcementCheck().run(ctx)

    assert result.event.reason_code == Msg.ALREADY_RENTED.reason
    assert _pod_states(result) == {"pod-1": "running"}
    observed_at = result.updates["state"].pod_states[0].observed_at
    assert observed_at.tzinfo is not None


@pytest.mark.asyncio
async def test_a_pod_stopped_by_its_renters_gpu_reset_is_not_penalised(context_factory):
    # DAH-4658: the reset stops the container for a moment; only the pod named on the mark is spared
    pods = {"containers": [{"name": "tenant-123", "pod_id": "pod-1"}]}
    for marked_pod, spared in (("pod-1", True), ("pod-2", False)):
        ctx = _tenant_ctx(context_factory, DummySSHClient(pod_running=False), pods)
        ctx.services.redis.gpu_reset_marks = AsyncMock(return_value=[{"pod_id": marked_pod, "kind": "gpu_reset"}])

        result = await TenantEnforcementCheck().run(ctx)

        assert result.passed is spared
        assert (result.event.reason_code == Msg.POD_NOT_RUNNING.reason) is not spared


@pytest.mark.asyncio
async def test_pod_states_reports_exited_when_inspect_shows_a_stopped_container(context_factory):
    # DummySSHClient answers docker inspect with Status=exited; the rental is still open, so
    # the cycle also fails with POD_NOT_RUNNING as before.
    ctx = _tenant_ctx(
        context_factory,
        DummySSHClient(pod_running=False),
        {"containers": [{"name": "tenant-123", "pod_id": "pod-1"}]},
    )

    result = await TenantEnforcementCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.POD_NOT_RUNNING.reason
    assert _pod_states(result) == {"pod-1": "exited"}


class RestartingContainerSSHClient(DummySSHClient):
    """`docker ps` lists nothing and `docker inspect` shows dockerd between restarts of the pod."""

    def __init__(self, *, error: str = "", exit_code: int = 1, oom_killed: bool = False):
        super().__init__(pod_running=False)
        self.error = error
        self.exit_code = exit_code
        self.oom_killed = oom_killed

    async def run(self, command: str):
        if "docker inspect" in command:
            self.commands_called.append(command)
            state = {
                "Status": "restarting",
                "ExitCode": self.exit_code,
                "OOMKilled": self.oom_killed,
                "Error": self.error,
                "FinishedAt": self.container_finished_at,
            }
            return Mock(stdout=json.dumps(state), stderr="")
        return await super().run(command)


@pytest.mark.asyncio
async def test_a_pod_whose_restart_fails_to_start_is_still_not_running(context_factory):
    ctx = _tenant_ctx(
        context_factory,
        RestartingContainerSSHClient(error="OCI runtime create failed: nvidia-container-cli: initialization error"),
        {"containers": [{"name": "tenant-123", "pod_id": "pod-1"}]},
    )

    result = await TenantEnforcementCheck().run(ctx)

    assert result.event.reason_code == Msg.POD_NOT_RUNNING.reason
    assert result.updates["clear_verified_job_reason"] == ResetVerifiedJobReason.POD_NOT_RUNNING.value


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("exit_code", "oom_killed", "case"),
    [
        (1, False, "provider sent a handled SIGUSR1; PID 1 chose exit 1"),
        (0, False, "graceful daemon restart — PID 1 handled SIGTERM and exited 0"),
        (143, False, "daemon stop / SIGTERM (128+15)"),
        (137, False, "host SIGKILL (128+9)"),
        (137, True, "host OOM-kill"),
        (139, False, "docker kill --signal=SEGV (128+11), forgeable by the provider"),
        (134, False, "docker kill --signal=ABRT (128+6), forgeable by the provider"),
        (129, False, "SIGHUP (128+1), a repo-canonical host kill"),
    ],
)
async def test_a_restarting_pod_without_renter_fault_evidence_is_not_shielded(
    context_factory, exit_code, oom_killed, case
):
    """Review finding (Serhii, #1534): a provider Docker-daemon restart or a host/provider signal can leave
    an unless-stopped container `restarting` with an empty State.Error, and the provider controls that
    tooling — no exit code (a handled signal can make PID 1 pick any) is trusted renter attribution. Each such case
    must fall through to POD_NOT_RUNNING (clears the verified job, provider-fault path), not be labelled a
    renter crash-loop that keeps verification."""
    ctx = _tenant_ctx(
        context_factory,
        RestartingContainerSSHClient(exit_code=exit_code, oom_killed=oom_killed),
        {"containers": [{"name": "tenant-123", "pod_id": "pod-1"}]},
    )

    result = await TenantEnforcementCheck().run(ctx)

    assert result.event.reason_code == Msg.POD_NOT_RUNNING.reason, case
    assert result.updates["clear_verified_job_reason"] == ResetVerifiedJobReason.POD_NOT_RUNNING.value


@pytest.mark.asyncio
async def test_pod_states_reports_absent_when_no_container_of_that_name_exists(context_factory):
    ctx = _tenant_ctx(
        context_factory,
        AbsentContainerSSHClient(),
        {"containers": [{"name": "tenant-123", "pod_id": "pod-1"}]},
    )

    result = await TenantEnforcementCheck().run(ctx)

    assert result.event.reason_code == Msg.POD_NOT_RUNNING.reason
    assert _pod_states(result) == {"pod-1": "absent"}


@pytest.mark.asyncio
async def test_pod_states_reports_the_state_when_the_backend_says_the_rental_is_closed(
    context_factory,
):
    # STALE_POD_NOT_RUNNING is a pass, and still carries what the host showed.
    ctx = _tenant_ctx(
        context_factory,
        AbsentContainerSSHClient(),
        {"containers": [{"name": "tenant-123", "pod_id": "pod-1"}]},
        backend=DummyBackendClient(active=False),
    )

    result = await TenantEnforcementCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.STALE_POD_NOT_RUNNING.reason
    assert _pod_states(result) == {"pod-1": "absent"}


@pytest.mark.asyncio
async def test_pod_states_is_unknown_not_absent_on_ssh_transport_loss(context_factory):
    ctx = _tenant_ctx(
        context_factory,
        DummySSHClient(raise_on_run=asyncssh.ConnectionLost("conntrack flush")),
        {
            "containers": [
                {"name": "tenant-123", "pod_id": "pod-1"},
                {"name": "tenant-456", "pod_id": "pod-2"},
            ]
        },
    )

    result = await TenantEnforcementCheck().run(ctx)

    assert result.event.reason_code == Msg.EXECUTOR_TRANSPORT_UNREACHABLE.reason
    # Neither pod was inspected: the one the transport died on and the one never reached.
    assert _pod_states(result) == {"pod-1": "unknown", "pod-2": "unknown"}


@pytest.mark.asyncio
async def test_pod_states_keeps_running_for_pods_seen_before_the_transport_died(context_factory):
    ssh = DummySSHClient(pod_running=True)
    seen: list[str] = []

    async def run(command: str):
        # The first pod's two commands (docker ps, authorized_keys) answer; then the link dies.
        seen.append(command)
        if "tenant-456" in command:
            raise asyncssh.ConnectionLost("link dropped")
        return await DummySSHClient.run(ssh, command)

    ssh.run = run
    ctx = _tenant_ctx(
        context_factory,
        ssh,
        {
            "containers": [
                {"name": "tenant-123", "pod_id": "pod-1"},
                {"name": "tenant-456", "pod_id": "pod-2"},
            ]
        },
    )

    result = await TenantEnforcementCheck().run(ctx)

    assert result.event.reason_code == Msg.EXECUTOR_TRANSPORT_UNREACHABLE.reason
    assert _pod_states(result) == {"pod-1": "running", "pod-2": "unknown"}


@pytest.mark.asyncio
async def test_pod_states_is_unknown_when_the_recovery_recheck_loses_ssh(context_factory):
    ssh = DummySSHClient(pod_running=False)
    docker = AsyncMock()

    async def start_pod_then_lose_transport(**kwargs):
        ssh.raise_on_run = asyncssh.ConnectionLost("host rebooted again")
        return True

    docker.recover_pod_after_stale_vloopback_mount.side_effect = start_pod_then_lose_transport
    ctx = build_recovery_context(context_factory, ssh, docker)

    result = await TenantEnforcementCheck().run(ctx)

    assert result.event.reason_code == Msg.EXECUTOR_TRANSPORT_UNREACHABLE.reason
    # inspect said exited before the repair, but the re-check never answered: unknown wins.
    assert _pod_states(result) == {"pod-1": "unknown"}


@pytest.mark.asyncio
async def test_pod_states_reports_running_after_a_successful_recovery(context_factory):
    ssh = DummySSHClient(pod_running=False, ssh_keys=["ssh-rsa recovered"])
    docker = AsyncMock()

    async def bring_pod_back_up(**kwargs):
        ssh.pod_running = True
        return True

    docker.recover_pod_after_stale_vloopback_mount.side_effect = bring_pod_back_up
    ctx = build_recovery_context(context_factory, ssh, docker)

    result = await TenantEnforcementCheck().run(ctx)

    assert result.event.reason_code == Msg.ALREADY_RENTED.reason
    assert _pod_states(result) == {"pod-1": "running"}


@pytest.mark.asyncio
async def test_pod_states_are_appended_to_what_the_stale_cleanup_reaped(context_factory):
    from datetime import UTC, datetime

    from protocol.vc_protocol.validator_requests import ContainerState, PodContainerState

    reaped = PodContainerState(
        pod_id="orphan-9", container_state=ContainerState.REAPED, observed_at=datetime.now(UTC)
    )
    ctx = _tenant_ctx(
        context_factory,
        DummySSHClient(pod_running=True),
        {"containers": [{"name": "tenant-123", "pod_id": "pod-1"}]},
        prior_state=[reaped],
    )

    result = await TenantEnforcementCheck().run(ctx)

    assert _pod_states(result) == {"orphan-9": "reaped", "pod-1": "running"}


@pytest.mark.asyncio
async def test_pod_states_is_untouched_when_the_executor_is_not_rented(context_factory):
    ctx = _tenant_ctx(context_factory, DummySSHClient(pod_running=True), None)

    result = await TenantEnforcementCheck().run(ctx)

    assert result.event.reason_code == Msg.NOT_RENTED.reason
    assert "state" not in result.updates


# DAH-4315: the backend lists a new pod before it records its container name.
UNNAMED_POD = {"containers": [{"name": "", "pod_id": "53a68595"}]}


def _unnamed_pod_ctx(context_factory, ssh_client, gpu_processes, backend=None, pods=UNNAMED_POD):
    services = build_services(
        score_calculator=DummyScoreCalculator(actual_score=1.0, job_score=1.0, warning=""),
        container_cleanup=MockContainerCleanup(),
        backend=backend or DummyBackendClient(active=True),
    )
    state = build_state(
        gpu_processes=gpu_processes,
        gpu_details=[{"gpu_utilization": 95, "memory_utilization": 80}],
        gpu_model="NVIDIA RTX 4090",
        rented_data=build_rented_data("executor-123", pods),
    )
    return context_factory(
        services=services,
        config=build_context_config(),
        state=state,
        ssh=ssh_client,
        collateral_deposited=True,
        is_rental_succeed=True,
        contract_version="v1.0.0",
    )


def test_rented_pod_without_a_container_name_resolves_to_its_pod_container():
    for unset in ("", None):
        pod = RentedPod(pod_id="53a68595", container_name=unset)
        assert pod.container_name_pending
        assert pod.docker_container_name == "pod_53a68595"
    named = RentedPod(pod_id="53a68595", container_name="pod_other")
    assert not named.container_name_pending
    assert named.docker_container_name == "pod_other"


@pytest.mark.asyncio
async def test_unnamed_pod_owns_the_gpu_process_in_its_pod_container(context_factory):
    ctx = _unnamed_pod_ctx(
        context_factory,
        DummySSHClient(pod_running=True),
        [{"container_name": "pod_53a68595", "pid": 1234}],
    )

    result = await TenantEnforcementCheck().run(ctx)

    assert result.passed
    assert result.event.reason_code == Msg.ALREADY_RENTED.reason


@pytest.mark.asyncio
async def test_unnamed_pod_still_fails_a_gpu_process_in_another_container(context_factory):
    ctx = _unnamed_pod_ctx(
        context_factory,
        DummySSHClient(pod_running=True),
        [{"container_name": "pod_someone_else", "pid": 1234}],
    )

    result = await TenantEnforcementCheck().run(ctx)

    assert not result.passed
    assert result.event.reason_code == Msg.GPU_OUTSIDE_TENANT.reason


@pytest.mark.asyncio
async def test_unnamed_pod_never_runs_a_docker_filter_on_an_empty_name(context_factory):
    ssh = DummySSHClient(pod_running=True)
    ctx = _unnamed_pod_ctx(context_factory, ssh, [])

    await TenantEnforcementCheck().run(ctx)

    ps_commands = [c for c in ssh.commands_called if "docker ps" in c]
    assert ps_commands == ["/usr/bin/docker ps -q -f 'name=^/?pod_53a68595$'"]
    assert not any(c.rstrip().endswith("name=") for c in ssh.commands_called)


@pytest.mark.asyncio
async def test_unnamed_pod_whose_container_is_not_up_yet_is_not_penalised(context_factory):
    backend = DummyBackendClient(active=True)
    ctx = _unnamed_pod_ctx(context_factory, DummySSHClient(pod_running=False), [], backend)

    result = await TenantEnforcementCheck().run(ctx)

    assert result.passed
    assert result.event.reason_code == Msg.ALREADY_RENTED.reason
    assert "clear_verified_job_reason" not in result.updates
    assert _pod_states(result) == {"53a68595": "unknown"}
    assert backend.called_with == []


@pytest.mark.asyncio
async def test_unnamed_pod_still_not_up_after_the_grace_is_not_running(context_factory):
    created_at = datetime.now(UTC) - timedelta(minutes=POD_CONTAINER_NAME_GRACE_MINUTES + 1)
    pods = {"containers": [{"name": "", "pod_id": "53a68595", "created_at": created_at}]}
    ssh = DummySSHClient(pod_running=False)
    ctx = _unnamed_pod_ctx(context_factory, ssh, [], DummyBackendClient(active=True), pods)

    result = await TenantEnforcementCheck().run(ctx)

    assert not result.passed
    assert result.event.reason_code == Msg.POD_NOT_RUNNING.reason


@pytest.mark.parametrize("module", ["services.docker_service", "services.task.checks.rented_machine"])
def test_a_fresh_process_imports_the_module(module):
    # an already-initialised pytest process hides an import cycle (docker_service -> task ->
    # rented_machine -> docker_service broke the connector's startup)
    src = Path(__file__).resolve().parents[1] / "src"
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"], cwd=src, capture_output=True, text=True, timeout=180
    )
    assert result.returncode == 0, result.stderr[-2000:]


REAPED_FILLER = "filler_5703f4c9-c2f4-4fae-a652-3dee4753030a"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "processes, passed",
    [
        pytest.param([REAPED_FILLER], True, id="reaped-filler-only"),
        pytest.param(["pod_53a68595", REAPED_FILLER], True, id="tenant-and-reaped-filler"),
        pytest.param(
            ["pod_53a68595", REAPED_FILLER, "my_miner"],
            False,
            id="a-surviving-foreign-process-still-fails",
        ),
    ],
)
async def test_a_container_the_cleanup_reaped_this_cycle_is_not_gpu_usage_outside_the_tenant(
    context_factory, processes, passed
):
    """Review 4371171967: TenantEnforcementCheck runs before GpuUsageCheck on the same processes."""
    from tests.test_gpu_usage_check import _after_the_cleanup_removed

    ctx = _unnamed_pod_ctx(
        context_factory,
        DummySSHClient(pod_running=True),
        [{"container_name": name, "pid": 1000 + i} for i, name in enumerate(processes)],
    )
    ctx = await _after_the_cleanup_removed(ctx, [REAPED_FILLER])

    assert all(process["container_name"] != REAPED_FILLER for process in ctx.state.gpu_processes)
    result = await TenantEnforcementCheck().run(ctx)

    assert result.passed is passed
    expected = Msg.ALREADY_RENTED.reason if passed else Msg.GPU_OUTSIDE_TENANT.reason
    assert result.event.reason_code == expected


@pytest.mark.asyncio
async def test_unnamed_pod_whose_create_is_still_running_after_the_grace_is_not_penalised(context_factory):
    # a custom-Dockerfile build keeps the pod PENDING with no container for longer than the grace
    created_at = datetime.now(UTC) - timedelta(minutes=POD_CONTAINER_NAME_GRACE_MINUTES + 20)
    pods = {"containers": [{"name": "", "pod_id": "53a68595", "created_at": created_at, "status": "PENDING"}]}
    ssh = DummySSHClient(pod_running=False)
    ctx = _unnamed_pod_ctx(context_factory, ssh, [], DummyBackendClient(active=True), pods)

    result = await TenantEnforcementCheck().run(ctx)

    assert result.passed
    assert result.event.reason_code == Msg.ALREADY_RENTED.reason
    assert "clear_verified_job_reason" not in result.updates


def test_ps_running_matches_the_whole_container_name_only():
    # Docker's name filter is an unanchored regex: `name=pod_1` also listed a `pod_12` container as this pod
    assert DockerCommand.ps_running("pod_1.a") == "/usr/bin/docker ps -q -f 'name=^/?pod_1\\.a$'"


FILLER_ID = "f1" * 32


async def _rented_cycle_with_filler_process(
    context_factory, redis, listed_fillers: list[str], filler: str, specs: dict | None = None
):
    executor_uuid = "executor-123"
    pod_container = "tenant-123"
    rented_data = build_rented_data(
        executor_uuid,
        {"containers": [{"name": pod_container, "pod_id": "pod-1"}], "owner_flag": False},
    )
    rented_data.all_filler_containers_by_executor[executor_uuid] = listed_fillers
    state = build_state(
        gpu_processes=[{"container_name": pod_container, "pid": 1001}, {"container_name": filler, "pid": 1002}],
        gpu_details=[{"gpu_utilization": 95, "memory_utilization": 80}],
        rented_data=rented_data,
        specs=specs or {"docker": {"containers": [{"name": filler, "container_id": FILLER_ID}]}},
    )
    ctx = context_factory(
        services=build_services(redis=redis, container_cleanup=MockContainerCleanup()),
        state=state,
        ssh=DummySSHClient(pod_running=True, ssh_keys=["ssh-rsa AAA..."]),
        collateral_deposited=True,
        contract_version="v1.0.0",
    )
    return await TenantEnforcementCheck().run(ctx)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("listed_last_cycle", "reason"),
    [(True, Msg.ALREADY_RENTED.reason), (False, Msg.GPU_OUTSIDE_TENANT.reason)],
    ids=["our-filler-stopping", "never-our-filler"],
)
async def test_a_filler_listed_on_this_node_last_cycle_still_counts_as_our_load(
    context_factory, listed_last_cycle, reason
):
    # DAH-5904: a filler preempted by a customer rent leaves the snapshot before its container is gone
    redis = build_services().redis
    filler = "filler_885e1616-28e0-4792-9890-47f71893d27f"
    await _rented_cycle_with_filler_process(context_factory, redis, [filler] if listed_last_cycle else [], filler)

    result = await _rented_cycle_with_filler_process(context_factory, redis, [], filler)

    assert result.event.reason_code == reason


@pytest.mark.asyncio
async def test_a_filler_created_and_stopped_between_two_snapshots_still_counts_as_our_load(context_factory):
    # DAH-5904 review: the create records the filler, so no cycle snapshot needs to have listed it
    redis = build_services().redis
    filler = "filler_0b6e2b1c-6f5e-4c43-9d55-2f6d1f0d7a10"
    await DockerService._remember_new_filler_best_effort(
        SimpleNamespace(redis_service=redis), "executor-123", filler, FILLER_ID
    )

    result = await _rented_cycle_with_filler_process(context_factory, redis, [], filler)

    assert result.event.reason_code == Msg.ALREADY_RENTED.reason


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("digest", "reason"), [("sha256:aaa", Msg.ALREADY_RENTED.reason), ("sha256:bbb", Msg.GPU_OUTSIDE_TENANT.reason)]
)
async def test_a_remembered_filler_name_running_another_image_is_not_our_load(context_factory, digest, reason):
    redis = build_services().redis
    filler = "filler_885e1616-28e0-4792-9890-47f71893d27f"
    await redis.set(f"filler_image_digests:{filler}", '["sha256:aaa"]')
    await _rented_cycle_with_filler_process(context_factory, redis, [filler], filler)
    specs = {"docker": {"containers": [{"name": filler, "digest": digest, "container_id": FILLER_ID}]}}
    result = await _rented_cycle_with_filler_process(context_factory, redis, [], filler, specs=specs)

    assert result.event.reason_code == reason


@pytest.mark.asyncio
@pytest.mark.parametrize("remembered_at", ["cycle", "create"])
async def test_a_replacement_container_under_a_remembered_filler_name_and_image_is_not_our_load(
    context_factory, remembered_at
):
    redis = build_services().redis
    filler = "filler_885e1616-28e0-4792-9890-47f71893d27f"
    await redis.set(f"filler_image_digests:{filler}", '["sha256:aaa"]')
    if remembered_at == "cycle":
        await _rented_cycle_with_filler_process(context_factory, redis, [filler], filler)
    else:
        await DockerService._remember_new_filler_best_effort(
            SimpleNamespace(redis_service=redis), "executor-123", filler, FILLER_ID
        )
    replacement = {"name": filler, "digest": "sha256:aaa", "container_id": "e2" * 32}
    specs = {"docker": {"containers": [replacement]}}

    result = await _rented_cycle_with_filler_process(context_factory, redis, [], filler, specs=specs)

    assert result.event.reason_code == Msg.GPU_OUTSIDE_TENANT.reason



@pytest.mark.asyncio
async def test_tenant_enforcement_a_pod_recovered_in_the_cycle_keeps_the_unanswered_count(context_factory):
    # a provider must not reset the limit by letting recovery start the pod and stopping it again
    ssh = DummySSHClient(pod_running=False, ssh_keys=["ssh-rsa renter"])
    docker = AsyncMock()
    docker.recover_pod_after_stale_vloopback_mount.return_value = False
    docker.recover_parked_pod_container.return_value = False
    docker.encrypted_volume_mount_missing.return_value = False
    backend, redis = DummyBackendClient(active=None), FakeRedis()

    def cycle():
        return TenantEnforcementCheck().run(build_recovery_context(context_factory, ssh, docker, backend=backend, redis=redis))

    assert [(await cycle()).passed for _ in range(2)] == [True, True]

    async def bring_pod_back_up(**kwargs):
        ssh.pod_running = True
        return True

    docker.recover_pod_after_stale_vloopback_mount.side_effect = bring_pod_back_up
    assert (await cycle()).passed is True
    assert "rental_state_unknown:pod-1" in redis.store

    ssh.pod_running = False
    docker.recover_pod_after_stale_vloopback_mount.side_effect = None
    result = await cycle()
    assert result.passed is False
    assert result.event.reason_code == "POD_NOT_RUNNING"


class _PerPodBackend(DummyBackendClient):
    def __init__(self, answers: dict[str, bool | None]):
        super().__init__()
        self.answers = answers

    async def get_pod_rental_active(self, pod_id: str):
        self.called_with.append(pod_id)
        if self.answers[pod_id] is None:
            return None
        return Mock(active=self.answers[pod_id], rental_closed_at=None, local_volume_path=None)


@pytest.mark.parametrize("unknown_first", [True, False])
@pytest.mark.parametrize("exhausted", [False, True])
@pytest.mark.asyncio
async def test_tenant_enforcement_a_closed_pod_beside_a_held_one_keeps_the_node_rented(
    context_factory, unknown_first, exhausted
):
    # pod-u's rental check is unanswered, pod-c's rental is confirmed closed; both are down
    pods = [
        RentedPod(pod_id="pod-u", container_name="tenant-u", rented_ports=[], gpu_count=2),
        RentedPod(pod_id="pod-c", container_name="tenant-c", rented_ports=[], gpu_count=6),
    ]
    rented_data = RentedExecutorsResponse(
        executors={
            "executor-123": RentedExecutor(
                miner_hotkey="miner-hotkey",
                executor_ip_address="127.0.0.1",
                executor_ip_port="22",
                pods=pods if unknown_first else pods[::-1],
                owner_flag=False,
            )
        },
        banned_guids=[],
    )
    redis = FakeRedis()
    if exhausted:
        await redis.set("rental_state_unknown:pod-u", "2")
    ctx = _unknown_rental_ctx(context_factory, _PerPodBackend({"pod-u": None, "pod-c": False}), redis)
    ctx = ctx.model_copy(update={"state": replace(ctx.state, rented_data=rented_data)})

    result = await TenantEnforcementCheck().run(ctx)

    if not exhausted:
        assert result.passed is True and result.halt is True
        assert result.updates["rented"] is True
        extra = result.updates["default_extra"]
        assert extra["pods_rental_state_unknown"] == ["pod-u"]
        assert extra["pods_rental_closed"] == ["pod-c"]
        assert extra["rented_gpu_count"] == 2
        live = result.updates["state"].rented_data.executors["executor-123"].pods
        assert [pod.pod_id for pod in live] == ["pod-u"]
    else:
        # past the hold limit the down pod is judged in either order; the closed pod cannot hide it
        assert result.event.reason_code == Msg.POD_NOT_RUNNING.reason


def _held_and_closed_ctx(context_factory, *, order, ssh=None, gpu_processes=(), gpu_details=()):
    pods = {
        "u": RentedPod(pod_id="pod-u", container_name="tenant-u", rented_ports=[], gpu_count=2),
        "c": RentedPod(pod_id="pod-c", container_name="tenant-c", rented_ports=[], gpu_count=6),
    }
    rented_data = RentedExecutorsResponse(
        executors={
            "executor-123": RentedExecutor(
                miner_hotkey="miner-hotkey",
                executor_ip_address="127.0.0.1",
                executor_ip_port="22",
                pods=[pods[key] for key in order],
                owner_flag=False,
            )
        },
        banned_guids=[],
    )
    ctx = _unknown_rental_ctx(context_factory, _PerPodBackend({"pod-u": None, "pod-c": False}), FakeRedis())
    state = replace(ctx.state, rented_data=rented_data, gpu_processes=list(gpu_processes), gpu_details=list(gpu_details))
    return ctx.model_copy(update={"state": state, **({"ssh": ssh} if ssh else {})})


@pytest.mark.asyncio
async def test_tenant_enforcement_gpu_work_of_the_closed_sibling_is_not_outside_the_tenant(context_factory):
    # the GPU snapshot predates the renter ending pod-c, so its processes are still the renter's
    ctx = _held_and_closed_ctx(
        context_factory,
        order="uc",
        gpu_processes=[{"container_name": "tenant-c", "pid": 1234}],
        gpu_details=[{"gpu_utilization": 100, "memory_utilization": 90}],
    )

    result = await TenantEnforcementCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.ALREADY_RENTED.reason


@pytest.mark.asyncio
async def test_tenant_enforcement_a_hung_sibling_read_reports_the_node_unreachable(context_factory, monkeypatch):
    from neurons.validators.src.services.task.checks import rented_machine

    class HangsOnPodU(DummySSHClient):
        async def run(self, command: str):
            if "tenant-u" in command:
                await asyncio.sleep(3600)
            return await super().run(command)

    monkeypatch.setattr(rented_machine, "task_seconds_left", lambda _ctx: rented_machine.POD_READ_RESERVE_SECONDS)
    ctx = _held_and_closed_ctx(context_factory, order="cu", ssh=HangsOnPodU(pod_running=False))

    result = await asyncio.wait_for(TenantEnforcementCheck().run(ctx), 5)

    assert result.passed is False
    assert result.event.reason_code == Msg.EXECUTOR_TRANSPORT_UNREACHABLE.reason
