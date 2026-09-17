import json
from contextlib import contextmanager
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import asyncssh
import pytest
from helpers import build_context_config, build_services, build_state
from neurons.validators.src.core.docker_utils import DockerCommand, _collect_host_context
from neurons.validators.src.services.task.checks import rented_machine as rented_machine_module
from neurons.validators.src.services.task.checks.rented_machine import (
    SSH_PORT_NOT_RENTED,
    TenantEnforcementCheck,
    _collect_pod_diagnostics,
    _published_ssh_port,
)
from neurons.validators.src.services.task.messages import TenantEnforcementMessages as Msg
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
        )
        for c in containers
    ]

    executor = RentedExecutor(
        miner_hotkey="test-miner",
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
async def test_tenant_enforcement_keeps_pod_not_running_when_backend_state_unknown(context_factory):
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

    assert result.passed is False
    assert result.event.reason_code == Msg.POD_NOT_RUNNING.reason
    assert backend.called_with == ["pod-1"]
    assert result.updates["clear_verified_job_info"] is True
    assert result.updates["clear_verified_job_reason"] == ResetVerifiedJobReason.POD_NOT_RUNNING.value


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


# --- DAH-2255: a running container is not a reachable pod; the renter's SSH port must answer ---------------

RENTER_SSH_PORT = 40299
RENTER_PORTS = [40299, 40300, 40301]
POD_AGE_OLD = datetime.utcnow() - timedelta(hours=2)


class PortAwareSSHClient(DummySSHClient):
    """DummySSHClient that also answers `docker port <pod> 22/tcp` the way the host does."""

    def __init__(self, *, docker_port_stdout: str = f"0.0.0.0:{RENTER_SSH_PORT}\n[::]:{RENTER_SSH_PORT}\n", **kwargs):
        super().__init__(**kwargs)
        self.docker_port_stdout = docker_port_stdout

    async def run(self, command: str):
        if "docker port" in command:
            self.commands_called.append(command)
            if self.raise_on_run is not None:
                raise self.raise_on_run
            return Mock(stdout=self.docker_port_stdout, stderr="", exit_status=0 if self.docker_port_stdout else 1)
        return await super().run(command)


def build_ssh_check_context(
    context_factory,
    *,
    ssh: DummySSHClient,
    pods: list[RentedPod] | None = None,
) -> Context:
    pods = pods or [
        RentedPod(pod_id="pod-1", container_name="pod_pod-1", rented_ports=RENTER_PORTS, created_at=POD_AGE_OLD)
    ]
    rented_data = RentedExecutorsResponse(
        executors={
            "executor-123": RentedExecutor(
                miner_hotkey="test-miner", executor_ip_address="127.0.0.1", executor_ip_port="22", pods=pods
            )
        }
    )
    services = build_services(
        score_calculator=DummyScoreCalculator(actual_score=1.0, job_score=1.0, warning=""),
        container_cleanup=MockContainerCleanup(),
        backend=DummyBackendClient(active=True),
    )
    return context_factory(
        services=services,
        config=build_context_config(),
        state=build_state(gpu_processes=[], gpu_details=[], gpu_model="NVIDIA RTX 4090", rented_data=rented_data),
        ssh=ssh,
        collateral_deposited=True,
        is_rental_succeed=True,
        contract_version="v1.0.0",
    )


@contextmanager
def ssh_check_settings(*, enabled: bool = True, deadline: float = 0, grace: int = 10):
    with patch("neurons.validators.src.services.task.checks.rented_machine.settings") as s:
        s.RENTED_POD_SSH_CHECK_ENABLED = enabled
        s.RENTED_POD_SSH_DEADLINE_SECONDS = deadline
        s.RENTED_POD_SSH_GRACE_MINUTES = grace
        yield s


@contextmanager
def renter_ssh_port(*answers):
    """What the validator sees when it dials the renter's port: one entry per attempt, None = sshd's banner,
    `(kind, detail)` = the failure; the last entry repeats. Yields the mock so tests read the dial targets."""
    answers = list(answers) or [None]

    async def dial(host, port, **kwargs):
        return answers.pop(0) if len(answers) > 1 else answers[0]

    dialled = AsyncMock(side_effect=dial)
    with (
        patch("neurons.validators.src.services.task.checks.rented_machine.ssh_banner_error", dialled),
        patch("neurons.validators.src.services.task.checks.rented_machine._SSH_BANNER_POLL_SECONDS", 0.0),
    ):
        yield dialled


def test_rented_pod_ssh_check_is_off_by_default():
    """Regression: the flag's default flipped to True, so every validator dials every renter's port on deploy."""
    assert type(rented_machine_module.settings).model_fields["RENTED_POD_SSH_CHECK_ENABLED"].default is False


@pytest.mark.asyncio
async def test_rented_pod_ssh_check_off_dials_nothing(context_factory):
    """Regression: the check ignores the flag (dials with the default settings object, no patch)."""
    ssh = PortAwareSSHClient(pod_running=True, ssh_keys=["ssh-rsa AAA..."])
    ctx = build_ssh_check_context(context_factory, ssh=ssh)

    with renter_ssh_port(("refused", "Connection refused")) as dialled:
        result = await TenantEnforcementCheck().run(ctx)

    assert result.passed is True and result.event.reason_code == Msg.ALREADY_RENTED.reason
    dialled.assert_not_awaited()
    assert not any("docker port" in cmd for cmd in ssh.commands_called)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["refused", "timeout", "no_banner", "unreachable"])
async def test_rented_pod_whose_ssh_port_does_not_answer_fails_with_a_reason_code(context_factory, kind):
    """Regression (DAH-2255, prod 17 Sep 2026): `docker ps` says running, the renter's port refuses, the check
    returned RENTED / score 1.0 every cycle. Now: RENTED_POD_SSH_UNREACHABLE, verified job cleared, the dialled
    host and port and the failure kind in the reset evidence the backend's penalty row carries."""
    ssh = PortAwareSSHClient(pod_running=True, ssh_keys=["ssh-rsa AAA..."])
    ctx = build_ssh_check_context(context_factory, ssh=ssh)

    with ssh_check_settings(), renter_ssh_port((kind, f"port {RENTER_SSH_PORT}: {kind}")) as dialled:
        result = await TenantEnforcementCheck().run(ctx)

    assert result.passed is False
    assert result.halt is False
    assert result.event.reason_code == Msg.POD_SSH_UNREACHABLE.reason
    # dialled from the validator to the executor's public address on the port docker publishes for 22
    assert dialled.await_args.args == ("127.0.0.1", RENTER_SSH_PORT)
    assert any("docker port" in cmd and "pod_pod-1" in cmd and "22/tcp" in cmd for cmd in ssh.commands_called)
    saw = result.event.what_we_saw
    assert (saw["pod_id"], saw["ssh_host"], saw["ssh_port"], saw["ssh_failure"]) == ("pod-1", "127.0.0.1", RENTER_SSH_PORT, kind)
    assert str(RENTER_SSH_PORT) in result.event.remediation and kind in result.event.remediation
    assert result.updates["clear_verified_job_info"] is True
    assert "clear_verified_job_reason" not in result.updates  # not POD_NOT_RUNNING: the container runs
    evidence = result.updates["clear_verified_job_evidence"]
    assert evidence["reason_code"] == Msg.POD_SSH_UNREACHABLE.reason
    assert evidence["check_id"] == TenantEnforcementCheck.check_id
    assert (evidence["pod_id"], evidence["ssh_host"], evidence["ssh_port"], evidence["ssh_failure"]) == (
        "pod-1", "127.0.0.1", RENTER_SSH_PORT, kind
    )


@pytest.mark.asyncio
async def test_rented_pod_whose_ssh_port_answers_stays_rented(context_factory):
    ssh = PortAwareSSHClient(pod_running=True, ssh_keys=["ssh-rsa AAA..."])
    ctx = build_ssh_check_context(context_factory, ssh=ssh)

    with ssh_check_settings(), renter_ssh_port(None) as dialled:
        result = await TenantEnforcementCheck().run(ctx)

    assert result.passed is True and result.halt is True
    assert result.event.reason_code == Msg.ALREADY_RENTED.reason
    assert result.updates["rented"] is True
    assert dialled.await_count == 1


@pytest.mark.asyncio
async def test_rented_pod_ssh_check_retries_until_the_deadline(context_factory):
    """Regression: one dropped connection fails the pod (sshd under MaxStartups drops a connection now and then)."""
    ssh = PortAwareSSHClient(pod_running=True, ssh_keys=["ssh-rsa AAA..."])
    ctx = build_ssh_check_context(context_factory, ssh=ssh)

    with (
        ssh_check_settings(deadline=30),
        renter_ssh_port(("unreachable", "reset"), ("no_banner", "closed"), None) as dialled,
    ):
        result = await TenantEnforcementCheck().run(ctx)

    assert result.passed is True and result.event.reason_code == Msg.ALREADY_RENTED.reason
    assert dialled.await_count == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "created_at",
    [
        pytest.param(datetime.utcnow() - timedelta(minutes=1), id="within_grace"),
        pytest.param(None, id="age_unknown"),
    ],
)
async def test_rented_pod_inside_its_startup_grace_is_not_dialled(context_factory, created_at):
    """Regression: a pod created a minute ago fails while its start script is still bringing sshd up."""
    ssh = PortAwareSSHClient(pod_running=True, ssh_keys=["ssh-rsa AAA..."])
    pod = RentedPod(pod_id="pod-1", container_name="pod_pod-1", rented_ports=RENTER_PORTS, created_at=created_at)
    ctx = build_ssh_check_context(context_factory, ssh=ssh, pods=[pod])

    with ssh_check_settings(grace=10), renter_ssh_port(("refused", "refused")) as dialled:
        result = await TenantEnforcementCheck().run(ctx)

    assert result.passed is True and result.event.reason_code == Msg.ALREADY_RENTED.reason
    dialled.assert_not_awaited()
    assert not any("docker port" in cmd for cmd in ssh.commands_called)


@pytest.mark.asyncio
async def test_rented_pod_ssh_port_outside_the_renters_ports_fails_without_a_dial(context_factory):
    """Regression: the host publishes container port 22 on a port the backend never gave this renter; the
    renter's `ssh -p` goes to a port that maps nowhere, whatever answers on the host's port."""
    ssh = PortAwareSSHClient(pod_running=True, ssh_keys=["ssh-rsa AAA..."], docker_port_stdout="0.0.0.0:50000\n")
    ctx = build_ssh_check_context(context_factory, ssh=ssh)

    with ssh_check_settings(), renter_ssh_port(None) as dialled:
        result = await TenantEnforcementCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.POD_SSH_UNREACHABLE.reason
    assert result.event.what_we_saw["ssh_failure"] == SSH_PORT_NOT_RENTED
    assert result.event.what_we_saw["ssh_port"] == 50000
    assert result.event.what_we_saw["rented_ports"] == RENTER_PORTS
    assert result.updates["clear_verified_job_evidence"]["ssh_failure"] == SSH_PORT_NOT_RENTED
    dialled.assert_not_awaited()


@pytest.mark.asyncio
async def test_rented_pod_ssh_check_reaches_no_verdict_when_the_host_reports_no_port(context_factory):
    """A Docker daemon that cannot answer `docker port` is not proof about sshd: no dial, no penalty."""
    ssh = PortAwareSSHClient(pod_running=True, ssh_keys=["ssh-rsa AAA..."], docker_port_stdout="")
    ctx = build_ssh_check_context(context_factory, ssh=ssh)

    with ssh_check_settings(), renter_ssh_port(("refused", "refused")) as dialled:
        result = await TenantEnforcementCheck().run(ctx)

    assert result.passed is True and result.event.reason_code == Msg.ALREADY_RENTED.reason
    dialled.assert_not_awaited()


@pytest.mark.asyncio
async def test_rented_pod_ssh_check_reports_a_lost_transport_not_the_pod(context_factory):
    """DAH-2055 holds here too: the management shell dying during `docker port` is unknown pod state."""

    class LoseTransportOnDockerPort(PortAwareSSHClient):
        async def run(self, command: str):
            if "docker port" in command:
                raise asyncssh.ConnectionLost("conntrack flush")
            return await super().run(command)

    ssh = LoseTransportOnDockerPort(pod_running=True, ssh_keys=["ssh-rsa AAA..."])
    ctx = build_ssh_check_context(context_factory, ssh=ssh)

    with ssh_check_settings(), renter_ssh_port(None) as dialled:
        result = await TenantEnforcementCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.EXECUTOR_TRANSPORT_UNREACHABLE.reason
    assert "clear_verified_job_info" not in result.updates
    dialled.assert_not_awaited()


@pytest.mark.asyncio
async def test_rented_pod_ssh_check_judges_every_pod_on_a_split_node(context_factory):
    """Regression: only the first pod is dialled, so a GPU-split node's second renter is left unreachable."""

    class PerPodPorts(PortAwareSSHClient):
        async def run(self, command: str):
            if "docker port" in command:
                self.docker_port_stdout = "0.0.0.0:40299\n" if "pod_pod-1" in command else "0.0.0.0:40400\n"
            return await super().run(command)

    ssh = PerPodPorts(pod_running=True, ssh_keys=["ssh-rsa AAA..."])
    pods = [
        RentedPod(pod_id="pod-1", container_name="pod_pod-1", rented_ports=[40299, 40300], created_at=POD_AGE_OLD),
        RentedPod(pod_id="pod-2", container_name="pod_pod-2", rented_ports=[40400, 40401], created_at=POD_AGE_OLD),
    ]
    ctx = build_ssh_check_context(context_factory, ssh=ssh, pods=pods)

    async def dial(host, port, **kwargs):
        return None if port == 40299 else ("refused", "Connection refused")

    with (
        ssh_check_settings(),
        patch("neurons.validators.src.services.task.checks.rented_machine.ssh_banner_error", AsyncMock(side_effect=dial)) as dialled,
    ):
        result = await TenantEnforcementCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.POD_SSH_UNREACHABLE.reason
    assert result.event.what_we_saw["pod_id"] == "pod-2"
    assert result.event.what_we_saw["ssh_port"] == 40400
    assert sorted(call.args[1] for call in dialled.await_args_list) == [40299, 40400]


@pytest.mark.parametrize(
    "stdout,expected",
    [
        ("0.0.0.0:40299\n[::]:40299\n", 40299),
        ("[::]:40299\n", 40299),
        ("", None),
        ("Error: No public port '22/tcp' published for pod_x\n", None),
    ],
)
def test_published_ssh_port_reads_docker_port_output(stdout, expected):
    assert _published_ssh_port(stdout) == expected


def test_docker_port_command_quotes_the_container_name():
    command = DockerCommand.published_port("pod_x; rm -rf /", 22)
    assert command == "/usr/bin/docker port 'pod_x; rm -rf /' 22/tcp"
