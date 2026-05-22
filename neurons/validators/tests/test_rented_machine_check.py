import json
from unittest.mock import Mock

import asyncssh
import pytest
from helpers import build_context_config, build_services, build_state
from neurons.validators.src.services.task.checks.rented_machine import (
    TenantEnforcementCheck,
    _collect_host_context,
    _collect_pod_diagnostics,
)
from neurons.validators.src.services.task.messages import TenantEnforcementMessages as Msg
from protocol.vc_protocol.compute_requests import NetworkEMA, RentedExecutor, RentedExecutorsResponse, RentedPod
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
            assert any("authorized_keys" in cmd for cmd in ssh_client.commands_called)

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

    assert diagnostics["state"] == state
    assert diagnostics["logs_tail"].endswith("traceback line 2\n")
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

    assert "state" not in diagnostics
    assert diagnostics["inspect_error"] == "Error: No such object: pod_abc"
    assert "logs_tail" not in diagnostics


@pytest.mark.asyncio
async def test_collect_pod_diagnostics_never_raises_on_ssh_failure():
    ssh = DiagnosticsSSHClient(raise_on="docker inspect")

    diagnostics = await _collect_pod_diagnostics(ssh, "pod_abc")

    assert "inspect_error" in diagnostics
    assert "ssh failed" in diagnostics["inspect_error"]


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

    assert diagnostics["state"]["OOMKilled"] is True
    assert diagnostics["host_context"] == {
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
async def test_tenant_enforcement_protects_filler_only_without_marking_rented(context_factory):
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
            network_ema={
                executor_uuid: NetworkEMA(
                    ema_verifyx_download_speed=250.0,
                    ema_verifyx_upload_speed=50.0,
                )
            },
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
    assert result.halt is True
    assert result.event.reason_code == Msg.NOT_RENTED.reason
    assert result.updates["rented"] is False
    assert result.updates["success"] is True
    assert score_calculator.called_with["rented"] is False
    assert score_calculator.called_with["ctx"].state.specs["network"]["ema_verifyx_download_speed"] == 250.0
    assert score_calculator.called_with["ctx"].state.specs["network"]["ema_verifyx_upload_speed"] == 50.0
    assert not any("docker ps" in cmd for cmd in ssh_client.commands_called)


@pytest.mark.asyncio
async def test_tenant_enforcement_rejects_unmapped_filler_process(context_factory):
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

    assert result.passed is False
    assert result.event.reason_code == Msg.GPU_OUTSIDE_TENANT.reason
    assert result.event.what_we_saw["expected_containers"] == [filler_container]


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
