import json
from unittest.mock import AsyncMock, Mock

import pytest
from helpers import build_context_config, build_services, build_state
from neurons.validators.src.services.task.checks.rented_machine import (
    TenantEnforcementCheck,
)
from neurons.validators.src.services.task.messages import TenantEnforcementMessages as Msg
from neurons.validators.src.services.task.pipeline import Context

from protocol.vc_protocol.compute_requests import RentedExecutor, RentedExecutorsResponse, RentedPod
from protocol.vc_protocol.validator_requests import ResetVerifiedJobReason


class MockContainerCleanup:
    """Mock container cleanup service for tests."""
    async def cleanup(self, ssh_client, rented_data, executor_uuid):
        return 0, []


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
async def test_tenant_enforcement_penalises_when_pod_stays_down_after_recovery(context_factory):
    docker = AsyncMock()
    docker.recover_pod_after_stale_vloopback_mount.return_value = True
    ctx = build_recovery_context(context_factory, DummySSHClient(pod_running=False), docker)

    result = await TenantEnforcementCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.POD_NOT_RUNNING.reason


# ---------------------------------------------------------------------------
# DAH-3338: the container state of every rented pod rides ctx.state.pod_states
# ---------------------------------------------------------------------------


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

