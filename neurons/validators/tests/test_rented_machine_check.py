import pytest
from unittest.mock import Mock

from neurons.validators.src.services.task.checks.rented_machine import TenantEnforcementCheck
from neurons.validators.src.services.task.messages import TenantEnforcementMessages as Msg
from protocol.vc_protocol.validator_requests import ResetVerifiedJobReason

from tests.helpers import build_context_config, build_services, build_state


class DummyRedisService:
    """Mock Redis service for rental status and port maps."""

    def __init__(self, *, rented_machine: dict | None = None, port_maps: list[bytes] | None = None):
        """
        Args:
            rented_machine: The rental info to return from get_rented_machine
            port_maps: The port map bytes to return from lrange
        """
        self.rented_machine = rented_machine
        self.port_maps = port_maps or []
        self.get_rented_called = False
        self.lrange_called_with: str | None = None

    async def get_rented_machine(self, executor):
        """Mock get_rented_machine."""
        self.get_rented_called = True
        return self.rented_machine

    async def lrange(self, key: str):
        """Mock lrange for port maps."""
        self.lrange_called_with = key
        return self.port_maps


class DummySSHClient:
    """Mock SSH client for pod health and SSH keys checks."""

    def __init__(self, *, pod_running: bool = True, ssh_keys: list[str] | None = None, should_raise: bool = False):
        """
        Args:
            pod_running: Whether the container is running
            ssh_keys: SSH keys to return from authorized_keys
            should_raise: Whether to raise an exception
        """
        self.pod_running = pod_running
        self.ssh_keys = ssh_keys or []
        self.should_raise = should_raise
        self.commands_called: list[str] = []

    async def run(self, command: str):
        """Mock SSH run command."""
        self.commands_called.append(command)

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


class DummyPortMappingService:
    """Mock port mapping service."""

    def __init__(self, *, port_count: int = 0, should_raise: bool = False):
        """
        Args:
            port_count: Number of successful ports
            should_raise: Whether to raise an exception
        """
        self.port_count = port_count
        self.should_raise = should_raise
        self.get_called_with: str | None = None

    async def get_successful_ports_count(self, executor_uuid: str) -> int:
        """Mock get_successful_ports_count."""
        self.get_called_with = executor_uuid
        if self.should_raise:
            raise RuntimeError("Port mapping error")
        return self.port_count


class DummyScoreCalculator:
    """Mock score calculator."""

    def __init__(self, *, actual_score: float = 1.0, job_score: float = 1.0, warning: str = ""):
        self.actual_score = actual_score
        self.job_score = job_score
        self.warning = warning
        self.called_with: dict | None = None

    def __call__(self, gpu_model: str, collateral: bool, rental_succeed: bool, contract_version: str, rented: bool, port_count: int):
        self.called_with = {
            "gpu_model": gpu_model,
            "collateral": collateral,
            "rental_succeed": rental_succeed,
            "contract_version": contract_version,
            "rented": rented,
            "port_count": port_count,
        }
        return self.actual_score, self.job_score, self.warning


@pytest.mark.parametrize(
    "rented_machine,pod_running,ssh_keys,owner_flag,gpu_processes,gpu_details,port_count_db,port_maps,expected_pass,expected_reason,expect_halt",
    [
        # Not rented - should pass and continue
        (None, True, [], False, [], [], 0, [], True, Msg.NOT_RENTED.reason, False),
        # Not rented (no container_name) - should pass and continue
        ({"container_name": ""}, True, [], False, [], [], 0, [], True, Msg.NOT_RENTED.reason, False),

        # Rented but pod not running - should fail
        (
            {"container_name": "tenant-123"},
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
            {"container_name": "tenant-123", "owner_flag": False},
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
            {"container_name": "tenant-123", "owner_flag": True},
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
            {"container_name": "tenant-123", "owner_flag": False},
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
            {"container_name": "tenant-123", "owner_flag": False},
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
            {"container_name": "tenant-123", "owner_flag": False},
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
    # Create mock Redis service
    redis_service = DummyRedisService(
        rented_machine=rented_machine,
        port_maps=port_maps,
    )

    # Create mock SSH client
    ssh_client = DummySSHClient(
        pod_running=pod_running,
        ssh_keys=ssh_keys,
    )

    # Create mock port mapping service
    port_mapping_service = DummyPortMappingService(port_count=port_count_db)

    # Create mock score calculator
    score_calculator = DummyScoreCalculator(actual_score=1.0, job_score=1.0, warning="")

    # Setup services
    services = build_services(
        redis=redis_service,
        port_mapping=port_mapping_service,
        score_calculator=score_calculator,
    )

    # Setup config
    config = build_context_config()

    # Setup state with GPU info
    state = build_state(
        gpu_processes=gpu_processes,
        gpu_details=gpu_details,
        gpu_model="NVIDIA RTX 4090",
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

    # Verify Redis was called
    assert redis_service.get_rented_called is True

    # Verify SSH interactions for rented machines
    if rented_machine and rented_machine.get("container_name"):
        container_name = rented_machine.get("container_name")
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
    if not rented_machine or not rented_machine.get("container_name"):
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
