from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import asyncssh
import pytest
from neurons.validators.src.core.docker_utils import ContainerDeathDiagnostics
from neurons.validators.src.protocol.vc_protocol.compute_requests import (
    CPU_QUOTA_EXCEEDS_HOST_REASON,
    GPU_RUNTIME_DEVICE_FAULT_REASON,
    GPU_RUNTIME_NVML_MISMATCH_REASON,
    ExecutorHealthCheckResponse,
    FillerRunActiveResponse,
)
from neurons.validators.src.services.container_cleanup import ContainerCleanup
from neurons.validators.src.services.task.checks.rental_verification import RentalVerificationCheck
from neurons.validators.src.services.task.messages import RentalVerificationMessages as Msg
from neurons.validators.src.services.task.pipeline import CheckResult, Context

from protocol.vc_protocol.compute_requests import RentedExecutor, RentedExecutorsResponse, RentedPod
from tests.helpers import build_services, build_state


class DummyBackendClient:
    def __init__(
        self,
        *,
        response: ExecutorHealthCheckResponse | None,
        filler_run_active: FillerRunActiveResponse | None = None,
    ):
        self.response = response
        self.called_with: dict | None = None
        self.filler_run_active = filler_run_active
        self.filler_run_active_calls: list[str] = []
        self.filler_run_container_missing_flags: list[bool] = []

    async def get_filler_run_active(
        self, filler_run_id: str, *, container_missing: bool = False
    ) -> FillerRunActiveResponse | None:
        self.filler_run_active_calls.append(filler_run_id)
        self.filler_run_container_missing_flags.append(container_missing)
        return self.filler_run_active

    async def check_executor_health(
        self,
        miner_address: str,
        miner_port: int,
        miner_hotkey: str,
        container_port: int,
        executor_id: str | None = None,
        rental_in_progress: bool = False,
        gpu_uuids: list[str] | None = None,
        cpu_count: int | None = None,
    ):
        self.called_with = {
            "miner_address": miner_address,
            "miner_port": miner_port,
            "miner_hotkey": miner_hotkey,
            "container_port": container_port,
            "executor_id": executor_id,
            "rental_in_progress": rental_in_progress,
            "gpu_uuids": gpu_uuids,
            "cpu_count": cpu_count,
        }
        return self.response


@pytest.mark.parametrize(
    "skip_verification,expected_pass,expected_reason",
    [
        (True, True, Msg.SKIPPED.reason),
        (False, True, Msg.VERIFIED.reason),
    ],
)
@pytest.mark.asyncio
async def test_rental_verification_skip(
    skip_verification,
    expected_pass,
    expected_reason,
    context_factory,
):
    """Test that rental verification is skipped when SKIP_RENTAL_VERIFICATION is enabled."""
    backend_client = DummyBackendClient(
        response=ExecutorHealthCheckResponse(success=True, error=None, details={})
    )
    services = build_services(backend=backend_client, container_cleanup=ContainerCleanup())
    state = build_state(specs={"verified_ports": [8080]})

    ctx = context_factory(services=services, state=state)

    with patch("neurons.validators.src.services.task.checks.rental_verification.settings") as mock_settings:
        mock_settings.SKIP_RENTAL_VERIFICATION = skip_verification
        result = await RentalVerificationCheck().run(ctx)

    assert result.passed is expected_pass
    assert result.event.reason_code == expected_reason

    # Should not call API when skipped
    if skip_verification:
        assert backend_client.called_with is None
    else:
        assert backend_client.called_with is not None


@pytest.mark.asyncio
async def test_rental_verification_no_ports():
    """Test that check fails when no verified ports are available."""
    backend_client = DummyBackendClient(
        response=ExecutorHealthCheckResponse(success=True, error=None, details={})
    )
    services = build_services(backend=backend_client, container_cleanup=ContainerCleanup())
    # No verified ports
    state = build_state(specs={})

    from tests.helpers import make_context
    ctx = make_context(services=services, state=state)

    with patch("neurons.validators.src.services.task.checks.rental_verification.settings") as mock_settings:
        mock_settings.SKIP_RENTAL_VERIFICATION = False
        result = await RentalVerificationCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.FAILED.reason
    assert "No verified ports available" in result.event.what_we_saw["error"]
    # Should not call API when no ports
    assert backend_client.called_with is None


@pytest.mark.asyncio
async def test_rental_verification_success():
    """Test successful rental verification."""
    backend_client = DummyBackendClient(
        response=ExecutorHealthCheckResponse(
            success=True,
            error=None,
            details={"container_healthy": True}
        )
    )
    services = build_services(backend=backend_client, container_cleanup=ContainerCleanup())
    state = build_state(specs={"verified_ports": [8080, 8081, 8082]})

    from tests.helpers import make_context
    ctx = make_context(services=services, state=state)

    with patch("neurons.validators.src.services.task.checks.rental_verification.settings") as mock_settings:
        mock_settings.SKIP_RENTAL_VERIFICATION = False
        result = await RentalVerificationCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.VERIFIED.reason
    assert result.event.what_we_saw["verified"] is True
    assert result.event.what_we_saw["details"]["container_healthy"] is True

    # Verify API was called with correct params
    assert backend_client.called_with == {
        "miner_address": "127.0.0.1",
        "miner_port": 8000,
        "miner_hotkey": "miner-hotkey",
        "container_port": 8080,  # First verified port
        "executor_id": "executor-123",
        "rental_in_progress": False,  # no customer rental in this state
        "gpu_uuids": [],  # this state carries no scraped gpu details
        "cpu_count": None,  # this state carries no scraped cpu count
    }


@pytest.mark.asyncio
async def test_rental_verification_failed():
    """Test rental verification failure."""
    backend_client = DummyBackendClient(
        response=ExecutorHealthCheckResponse(
            success=False,
            error="Container not responding",
            details={"timeout": True}
        )
    )
    services = build_services(backend=backend_client, container_cleanup=ContainerCleanup())
    state = build_state(specs={"verified_ports": [8080]})

    from tests.helpers import make_context
    ctx = make_context(services=services, state=state)

    with patch("neurons.validators.src.services.task.checks.rental_verification.settings") as mock_settings:
        mock_settings.SKIP_RENTAL_VERIFICATION = False
        result = await RentalVerificationCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.FAILED.reason
    assert result.event.what_we_saw["verified"] is False
    assert result.event.what_we_saw["error"] == "Container not responding"
    assert result.event.what_we_saw["details"]["timeout"] is True
    assert "clear_verified_job_info" not in result.updates
    assert "clear_verified_job_reason" not in result.updates


@pytest.mark.asyncio
async def test_rental_verification_nvml_mismatch_quarantines_without_clearing_verified_job():
    """Exact GPU runtime mismatch should mark the executor unhealthy."""
    stderr = (
        "docker: Error response from daemon: failed to create task for container: "
        "failed to initialize NVML: Driver/library version mismatch"
    )
    backend_client = DummyBackendClient(
        response=ExecutorHealthCheckResponse(
            success=False,
            error=stderr,
            details={"docker_stderr": stderr},
            reason_code=GPU_RUNTIME_NVML_MISMATCH_REASON,
        )
    )
    services = build_services(backend=backend_client, container_cleanup=ContainerCleanup())
    state = build_state(specs={"verified_ports": [8080]})

    from tests.helpers import make_context
    ctx = make_context(services=services, state=state)

    with patch("neurons.validators.src.services.task.checks.rental_verification.settings") as mock_settings:
        mock_settings.SKIP_RENTAL_VERIFICATION = False
        result = await RentalVerificationCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == GPU_RUNTIME_NVML_MISMATCH_REASON
    assert result.event.what_we_saw["source"] == "rental_verification"
    assert "failed to initialize NVML" in result.event.what_we_saw["stderr"]
    assert "clear_verified_job_info" not in result.updates
    assert "clear_verified_job_reason" not in result.updates


@pytest.mark.asyncio
async def test_rental_verification_device_fault_quarantines_the_host_too():
    """DAH-2336: a GPU that fell off the bus is as fatal as a driver mismatch — the probe must
    quarantine it, not fall into the generic fail branch that leaves the host rentable."""
    stderr = "nvidia-container-cli: device error: GPU-d2d12cfb-eb9e-03f0-b007-785d32aed1b2: unknown device"
    backend_client = DummyBackendClient(
        response=ExecutorHealthCheckResponse(
            success=False,
            error=stderr,
            details={"docker_stderr": stderr},
            reason_code=GPU_RUNTIME_DEVICE_FAULT_REASON,
        )
    )
    services = build_services(backend=backend_client, container_cleanup=ContainerCleanup())
    state = build_state(specs={"verified_ports": [8080]})

    from tests.helpers import make_context
    ctx = make_context(services=services, state=state)

    with patch("neurons.validators.src.services.task.checks.rental_verification.settings") as mock_settings:
        mock_settings.SKIP_RENTAL_VERIFICATION = False
        result = await RentalVerificationCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == GPU_RUNTIME_DEVICE_FAULT_REASON
    assert result.updates["score"] == 0.0
    assert "clear_verified_job_info" not in result.updates
    assert "unknown device" in result.event.what_we_saw["stderr"]


@pytest.mark.asyncio
async def test_rental_verification_api_error():
    """Test handling of API errors (returns None)."""
    backend_client = DummyBackendClient(response=None)  # API failure
    services = build_services(backend=backend_client, container_cleanup=ContainerCleanup())
    state = build_state(specs={"verified_ports": [8080]})

    from tests.helpers import make_context
    ctx = make_context(services=services, state=state)

    with patch("neurons.validators.src.services.task.checks.rental_verification.settings") as mock_settings:
        mock_settings.SKIP_RENTAL_VERIFICATION = False
        result = await RentalVerificationCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.API_ERROR.reason
    assert "API returned None" in result.event.what_we_saw["error"]


@pytest.mark.asyncio
async def test_rental_verification_exception():
    """Test handling of exceptions during API call."""
    class ExceptionBackendClient:
        async def check_executor_health(self, **kwargs):
            raise Exception("Network timeout")

    backend_client = ExceptionBackendClient()
    services = build_services(backend=backend_client, container_cleanup=ContainerCleanup())
    state = build_state(specs={"verified_ports": [8080]})

    from tests.helpers import make_context
    ctx = make_context(services=services, state=state)

    with patch("neurons.validators.src.services.task.checks.rental_verification.settings") as mock_settings:
        mock_settings.SKIP_RENTAL_VERIFICATION = False
        result = await RentalVerificationCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.API_ERROR.reason
    assert "Network timeout" in result.event.what_we_saw["error"]


@pytest.mark.asyncio
async def test_rental_verification_uses_first_verified_port():
    """Test that the check uses the first verified port from the list."""
    backend_client = DummyBackendClient(
        response=ExecutorHealthCheckResponse(success=True, error=None, details={})
    )
    services = build_services(backend=backend_client, container_cleanup=ContainerCleanup())
    # Multiple verified ports
    state = build_state(specs={"verified_ports": [9001, 9002, 9003]})

    from tests.helpers import make_context
    ctx = make_context(services=services, state=state)

    with patch("neurons.validators.src.services.task.checks.rental_verification.settings") as mock_settings:
        mock_settings.SKIP_RENTAL_VERIFICATION = False
        result = await RentalVerificationCheck().run(ctx)

    assert result.passed is True
    # Should use first port (9001)
    assert backend_client.called_with["container_port"] == 9001


# ---------------------------------------------------------------------------
# DAH-1991: post-rental-check cleanup of `health_check_*` containers.
# Force-remove must run on success, on backend failure, and on exception
# paths — but never alter the check's outcome.
# ---------------------------------------------------------------------------


class _RecordingSSH:
    def __init__(self):
        self.commands: list[str] = []

    async def run(self, cmd):
        self.commands.append(cmd)

        class _R:
            stdout = ""
            stderr = ""
            exit_status = 0

        return _R()


def _has_health_check_cleanup(commands: list[str]) -> bool:
    return any(
        "docker ps" in c
        and "name=^health_check_" in c
        and "docker rm -f" in c
        for c in commands
    )


@pytest.mark.asyncio
async def test_rental_verification_cleans_health_check_on_success():
    backend_client = DummyBackendClient(
        response=ExecutorHealthCheckResponse(success=True, error=None, details={})
    )
    services = build_services(backend=backend_client, container_cleanup=ContainerCleanup())
    state = build_state(specs={"verified_ports": [8080]})

    from tests.helpers import make_context
    ssh = _RecordingSSH()
    ctx = make_context(services=services, state=state, ssh=ssh)

    with patch("neurons.validators.src.services.task.checks.rental_verification.settings") as mock_settings:
        mock_settings.SKIP_RENTAL_VERIFICATION = False
        result = await RentalVerificationCheck().run(ctx)

    assert result.passed is True
    assert _has_health_check_cleanup(ssh.commands)


@pytest.mark.asyncio
async def test_rental_verification_cleans_health_check_on_failure():
    backend_client = DummyBackendClient(
        response=ExecutorHealthCheckResponse(
            success=False, error="Container not responding", details={}
        )
    )
    services = build_services(backend=backend_client, container_cleanup=ContainerCleanup())
    state = build_state(specs={"verified_ports": [8080]})

    from tests.helpers import make_context
    ssh = _RecordingSSH()
    ctx = make_context(services=services, state=state, ssh=ssh)

    with patch("neurons.validators.src.services.task.checks.rental_verification.settings") as mock_settings:
        mock_settings.SKIP_RENTAL_VERIFICATION = False
        result = await RentalVerificationCheck().run(ctx)

    assert result.passed is False
    assert _has_health_check_cleanup(ssh.commands)


@pytest.mark.asyncio
async def test_rental_verification_cleans_health_check_on_exception():
    class ExceptionBackendClient:
        async def check_executor_health(self, **kwargs):
            raise Exception("Network timeout")

    services = build_services(backend=ExceptionBackendClient(), container_cleanup=ContainerCleanup())
    state = build_state(specs={"verified_ports": [8080]})

    from tests.helpers import make_context
    ssh = _RecordingSSH()
    ctx = make_context(services=services, state=state, ssh=ssh)

    with patch("neurons.validators.src.services.task.checks.rental_verification.settings") as mock_settings:
        mock_settings.SKIP_RENTAL_VERIFICATION = False
        result = await RentalVerificationCheck().run(ctx)

    assert result.passed is False
    assert _has_health_check_cleanup(ssh.commands)


@pytest.mark.asyncio
async def test_rental_verification_cleanup_failure_does_not_change_result():
    """SSH error during cleanup must not flip the check's outcome."""
    backend_client = DummyBackendClient(
        response=ExecutorHealthCheckResponse(success=True, error=None, details={})
    )
    services = build_services(backend=backend_client, container_cleanup=ContainerCleanup())
    state = build_state(specs={"verified_ports": [8080]})

    class FailingSSH:
        async def run(self, cmd):
            raise Exception("ssh broken")

    from tests.helpers import make_context
    ctx = make_context(services=services, state=state, ssh=FailingSSH())

    with patch("neurons.validators.src.services.task.checks.rental_verification.settings") as mock_settings:
        mock_settings.SKIP_RENTAL_VERIFICATION = False
        result = await RentalVerificationCheck().run(ctx)

    # Cleanup blew up, but verification still passed.
    assert result.passed is True


@pytest.mark.asyncio
async def test_rental_verification_skips_cleanup_when_no_ports():
    """When the check exits early (no verified ports) the cleanup must NOT run.

    The early-return paths happen before any health_check_* probe could exist,
    so issuing a destructive cleanup there would be wasted work.
    """
    backend_client = DummyBackendClient(
        response=ExecutorHealthCheckResponse(success=True, error=None, details={})
    )
    services = build_services(backend=backend_client, container_cleanup=ContainerCleanup())
    state = build_state(specs={})

    from tests.helpers import make_context
    ssh = _RecordingSSH()
    ctx = make_context(services=services, state=state, ssh=ssh)

    with patch("neurons.validators.src.services.task.checks.rental_verification.settings") as mock_settings:
        mock_settings.SKIP_RENTAL_VERIFICATION = False
        result = await RentalVerificationCheck().run(ctx)

    assert result.passed is False
    # Early-return path: no docker rm -f issued.
    assert not _has_health_check_cleanup(ssh.commands)


class FillerSSHClient:
    """Mock SSH client answering the filler docker-ps liveness probe."""

    def __init__(
        self,
        *,
        running: bool = True,
        exit_status: int = 0,
        raise_on_run: BaseException | None = None,
        dead_containers: set[str] | None = None,
        unreachable_containers: set[str] | None = None,
        removed_from_host: bool = True,
        exit_code: int = 255,
        oom_killed: bool = False,
    ):
        # True = `docker inspect` says "No such object" (a removal); False = still there, exited.
        self.removed_from_host = removed_from_host
        self.exit_code = exit_code
        self.oom_killed = oom_killed
        self.running = running
        self.exit_status = exit_status
        self.raise_on_run = raise_on_run
        # Per-container control for GPU-split nodes: these answer the probe as gone while the rest
        # of the node's containers answer as alive.
        self.dead_containers = dead_containers or set()
        # Per-container SSH failure, so one bundle can be unreachable while another is a real kill.
        self.unreachable_containers = unreachable_containers or set()
        self.commands: list[str] = []

    async def run(self, command: str) -> Mock:
        self.commands.append(command)
        if self.raise_on_run is not None:
            raise self.raise_on_run
        if any(container in command for container in self.unreachable_containers):
            raise asyncssh.Error(code=1, reason="transport lost")
        result = Mock()
        result.exit_status = self.exit_status
        probed_container_is_dead = any(container in command for container in self.dead_containers)
        container_running = self.running and not probed_container_is_dead
        result.stdout = "container_id_123" if container_running and "docker ps" in command else ""
        result.stderr = ""
        if "docker inspect" in command and not container_running:
            if self.removed_from_host:
                result.exit_status = 1
                result.stderr = f"Error: No such object: {command.split()[-1]}"
            else:
                result.stdout = (
                    f'{{"Status": "exited", "ExitCode": {self.exit_code},'
                    f' "OOMKilled": {str(self.oom_killed).lower()},'
                    ' "Error": "error while mounting volume", "StartedAt": "2026-08-18T06:45:54Z",'
                    ' "FinishedAt": "2026-08-18T22:44:30Z"}'
                )
        return result


def _filler_context(
    *,
    backend_client: DummyBackendClient,
    ssh_client: FillerSSHClient,
    filler_container: str = "filler_11111111-2222-3333-4444-555555555555",
    filler_containers: list[str] | None = None,
    running_fillers: bool = True,
    create_killed: bool = False,
    rented_pods: list[RentedPod] | None = None,
) -> Context:
    # filler_containers = a GPU-split node's full bundle list (DAH-2465); the legacy single map is
    # still populated so the protocol's fallback path stays covered. running_fillers=False is the
    # DAH-2703 shape: the node reports no filler container because none ever survived its create.
    all_containers: list[str] = filler_containers if filler_containers else [filler_container]
    legacy_filler_by_executor: dict[str, str] = (
        {"executor-123": all_containers[0]} if running_fillers else {}
    )
    all_fillers_by_executor: dict[str, list[str]] = (
        {"executor-123": all_containers} if running_fillers else {}
    )
    executors: dict[str, RentedExecutor] = (
        {
            "executor-123": RentedExecutor(
                miner_hotkey="miner-hotkey",
                executor_ip_address="127.0.0.1",
                executor_ip_port="8000",
                pods=rented_pods,
            )
        }
        if rented_pods
        else {}
    )
    services = build_services(backend=backend_client, container_cleanup=ContainerCleanup())
    state = build_state(
        specs={"verified_ports": [8080]},
        rented_data=RentedExecutorsResponse(
            executors=executors,
            filler_containers_by_executor=legacy_filler_by_executor,
            all_filler_containers_by_executor=all_fillers_by_executor,
            filler_create_kill_executor_ids=["executor-123"] if create_killed else [],
        ),
    )
    from tests.helpers import make_context

    return make_context(services=services, state=state, ssh=ssh_client)


async def _run_filler_check(ctx: Context, *, check_enabled: bool = True, enforcement: bool = False) -> CheckResult:
    with patch("neurons.validators.src.services.task.checks.rental_verification.settings") as mock_settings:
        mock_settings.SKIP_RENTAL_VERIFICATION = False
        mock_settings.FILLER_LIVENESS_CHECK_ENABLED = check_enabled
        mock_settings.FILLER_LIVENESS_ENFORCEMENT_ENABLED = enforcement
        return await RentalVerificationCheck().run(ctx)


def _killed_filler_backend() -> DummyBackendClient:
    return DummyBackendClient(
        response=ExecutorHealthCheckResponse(success=True, error=None, details={}),
        filler_run_active=FillerRunActiveResponse(
            active=True,
            status="RUNNING",
            started_at=datetime.utcnow() - timedelta(minutes=30),
        ),
    )


@pytest.mark.parametrize("enforcement", [False, True])
@pytest.mark.asyncio
async def test_rental_verification_filler_liveness_disabled_keeps_legacy_skip(enforcement: bool):
    """CHECK_ENABLED is the master kill switch: off means no probe, even with enforcement on."""
    backend_client = DummyBackendClient(
        response=ExecutorHealthCheckResponse(success=True, error=None, details={})
    )
    ssh_client = FillerSSHClient(running=False)
    ctx = _filler_context(backend_client=backend_client, ssh_client=ssh_client)

    result = await _run_filler_check(ctx, check_enabled=False, enforcement=enforcement)

    assert result.passed is True
    assert result.event.reason_code == Msg.SKIPPED.reason
    assert ssh_client.commands == []
    assert backend_client.filler_run_active_calls == []


@pytest.mark.asyncio
async def test_rental_verification_filler_running_passes():
    """A filler container alive on the host passes; no health check, no re-check."""
    backend_client = DummyBackendClient(
        response=ExecutorHealthCheckResponse(success=True, error=None, details={})
    )
    ssh_client = FillerSSHClient(running=True)
    ctx = _filler_context(backend_client=backend_client, ssh_client=ssh_client)

    result = await _run_filler_check(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.FILLER_VERIFIED.reason
    assert backend_client.called_with is None
    assert backend_client.filler_run_active_calls == []


@pytest.mark.asyncio
async def test_rental_verification_filler_killed_shadow_mode_passes_but_logs():
    """Shadow mode: a killed filler is logged with enforced=False but incentive is untouched."""
    backend_client = _killed_filler_backend()
    ssh_client = FillerSSHClient(running=False)
    ctx = _filler_context(backend_client=backend_client, ssh_client=ssh_client)

    result = await _run_filler_check(ctx, enforcement=False)

    assert result.passed is True
    assert result.event.reason_code == Msg.FILLER_KILLED.reason
    assert result.event.what_we_saw["enforced"] is False
    assert backend_client.filler_run_active_calls == ["11111111-2222-3333-4444-555555555555"]
    # The re-check only happens when the container is gone, and it must SAY so — the backend's
    # zombie-run reconciliation (DAH-2471) is fed by this flag.
    assert backend_client.filler_run_container_missing_flags == [True]


@pytest.mark.asyncio
async def test_rental_verification_filler_killed_enforcement_fails():
    """Enforcement mode: a killed filler fails the fatal check -> no unrented incentive."""
    backend_client = _killed_filler_backend()
    ssh_client = FillerSSHClient(running=False, removed_from_host=True)
    ctx = _filler_context(backend_client=backend_client, ssh_client=ssh_client)

    result = await _run_filler_check(ctx, enforcement=True)

    assert result.passed is False
    assert result.event.reason_code == Msg.FILLER_KILLED.reason
    assert result.event.what_we_saw["enforced"] is True
    assert backend_client.called_with is None


@pytest.mark.asyncio
async def test_rental_verification_filler_missing_within_grace_passes():
    """A run younger than the grace window is not penalized for a missing container."""
    backend_client = DummyBackendClient(
        response=ExecutorHealthCheckResponse(success=True, error=None, details={}),
        filler_run_active=FillerRunActiveResponse(
            active=True,
            status="RUNNING",
            started_at=datetime.utcnow() - timedelta(minutes=2),
        ),
    )
    ssh_client = FillerSSHClient(running=False)
    ctx = _filler_context(backend_client=backend_client, ssh_client=ssh_client)

    result = await _run_filler_check(ctx, enforcement=True)

    assert result.passed is True
    assert result.event.reason_code == Msg.FILLER_STATE_UNKNOWN.reason


@pytest.mark.parametrize("enforcement", [False, True])
@pytest.mark.asyncio
async def test_rental_verification_filler_not_running_state_is_benign(enforcement: bool):
    """A run not in RUNNING (stopped or mid-transition) is benign: pass, no health check."""
    backend_client = DummyBackendClient(
        response=ExecutorHealthCheckResponse(success=True, error=None, details={}),
        filler_run_active=FillerRunActiveResponse(active=False, status="STOPPED"),
    )
    ssh_client = FillerSSHClient(running=False)
    ctx = _filler_context(backend_client=backend_client, ssh_client=ssh_client)

    result = await _run_filler_check(ctx, enforcement=enforcement)

    assert result.passed is True
    assert result.event.reason_code == Msg.FILLER_STATE_UNKNOWN.reason
    assert backend_client.called_with is None


@pytest.mark.asyncio
async def test_rental_verification_filler_docker_ps_error_fails_open():
    """A failing docker daemon (non-zero docker ps exit) must not be mistaken for a kill."""
    backend_client = _killed_filler_backend()
    ssh_client = FillerSSHClient(running=False, exit_status=1)
    ctx = _filler_context(backend_client=backend_client, ssh_client=ssh_client)

    result = await _run_filler_check(ctx, enforcement=True)

    assert result.passed is True
    assert result.event.reason_code == Msg.FILLER_STATE_UNKNOWN.reason
    assert backend_client.filler_run_active_calls == []


@pytest.mark.asyncio
async def test_rental_verification_filler_active_api_error_fails_open():
    """If the filler-run re-check API is unavailable, do not punish the miner."""
    backend_client = DummyBackendClient(
        response=ExecutorHealthCheckResponse(success=True, error=None, details={}),
        filler_run_active=None,
    )
    ssh_client = FillerSSHClient(running=False)
    ctx = _filler_context(backend_client=backend_client, ssh_client=ssh_client)

    result = await _run_filler_check(ctx, enforcement=True)

    assert result.passed is True
    assert result.event.reason_code == Msg.FILLER_STATE_UNKNOWN.reason
    assert backend_client.called_with is None


@pytest.mark.asyncio
async def test_rental_verification_filler_ssh_transport_error_fails_without_recheck():
    """Enforcement: a dead SSH transport means filler state is unknown -> fail the cycle, no re-check."""
    backend_client = DummyBackendClient(
        response=ExecutorHealthCheckResponse(success=True, error=None, details={})
    )
    ssh_client = FillerSSHClient(raise_on_run=asyncssh.Error(code=1, reason="connection lost"))
    ctx = _filler_context(backend_client=backend_client, ssh_client=ssh_client)

    result = await _run_filler_check(ctx, enforcement=True)

    assert result.passed is False
    assert result.event.reason_code == Msg.FILLER_TRANSPORT_UNREACHABLE.reason
    assert backend_client.filler_run_active_calls == []


@pytest.mark.asyncio
async def test_rental_verification_filler_ssh_transport_error_shadow_mode_passes():
    """Shadow mode: a dead SSH transport is logged but never fails the cycle."""
    backend_client = DummyBackendClient(
        response=ExecutorHealthCheckResponse(success=True, error=None, details={})
    )
    ssh_client = FillerSSHClient(raise_on_run=asyncssh.Error(code=1, reason="connection lost"))
    ctx = _filler_context(backend_client=backend_client, ssh_client=ssh_client)

    result = await _run_filler_check(ctx, enforcement=False)

    assert result.passed is True
    assert result.event.reason_code == Msg.FILLER_TRANSPORT_UNREACHABLE.reason


@pytest.mark.asyncio
async def test_rental_verification_sends_flag_when_customer_rental():
    """When this validator already sees a customer rental, it flags the backend to skip the check."""
    backend_client = DummyBackendClient(
        response=ExecutorHealthCheckResponse(success=True, error=None, details={"skipped": True})
    )
    services = build_services(backend=backend_client, container_cleanup=ContainerCleanup())
    state = build_state(
        specs={"verified_ports": [8080]},
        rented_data=RentedExecutorsResponse(
            executors={
                "executor-123": RentedExecutor(
                    miner_hotkey="miner-hotkey",
                    executor_ip_address="127.0.0.1",
                    executor_ip_port="8000",
                    pods=[RentedPod(pod_id="pod-1", container_name="c1")],
                )
            }
        ),
    )

    from tests.helpers import make_context

    ctx = make_context(services=services, state=state)

    with patch("neurons.validators.src.services.task.checks.rental_verification.settings") as mock_settings:
        mock_settings.SKIP_RENTAL_VERIFICATION = False
        result = await RentalVerificationCheck().run(ctx)

    assert result.passed is True
    assert backend_client.called_with is not None
    assert backend_client.called_with["rental_in_progress"] is True


@pytest.mark.asyncio
async def test_rental_verification_flag_false_when_no_customer_rental():
    """No customer rental → the flag is False and the backend still runs its own DB check."""
    backend_client = DummyBackendClient(
        response=ExecutorHealthCheckResponse(success=True, error=None, details={})
    )
    services = build_services(backend=backend_client, container_cleanup=ContainerCleanup())
    state = build_state(specs={"verified_ports": [8080]})

    from tests.helpers import make_context

    ctx = make_context(services=services, state=state)

    with patch("neurons.validators.src.services.task.checks.rental_verification.settings") as mock_settings:
        mock_settings.SKIP_RENTAL_VERIFICATION = False
        result = await RentalVerificationCheck().run(ctx)

    assert result.passed is True
    assert backend_client.called_with["rental_in_progress"] is False


# ---------------------------------------------------------------- GPU-split nodes (DAH-2465 bundles)

_BUNDLE_A = "filler_aaaaaaaa-1111-2222-3333-444444444444"
_BUNDLE_B = "filler_bbbbbbbb-1111-2222-3333-444444444444"


@pytest.mark.asyncio
async def test_filler_liveness_probes_every_bundle_container():
    """A GPU-split node has one filler per bundle; all of them must be probed, not just the first."""
    backend_client = DummyBackendClient(
        response=ExecutorHealthCheckResponse(success=True, error=None, details={})
    )
    ssh_client = FillerSSHClient(running=True)
    ctx = _filler_context(
        backend_client=backend_client, ssh_client=ssh_client, filler_containers=[_BUNDLE_A, _BUNDLE_B]
    )

    result = await _run_filler_check(ctx)

    probed = [c for c in ssh_client.commands if "docker ps" in c]
    assert any(_BUNDLE_A in c for c in probed)
    assert any(_BUNDLE_B in c for c in probed)
    assert result.passed is True
    assert [entry["reason_code"] for entry in result.event.what_we_saw["checked_containers"]] == [
        Msg.FILLER_VERIFIED.reason,
        Msg.FILLER_VERIFIED.reason,
    ]


@pytest.mark.asyncio
async def test_filler_liveness_reports_dead_non_first_bundle_to_backend():
    """Regression: a removed SIBLING bundle used to be invisible — no probe, no re-check, so the
    backend never got the container_missing evidence and DAH-2471 could not reconcile that run."""
    backend_client = _killed_filler_backend()
    ssh_client = FillerSSHClient(dead_containers={_BUNDLE_B})
    ctx = _filler_context(
        backend_client=backend_client, ssh_client=ssh_client, filler_containers=[_BUNDLE_A, _BUNDLE_B]
    )

    await _run_filler_check(ctx, enforcement=False)

    # Only the dead sibling triggers the live re-check, and it carries the reconciliation flag.
    assert backend_client.filler_run_active_calls == [_BUNDLE_B.removeprefix("filler_")]
    assert backend_client.filler_run_container_missing_flags == [True]


@pytest.mark.asyncio
async def test_filler_liveness_dead_sibling_decides_the_node_under_enforcement():
    """Under enforcement one killed bundle withholds incentive even if the other bundle is fine."""
    backend_client = _killed_filler_backend()
    ssh_client = FillerSSHClient(dead_containers={_BUNDLE_B})
    ctx = _filler_context(
        backend_client=backend_client, ssh_client=ssh_client, filler_containers=[_BUNDLE_A, _BUNDLE_B]
    )

    result = await _run_filler_check(ctx, enforcement=True)

    assert result.passed is False
    reason_codes = [entry["reason_code"] for entry in result.event.what_we_saw["checked_containers"]]
    assert reason_codes[0] == Msg.FILLER_VERIFIED.reason
    assert reason_codes[1] != Msg.FILLER_VERIFIED.reason


@pytest.mark.asyncio
async def test_killed_bundle_decides_even_when_a_healthy_sibling_comes_last():
    """Shadow: every verdict passes, so a positional pick would emit the healthy sibling's event
    and the kill would never surface — the node must be judged by its worst container."""
    backend_client = _killed_filler_backend()
    # A is removed, B is alive and LAST in the list.
    ssh_client = FillerSSHClient(dead_containers={_BUNDLE_A})
    ctx = _filler_context(
        backend_client=backend_client, ssh_client=ssh_client, filler_containers=[_BUNDLE_A, _BUNDLE_B]
    )

    result = await _run_filler_check(ctx, enforcement=False)

    assert result.event.reason_code == Msg.FILLER_KILLED.reason
    assert result.event.what_we_saw["filler_container"] == _BUNDLE_A
    assert result.passed is True  # shadow never withholds incentive


@pytest.mark.asyncio
async def test_unreachable_bundle_does_not_mask_a_kill_behind_it():
    """Enforcement: an SSH failure on an earlier bundle must not decide the node while a real kill
    sits behind it — otherwise the node is penalised under the wrong reason and the kill is lost."""
    backend_client = _killed_filler_backend()
    ssh_client = FillerSSHClient(unreachable_containers={_BUNDLE_A}, dead_containers={_BUNDLE_B})
    ctx = _filler_context(
        backend_client=backend_client, ssh_client=ssh_client, filler_containers=[_BUNDLE_A, _BUNDLE_B]
    )

    result = await _run_filler_check(ctx, enforcement=True)

    assert result.event.reason_code == Msg.FILLER_KILLED.reason
    assert result.event.what_we_saw["filler_container"] == _BUNDLE_B
    assert result.passed is False
    assert [entry["reason_code"] for entry in result.event.what_we_saw["checked_containers"]] == [
        Msg.FILLER_TRANSPORT_UNREACHABLE.reason,
        Msg.FILLER_KILLED.reason,
    ]


# ---------------------------------------------------------------------------
# DAH-2614: the backend's probe must name the cards this cycle's scrape saw,
# instead of `--gpus all`, which never names one.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rental_verification_sends_scraped_gpu_uuids():
    backend_client = DummyBackendClient(
        response=ExecutorHealthCheckResponse(success=True, error=None, details={})
    )
    services = build_services(backend=backend_client, container_cleanup=ContainerCleanup())
    state = build_state(
        specs={
            "verified_ports": [9001],
            "gpu": {
                "count": 2,
                "details": [
                    {"uuid": "GPU-f2bfa67f-5281-aabc-aa90-c91764f90d17", "name": "B200"},
                    {"uuid": "GPU-f2bfa67f-5281-aabc-aa90-c91764f90d18", "name": "B200"},
                ],
            },
        }
    )

    from tests.helpers import make_context

    ctx = make_context(services=services, state=state)

    with patch("neurons.validators.src.services.task.checks.rental_verification.settings") as mock_settings:
        mock_settings.SKIP_RENTAL_VERIFICATION = False
        result = await RentalVerificationCheck().run(ctx)

    assert result.passed is True
    assert backend_client.called_with["gpu_uuids"] == [
        "GPU-f2bfa67f-5281-aabc-aa90-c91764f90d17",
        "GPU-f2bfa67f-5281-aabc-aa90-c91764f90d18",
    ]


@pytest.mark.asyncio
async def test_rental_verification_sends_empty_list_when_the_scrape_has_no_gpu_details():
    # The backend falls back to `--gpus all`, i.e. exactly the behaviour before this change.
    backend_client = DummyBackendClient(
        response=ExecutorHealthCheckResponse(success=True, error=None, details={})
    )
    services = build_services(backend=backend_client, container_cleanup=ContainerCleanup())
    state = build_state(specs={"verified_ports": [9001]})

    from tests.helpers import make_context

    ctx = make_context(services=services, state=state)

    with patch("neurons.validators.src.services.task.checks.rental_verification.settings") as mock_settings:
        mock_settings.SKIP_RENTAL_VERIFICATION = False
        result = await RentalVerificationCheck().run(ctx)

    assert result.passed is True
    assert backend_client.called_with["gpu_uuids"] == []


@pytest.mark.asyncio
async def test_rental_verification_skips_gpu_details_without_a_uuid():
    backend_client = DummyBackendClient(
        response=ExecutorHealthCheckResponse(success=True, error=None, details={})
    )
    services = build_services(backend=backend_client, container_cleanup=ContainerCleanup())
    state = build_state(
        specs={
            "verified_ports": [9001],
            "gpu": {
                "count": 3,
                "details": [
                    {"uuid": "GPU-f2bfa67f-5281-aabc-aa90-c91764f90d17"},
                    {"uuid": ""},
                    {"name": "B200"},
                ],
            },
        }
    )

    from tests.helpers import make_context

    ctx = make_context(services=services, state=state)

    with patch("neurons.validators.src.services.task.checks.rental_verification.settings") as mock_settings:
        mock_settings.SKIP_RENTAL_VERIFICATION = False
        result = await RentalVerificationCheck().run(ctx)

    assert result.passed is True
    assert backend_client.called_with["gpu_uuids"] == ["GPU-f2bfa67f-5281-aabc-aa90-c91764f90d17"]


# --- DAH-2671 item 2b: send the advertised CPU count, quarantine a daemon-rejected count ---

_CPU_QUOTA_STDERR = (
    "docker: Error response from daemon: Range of CPUs is from 0.01 to 20.00, "
    "as there are only 20 CPUs available."
)


class _OnlineCpuSSH(_RecordingSSH):
    """An executor reporting `online` logical CPUs in `/proc/cpuinfo` and `present` kernel-present
    cores in sysfs; every other command is a no-op. `present` defaults to `online`."""

    def __init__(self, online: int, present: int | None = None):
        super().__init__()
        self.online = online
        self.present = online if present is None else present

    async def run(self, cmd):
        res = await super().run(cmd)
        if "processor /proc/cpuinfo" in cmd:
            res.stdout = f"{self.online}\n"
        if "/sys/devices/system/cpu/present" in cmd:
            res.stdout = f"0-{self.present - 1}\n"
        return res


def _cpu_quota_response() -> ExecutorHealthCheckResponse:
    return ExecutorHealthCheckResponse(
        success=False,
        error=_CPU_QUOTA_STDERR,
        details={"docker_stderr": _CPU_QUOTA_STDERR},
        reason_code=CPU_QUOTA_EXCEEDS_HOST_REASON,
    )


@pytest.mark.asyncio
async def test_cpu_quota_verdict_is_observe_only_in_shadow():
    """CHECK on, ENFORCEMENT off: the daemon-rejected count is logged but the score is untouched."""
    backend_client = DummyBackendClient(response=_cpu_quota_response())
    services = build_services(backend=backend_client, container_cleanup=ContainerCleanup())
    state = build_state(specs={"verified_ports": [8080], "cpu": {"count": 176}})

    from tests.helpers import make_context
    ctx = make_context(services=services, state=state, ssh=_RecordingSSH())

    with patch("neurons.validators.src.services.task.checks.rental_verification.settings") as mock_settings:
        mock_settings.SKIP_RENTAL_VERIFICATION = False
        mock_settings.RENTAL_CPU_LIMIT_CHECK_ENABLED = True
        mock_settings.RENTAL_CPU_LIMIT_ENFORCEMENT_ENABLED = False
        result = await RentalVerificationCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == CPU_QUOTA_EXCEEDS_HOST_REASON
    assert result.updates == {}
    assert result.event.what_we_saw["advertised_cpu_count"] == 176
    assert "Range of CPUs is from" in result.event.what_we_saw["stderr"]


@pytest.mark.asyncio
async def test_cpu_quota_verdict_zeroes_score_under_enforcement():
    backend_client = DummyBackendClient(response=_cpu_quota_response())
    services = build_services(backend=backend_client, container_cleanup=ContainerCleanup())
    state = build_state(specs={"verified_ports": [8080], "cpu": {"count": 176}})

    from tests.helpers import make_context
    ctx = make_context(services=services, state=state, ssh=_RecordingSSH())

    with patch("neurons.validators.src.services.task.checks.rental_verification.settings") as mock_settings:
        mock_settings.SKIP_RENTAL_VERIFICATION = False
        mock_settings.RENTAL_CPU_LIMIT_CHECK_ENABLED = True
        mock_settings.RENTAL_CPU_LIMIT_ENFORCEMENT_ENABLED = True
        result = await RentalVerificationCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == CPU_QUOTA_EXCEEDS_HOST_REASON
    assert result.updates["score"] == 0.0
    assert result.updates["job_score"] == 0.0
    assert "clear_verified_job_info" not in result.updates


@pytest.mark.asyncio
async def test_advertised_cpu_count_is_sent_to_the_backend():
    backend_client = DummyBackendClient(
        response=ExecutorHealthCheckResponse(success=True, error=None, details={})
    )
    services = build_services(backend=backend_client, container_cleanup=ContainerCleanup())
    state = build_state(specs={"verified_ports": [8080], "cpu": {"count": 44}})

    from tests.helpers import make_context
    ctx = make_context(services=services, state=state, ssh=_RecordingSSH())

    with patch("neurons.validators.src.services.task.checks.rental_verification.settings") as mock_settings:
        mock_settings.SKIP_RENTAL_VERIFICATION = False
        mock_settings.RENTAL_CPU_LIMIT_CHECK_ENABLED = True
        await RentalVerificationCheck().run(ctx)

    assert backend_client.called_with["cpu_count"] == 44


@pytest.mark.asyncio
async def test_fake_present_population_does_not_cap_the_advertised_count():
    """Bypass regression: a spoofer faking `present` == advertised (176) while /proc/cpuinfo stays
    real (44) must NOT get the advertised count capped to the online population — a host-read cap
    would turn the lie into a quota the daemon happily grants. The count goes out AS-IS and the
    daemon/backend classifier judges it."""
    backend_client = DummyBackendClient(
        response=ExecutorHealthCheckResponse(success=True, error=None, details={})
    )
    services = build_services(backend=backend_client, container_cleanup=ContainerCleanup())
    state = build_state(specs={"verified_ports": [8080], "cpu": {"count": 176}})

    from tests.helpers import make_context
    ctx = make_context(services=services, state=state, ssh=_OnlineCpuSSH(44, present=176))

    with patch("neurons.validators.src.services.task.checks.rental_verification.settings") as mock_settings:
        mock_settings.SKIP_RENTAL_VERIFICATION = False
        mock_settings.RENTAL_CPU_LIMIT_CHECK_ENABLED = True
        await RentalVerificationCheck().run(ctx)

    assert backend_client.called_with["cpu_count"] == 176


@pytest.mark.asyncio
async def test_cpu_count_is_not_sent_when_the_check_is_disabled():
    backend_client = DummyBackendClient(
        response=ExecutorHealthCheckResponse(success=True, error=None, details={})
    )
    services = build_services(backend=backend_client, container_cleanup=ContainerCleanup())
    state = build_state(specs={"verified_ports": [8080], "cpu": {"count": 44}})

    from tests.helpers import make_context
    ctx = make_context(services=services, state=state)

    with patch("neurons.validators.src.services.task.checks.rental_verification.settings") as mock_settings:
        mock_settings.SKIP_RENTAL_VERIFICATION = False
        mock_settings.RENTAL_CPU_LIMIT_CHECK_ENABLED = False
        await RentalVerificationCheck().run(ctx)

    assert backend_client.called_with["cpu_count"] is None


@pytest.mark.asyncio
async def test_cpu_count_is_not_sent_for_a_tdx_host():
    """A real rent skips `--cpus` on a confidential VM (docker_service.py), so verification does too."""
    backend_client = DummyBackendClient(
        response=ExecutorHealthCheckResponse(success=True, error=None, details={})
    )
    services = build_services(backend=backend_client, container_cleanup=ContainerCleanup())
    state = build_state(specs={"verified_ports": [8080], "cpu": {"count": 44}})

    from tests.helpers import default_executor, make_context
    executor = default_executor()
    executor.tdx_quote = '{"quote": "..."}'
    ctx = make_context(executor=executor, services=services, state=state)

    with patch("neurons.validators.src.services.task.checks.rental_verification.settings") as mock_settings:
        mock_settings.SKIP_RENTAL_VERIFICATION = False
        mock_settings.RENTAL_CPU_LIMIT_CHECK_ENABLED = True
        await RentalVerificationCheck().run(ctx)

    assert backend_client.called_with["cpu_count"] is None


# DAH-2703: a host reaper that removes the filler seconds after `docker run` leaves no container
# to probe, so the liveness check above never runs. The backend reports the kill streak instead.


@pytest.mark.asyncio
async def test_rental_verification_filler_create_kill_shadow_mode_only_observes():
    """Shadow mode observes and then verifies as usual — it must never replace the normal check."""
    backend_client = DummyBackendClient(
        response=ExecutorHealthCheckResponse(success=True, error=None, details={})
    )
    ctx = _filler_context(
        backend_client=backend_client,
        ssh_client=FillerSSHClient(),
        running_fillers=False,
        create_killed=True,
    )

    with patch("neurons.validators.src.services.task.checks.rental_verification.logger") as mock_logger:
        result = await _run_filler_check(ctx, enforcement=False)

    assert result.passed is True
    assert result.event.reason_code == Msg.VERIFIED.reason
    # The node still went through the backend health check it would have had without this feature.
    assert backend_client.called_with is not None
    assert any(
        Msg.FILLER_KILLED_AT_CREATE.reason in str(call.args[0].extra)
        for call in mock_logger.warning.call_args_list
    )


@pytest.mark.asyncio
async def test_rental_verification_filler_create_kill_enforcement_fails():
    """Enforcement mode: the streak fails the fatal check -> no unrented incentive."""
    backend_client = DummyBackendClient(
        response=ExecutorHealthCheckResponse(success=True, error=None, details={})
    )
    ctx = _filler_context(
        backend_client=backend_client,
        ssh_client=FillerSSHClient(),
        running_fillers=False,
        create_killed=True,
    )

    result = await _run_filler_check(ctx, enforcement=True)

    assert result.passed is False
    assert result.event.reason_code == Msg.FILLER_KILLED_AT_CREATE.reason
    assert result.event.what_we_saw["enforced"] is True
    # The node is never contacted for a health check once the streak decides the verdict.
    assert backend_client.called_with is None


@pytest.mark.asyncio
async def test_rental_verification_filler_create_kill_ignored_when_check_disabled():
    """CHECK_ENABLED off is the master kill switch here too: normal verification runs."""
    backend_client = DummyBackendClient(
        response=ExecutorHealthCheckResponse(success=True, error=None, details={})
    )
    ctx = _filler_context(
        backend_client=backend_client,
        ssh_client=FillerSSHClient(),
        running_fillers=False,
        create_killed=True,
    )

    result = await _run_filler_check(ctx, check_enabled=False, enforcement=True)

    assert result.passed is True
    assert result.event.reason_code == Msg.VERIFIED.reason


@pytest.mark.asyncio
async def test_rental_verification_filler_create_kill_ignored_for_a_customer_rental():
    """A paying customer's pod is never punished for the node's default-job history."""
    backend_client = DummyBackendClient(
        response=ExecutorHealthCheckResponse(success=True, error=None, details={})
    )
    ctx = _filler_context(
        backend_client=backend_client,
        ssh_client=FillerSSHClient(),
        running_fillers=False,
        create_killed=True,
        rented_pods=[RentedPod(pod_id="pod-1", container_name="pod_1")],
    )

    result = await _run_filler_check(ctx, enforcement=True)

    assert result.passed is True
    assert result.event.reason_code == Msg.VERIFIED.reason


@pytest.mark.asyncio
async def test_rental_verification_create_kill_is_not_masked_by_a_surviving_filler():
    """A split node that lets one bundle run while killing the others is still killing them."""
    backend_client = DummyBackendClient(
        response=ExecutorHealthCheckResponse(success=True, error=None, details={})
    )
    ctx = _filler_context(
        backend_client=backend_client,
        ssh_client=FillerSSHClient(running=True),
        create_killed=True,
    )

    result = await _run_filler_check(ctx, enforcement=True)

    assert result.passed is False
    assert result.event.reason_code == Msg.FILLER_KILLED_AT_CREATE.reason


@pytest.mark.asyncio
async def test_filler_that_exited_on_the_host_is_not_a_kill():
    """Container still on the host, exit code 255: the node crashed the job, it did not remove it."""
    backend_client = _killed_filler_backend()
    ssh_client = FillerSSHClient(running=False, removed_from_host=False)
    ctx = _filler_context(backend_client=backend_client, ssh_client=ssh_client)

    result = await _run_filler_check(ctx, enforcement=True)

    assert result.passed is True
    assert result.event.reason_code == Msg.FILLER_CONTAINER_EXITED.reason
    assert result.event.what_we_saw["container_exit_code"] == 255
    assert result.event.what_we_saw["container_status"] == "exited"


@pytest.mark.asyncio
async def test_filler_terminated_by_a_signal_is_a_kill():
    """Exit 143 = SIGTERM, i.e. `docker stop`. Fillers carry `restart: unless-stopped`, so a
    container that is still exited was stopped deliberately — the restart policy would have brought
    back anything else. Prod: all 4 such runs in 7 days came from the one host with a container
    auto-killer."""
    backend_client = _killed_filler_backend()
    ssh_client = FillerSSHClient(running=False, removed_from_host=False, exit_code=143)
    ctx = _filler_context(backend_client=backend_client, ssh_client=ssh_client)

    result = await _run_filler_check(ctx, enforcement=True)

    assert result.passed is False
    assert result.event.reason_code == Msg.FILLER_STOPPED_BY_HOST.reason
    assert result.event.what_we_saw["container_exit_code"] == 143


@pytest.mark.asyncio
async def test_filler_killed_by_sigkill_is_a_kill():
    """Exit 137 = SIGKILL from outside the container."""
    backend_client = _killed_filler_backend()
    ssh_client = FillerSSHClient(running=False, removed_from_host=False, exit_code=137)
    ctx = _filler_context(backend_client=backend_client, ssh_client=ssh_client)

    result = await _run_filler_check(ctx, enforcement=True)

    assert result.passed is False
    assert result.event.reason_code == Msg.FILLER_STOPPED_BY_HOST.reason


@pytest.mark.asyncio
async def test_out_of_memory_kill_is_not_a_host_kill():
    """Same exit 137, but the kernel reclaimed memory — the host did not choose to stop the job."""
    backend_client = _killed_filler_backend()
    ssh_client = FillerSSHClient(
        running=False, removed_from_host=False, exit_code=137, oom_killed=True
    )
    ctx = _filler_context(backend_client=backend_client, ssh_client=ssh_client)

    result = await _run_filler_check(ctx, enforcement=True)

    assert result.passed is True
    assert result.event.reason_code == Msg.FILLER_CONTAINER_EXITED.reason


@pytest.mark.asyncio
async def test_signal_killed_bundle_outranks_a_healthy_sibling_on_a_split_node():
    """A stopped bundle must decide the node even when its siblings are fine (DAH-2465)."""
    backend_client = _killed_filler_backend()
    healthy_container = "filler_11111111-2222-3333-4444-555555555555"
    stopped_container = "filler_99999999-8888-7777-6666-555555555555"
    ssh_client = FillerSSHClient(
        dead_containers={stopped_container}, removed_from_host=False, exit_code=143
    )
    ctx = _filler_context(
        backend_client=backend_client,
        ssh_client=ssh_client,
        filler_containers=[healthy_container, stopped_container],
    )

    result = await _run_filler_check(ctx, enforcement=True)

    assert result.passed is False
    assert result.event.reason_code == Msg.FILLER_STOPPED_BY_HOST.reason


@pytest.mark.asyncio
async def test_container_state_without_evidence_is_unknown_not_exited():
    """Docker answered nothing useful: we cannot claim the container is still there either."""
    backend_client = _killed_filler_backend()
    ssh_client = FillerSSHClient(running=False)
    ctx = _filler_context(backend_client=backend_client, ssh_client=ssh_client)

    with patch(
        "neurons.validators.src.services.task.checks.rental_verification.collect_container_death_diagnostics",
        return_value=ContainerDeathDiagnostics(capture_error="inspect: Cannot connect to the Docker daemon"),
    ):
        result = await _run_filler_check(ctx, enforcement=True)

    assert result.passed is True
    assert result.event.reason_code == Msg.FILLER_STATE_UNKNOWN.reason


@pytest.mark.asyncio
async def test_exited_bundle_outranks_a_healthy_sibling_on_a_split_node():
    """DAH-2465 split node: the bundle that lost its job decides the node's logged verdict."""
    backend_client = _killed_filler_backend()
    healthy_container = "filler_11111111-2222-3333-4444-555555555555"
    exited_container = "filler_99999999-8888-7777-6666-555555555555"
    ssh_client = FillerSSHClient(dead_containers={exited_container}, removed_from_host=False)
    ctx = _filler_context(
        backend_client=backend_client,
        ssh_client=ssh_client,
        filler_containers=[healthy_container, exited_container],
    )

    result = await _run_filler_check(ctx, enforcement=True)

    assert result.passed is True
    assert result.event.reason_code == Msg.FILLER_CONTAINER_EXITED.reason


@pytest.mark.asyncio
async def test_container_running_again_by_the_time_we_inspect_is_not_a_death():
    """Filler containers run with `restart: unless-stopped`, so docker can bring one back
    between the ps probe and the inspect. A running container is not an exit and not a kill."""
    backend_client = _killed_filler_backend()
    ssh_client = FillerSSHClient(running=False)
    ctx = _filler_context(backend_client=backend_client, ssh_client=ssh_client)

    with patch(
        "neurons.validators.src.services.task.checks.rental_verification.collect_container_death_diagnostics",
        return_value=ContainerDeathDiagnostics(status="running"),
    ):
        result = await _run_filler_check(ctx, enforcement=True)

    assert result.passed is True
    assert result.event.reason_code == Msg.FILLER_STATE_UNKNOWN.reason


@pytest.mark.asyncio
async def test_container_caught_mid_removal_is_still_a_kill():
    """`docker rm` leaves State.Status="removing" for a moment — that is the offence in progress,
    not a container that exited (observed in prod on the ISSUE-050 class)."""
    backend_client = _killed_filler_backend()
    ssh_client = FillerSSHClient(running=False)
    ctx = _filler_context(backend_client=backend_client, ssh_client=ssh_client)

    with patch(
        "neurons.validators.src.services.task.checks.rental_verification.collect_container_death_diagnostics",
        return_value=ContainerDeathDiagnostics(status="removing"),
    ):
        result = await _run_filler_check(ctx, enforcement=True)

    assert result.passed is False
    assert result.event.reason_code == Msg.FILLER_KILLED.reason


@pytest.mark.parametrize("docker_status", ["created", "restarting", "paused", "dead", None])
@pytest.mark.asyncio
async def test_states_that_do_not_prove_a_death_are_never_punished(docker_status):
    """Only a recorded exit is an exit; every other state stays unproven and unpunished."""
    backend_client = _killed_filler_backend()
    ctx = _filler_context(backend_client=backend_client, ssh_client=FillerSSHClient(running=False))

    with patch(
        "neurons.validators.src.services.task.checks.rental_verification.collect_container_death_diagnostics",
        return_value=ContainerDeathDiagnostics(status=docker_status),
    ):
        result = await _run_filler_check(ctx, enforcement=True)

    assert result.passed is True
    assert result.event.reason_code == Msg.FILLER_STATE_UNKNOWN.reason
