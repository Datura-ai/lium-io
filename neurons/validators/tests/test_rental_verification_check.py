from datetime import datetime, timedelta

import asyncssh
import pytest
from unittest.mock import Mock, patch

from neurons.validators.src.protocol.vc_protocol.compute_requests import (
    GPU_RUNTIME_DEVICE_FAULT_REASON,
    GPU_RUNTIME_NVML_MISMATCH_REASON,
    ExecutorHealthCheckResponse,
    FillerRunActiveResponse,
)
from protocol.vc_protocol.compute_requests import RentedExecutor, RentedExecutorsResponse, RentedPod
from neurons.validators.src.services.container_cleanup import ContainerCleanup
from neurons.validators.src.services.task.checks.rental_verification import RentalVerificationCheck
from neurons.validators.src.services.task.messages import RentalVerificationMessages as Msg
from neurons.validators.src.services.task.pipeline import CheckResult, Context

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
    ):
        self.called_with = {
            "miner_address": miner_address,
            "miner_port": miner_port,
            "miner_hotkey": miner_hotkey,
            "container_port": container_port,
            "executor_id": executor_id,
            "rental_in_progress": rental_in_progress,
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
async def test_rental_verification_nvml_mismatch_clears_verified_job_info():
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
    assert result.updates["clear_verified_job_info"] is True
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
    assert result.updates["clear_verified_job_info"] is True
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
    ):
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
        return result


def _filler_context(
    *,
    backend_client: DummyBackendClient,
    ssh_client: FillerSSHClient,
    filler_container: str = "filler_11111111-2222-3333-4444-555555555555",
    filler_containers: list[str] | None = None,
) -> Context:
    # filler_containers = a GPU-split node's full bundle list (DAH-2465); the legacy single map is
    # still populated so the protocol's fallback path stays covered.
    all_containers: list[str] = filler_containers or [filler_container]
    services = build_services(backend=backend_client, container_cleanup=ContainerCleanup())
    state = build_state(
        specs={"verified_ports": [8080]},
        rented_data=RentedExecutorsResponse(
            executors={},
            filler_containers_by_executor={"executor-123": all_containers[0]},
            all_filler_containers_by_executor={"executor-123": all_containers},
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
    ssh_client = FillerSSHClient(running=False)
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
