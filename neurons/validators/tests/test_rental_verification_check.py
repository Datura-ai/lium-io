import pytest
from unittest.mock import patch

from neurons.validators.src.protocol.vc_protocol.compute_requests import (
    GPU_RUNTIME_NVML_MISMATCH_REASON,
    ExecutorHealthCheckResponse,
)
from neurons.validators.src.protocol.vc_protocol.validator_requests import ResetVerifiedJobReason
from protocol.vc_protocol.compute_requests import RentedExecutorsResponse
from neurons.validators.src.services.container_cleanup import ContainerCleanup
from neurons.validators.src.services.task.checks.rental_verification import RentalVerificationCheck
from neurons.validators.src.services.task.messages import RentalVerificationMessages as Msg

from tests.helpers import build_services, build_state


class DummyBackendClient:
    def __init__(self, *, response: ExecutorHealthCheckResponse | None):
        self.response = response
        self.called_with: dict | None = None

    async def check_executor_health(
        self,
        miner_address: str,
        miner_port: int,
        miner_hotkey: str,
        container_port: int,
        executor_id: str | None = None,
    ):
        self.called_with = {
            "miner_address": miner_address,
            "miner_port": miner_port,
            "miner_hotkey": miner_hotkey,
            "container_port": container_port,
            "executor_id": executor_id,
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
    assert (
        result.updates["clear_verified_job_reason"]
        == ResetVerifiedJobReason.GPU_RUNTIME_NVML_MISMATCH.value
    )


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


@pytest.mark.asyncio
async def test_rental_verification_skips_filler_only_executor():
    backend_client = DummyBackendClient(
        response=ExecutorHealthCheckResponse(success=True, error=None, details={})
    )
    services = build_services(backend=backend_client, container_cleanup=ContainerCleanup())
    state = build_state(
        specs={"verified_ports": [8080]},
        rented_data=RentedExecutorsResponse(
            executors={},
            filler_containers_by_executor={"executor-123": "filler_active"},
        ),
    )

    from tests.helpers import make_context
    ctx = make_context(services=services, state=state)

    with patch("neurons.validators.src.services.task.checks.rental_verification.settings") as mock_settings:
        mock_settings.SKIP_RENTAL_VERIFICATION = False
        result = await RentalVerificationCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.SKIPPED.reason
    assert backend_client.called_with is None
