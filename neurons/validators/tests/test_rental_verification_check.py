from datetime import datetime, timedelta

import asyncssh
import pytest
from unittest.mock import Mock, patch

from neurons.validators.src.protocol.vc_protocol.compute_requests import (
    GPU_RUNTIME_NVML_MISMATCH_REASON,
    ExecutorHealthCheckResponse,
    FillerRunActiveResponse,
)
from protocol.vc_protocol.compute_requests import RentedExecutor, RentedExecutorsResponse, RentedPod
from neurons.validators.src.services.container_cleanup import ContainerCleanup
from neurons.validators.src.services.task.checks.rental_verification import RentalVerificationCheck
from neurons.validators.src.services.task.messages import RentalVerificationMessages as Msg
from neurons.validators.src.services.task.pipeline import CheckResult, Context
from core.docker_utils import ContainerDeathDiagnostics

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

    async def get_filler_run_active(self, filler_run_id: str) -> FillerRunActiveResponse | None:
        self.filler_run_active_calls.append(filler_run_id)
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
    ):
        self.running = running
        self.exit_status = exit_status
        self.raise_on_run = raise_on_run
        self.commands: list[str] = []

    async def run(self, command: str) -> Mock:
        self.commands.append(command)
        if self.raise_on_run is not None:
            raise self.raise_on_run
        result = Mock()
        result.exit_status = self.exit_status
        result.stdout = "container_id_123" if self.running and "docker ps" in command else ""
        result.stderr = ""
        return result


def _filler_context(
    *,
    backend_client: DummyBackendClient,
    ssh_client: FillerSSHClient,
    filler_container: str = "filler_11111111-2222-3333-4444-555555555555",
    redis_service: "FakeStrikeRedis | None" = None,
) -> Context:
    services = build_services(backend=backend_client, container_cleanup=ContainerCleanup(), redis=redis_service)
    state = build_state(
        specs={"verified_ports": [8080]},
        rented_data=RentedExecutorsResponse(
            executors={},
            filler_containers_by_executor={"executor-123": filler_container},
        ),
    )
    from tests.helpers import make_context

    return make_context(services=services, state=state, ssh=ssh_client)


async def _run_filler_check(ctx: Context, *, check_enabled: bool = True, enforcement: bool = False) -> CheckResult:
    with patch("neurons.validators.src.services.task.checks.rental_verification.settings") as mock_settings:
        mock_settings.SKIP_RENTAL_VERIFICATION = False
        mock_settings.FILLER_LIVENESS_CHECK_ENABLED = check_enabled
        mock_settings.FILLER_LIVENESS_ENFORCEMENT_ENABLED = enforcement
        mock_settings.FILLER_KILL_STRIKE_THRESHOLD = 2
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


class FakeStrikeRedis:
    """Fake RedisService for the kill-strike path."""

    def __init__(self, strikes_to_return: int, raises: bool = False):
        self.strikes_to_return = strikes_to_return
        self.raises = raises
        self.calls: list[tuple[str, str, int]] = []

    async def register_filler_kill_strike(self, executor_uuid: str, filler_run_id: str, ttl_seconds: int) -> int:
        self.calls.append((executor_uuid, filler_run_id, ttl_seconds))
        if self.raises:
            raise ConnectionError("redis down")
        return self.strikes_to_return


REMOVED_DIAGNOSTICS = ContainerDeathDiagnostics(
    capture_error="inspect: error: no such object: filler_x",
    logs_tail="Error response from daemon: No such container: filler_x",
)
STOPPED_DIAGNOSTICS = ContainerDeathDiagnostics(
    status="exited", exit_code=143, oom_killed=False,
    started_at="2026-07-20T06:00:00Z", finished_at="2026-07-20T09:00:00Z",
)
CRASHED_DIAGNOSTICS = ContainerDeathDiagnostics(
    status="exited", exit_code=1, oom_killed=False,
    started_at="2026-07-20T06:00:00Z", finished_at="2026-07-20T06:01:00Z",
)
OOM_DIAGNOSTICS = ContainerDeathDiagnostics(status="exited", exit_code=137, oom_killed=True)
HOST_REBOOT_DIAGNOSTICS = ContainerDeathDiagnostics(
    status="exited", exit_code=143, oom_killed=False,
    finished_at="2026-07-20T09:00:00Z",
    host_context={"executor_container_started_at": "2026-07-20T09:00:20Z"},
)
NEVER_STARTED_DIAGNOSTICS = ContainerDeathDiagnostics(status="created", exit_code=0)


def _patch_diagnostics(monkeypatch, diagnostics: ContainerDeathDiagnostics) -> None:
    async def fake_collect(ssh_client, container_name):
        return diagnostics
    monkeypatch.setattr(
        "neurons.validators.src.services.task.checks.rental_verification.collect_container_death_diagnostics",
        fake_collect,
    )


@pytest.mark.asyncio
async def test_rental_verification_filler_removed_shadow_mode_passes_but_logs(monkeypatch):
    """Shadow mode: a REMOVED filler (external rm) is logged with enforced=False, incentive untouched."""
    backend_client = _killed_filler_backend()
    ssh_client = FillerSSHClient(running=False)
    ctx = _filler_context(backend_client=backend_client, ssh_client=ssh_client)
    _patch_diagnostics(monkeypatch, REMOVED_DIAGNOSTICS)

    result = await _run_filler_check(ctx, enforcement=False)

    assert result.passed is True
    # A removed container is "missing" with no reliable one-shot kill signal (prod shadow: every
    # removed case was a single unreconciled zombie run), so it is strike-gated like STOPPED.
    assert result.event.reason_code == Msg.FILLER_KILL_SUSPECTED.reason
    assert result.event.what_we_saw["enforced"] is False
    assert result.event.what_we_saw["death_kind"] == "removed"
    assert result.event.what_we_saw["kill_strikes"] is None
    assert backend_client.filler_run_active_calls == ["11111111-2222-3333-4444-555555555555"]


@pytest.mark.asyncio
async def test_rental_verification_filler_removed_first_strike_is_suspected_not_punished(monkeypatch):
    """A single removal is usually a zombie run (host reboot / watchtower recreation): strike 1, passed."""
    backend_client = _killed_filler_backend()
    ssh_client = FillerSSHClient(running=False)
    strike_redis = FakeStrikeRedis(strikes_to_return=1)
    ctx = _filler_context(backend_client=backend_client, ssh_client=ssh_client, redis_service=strike_redis)
    _patch_diagnostics(monkeypatch, REMOVED_DIAGNOSTICS)

    result = await _run_filler_check(ctx, enforcement=True)

    assert result.passed is True
    assert result.event.reason_code == Msg.FILLER_KILL_SUSPECTED.reason
    assert result.event.what_we_saw["death_kind"] == "removed"
    assert result.event.what_we_saw["kill_strikes"] == 1


@pytest.mark.asyncio
async def test_rental_verification_filler_removed_second_strike_is_punished(monkeypatch):
    """A repeat removal within the window -> treated as a targeted kill, incentive withheld."""
    backend_client = _killed_filler_backend()
    ssh_client = FillerSSHClient(running=False)
    strike_redis = FakeStrikeRedis(strikes_to_return=2)
    ctx = _filler_context(backend_client=backend_client, ssh_client=ssh_client, redis_service=strike_redis)
    _patch_diagnostics(monkeypatch, REMOVED_DIAGNOSTICS)

    result = await _run_filler_check(ctx, enforcement=True)

    assert result.passed is False
    assert result.event.reason_code == Msg.FILLER_KILLED.reason
    assert result.event.what_we_saw["death_kind"] == "removed"
    assert result.event.what_we_saw["kill_strikes"] == 2


@pytest.mark.asyncio
async def test_rental_verification_filler_stopped_first_strike_is_suspected_not_punished(monkeypatch):
    """A lone SIGTERM stop could be the worker itself: strike 1 -> logged, passed even in enforcement."""
    backend_client = _killed_filler_backend()
    ssh_client = FillerSSHClient(running=False)
    strike_redis = FakeStrikeRedis(strikes_to_return=1)
    ctx = _filler_context(backend_client=backend_client, ssh_client=ssh_client, redis_service=strike_redis)
    _patch_diagnostics(monkeypatch, STOPPED_DIAGNOSTICS)

    result = await _run_filler_check(ctx, enforcement=True)

    assert result.passed is True
    assert result.event.reason_code == Msg.FILLER_KILL_SUSPECTED.reason
    assert result.event.what_we_saw["kill_strikes"] == 1
    assert result.event.what_we_saw["enforced"] is True
    assert strike_redis.calls[0][1] == "11111111-2222-3333-4444-555555555555"


@pytest.mark.asyncio
async def test_rental_verification_filler_stopped_second_strike_is_punished(monkeypatch):
    """Second stop incident within the window -> treated as an external kill."""
    backend_client = _killed_filler_backend()
    ssh_client = FillerSSHClient(running=False)
    strike_redis = FakeStrikeRedis(strikes_to_return=2)
    ctx = _filler_context(backend_client=backend_client, ssh_client=ssh_client, redis_service=strike_redis)
    _patch_diagnostics(monkeypatch, STOPPED_DIAGNOSTICS)

    result = await _run_filler_check(ctx, enforcement=True)

    assert result.passed is False
    assert result.event.reason_code == Msg.FILLER_KILLED.reason
    assert result.event.what_we_saw["death_kind"] == "stopped"
    assert result.event.what_we_saw["kill_strikes"] == 2
    assert result.event.what_we_saw["kill_timing"] == "after_running"


@pytest.mark.asyncio
async def test_rental_verification_filler_stopped_shadow_does_not_register_strike(monkeypatch):
    """Shadow must not count strikes, so the first ENFORCED incident always gets the grace strike."""
    backend_client = _killed_filler_backend()
    ssh_client = FillerSSHClient(running=False)
    strike_redis = FakeStrikeRedis(strikes_to_return=99)
    ctx = _filler_context(backend_client=backend_client, ssh_client=ssh_client, redis_service=strike_redis)
    _patch_diagnostics(monkeypatch, STOPPED_DIAGNOSTICS)

    result = await _run_filler_check(ctx, enforcement=False)

    assert result.passed is True
    assert result.event.reason_code == Msg.FILLER_KILL_SUSPECTED.reason
    assert result.event.what_we_saw["kill_strikes"] is None
    assert strike_redis.calls == []


@pytest.mark.asyncio
async def test_rental_verification_host_reboot_never_punished(monkeypatch):
    """A SIGTERM that coincided with a host/executor restart is not a targeted kill: no penalty, no strike."""
    backend_client = _killed_filler_backend()
    ssh_client = FillerSSHClient(running=False)
    strike_redis = FakeStrikeRedis(strikes_to_return=99)
    ctx = _filler_context(backend_client=backend_client, ssh_client=ssh_client, redis_service=strike_redis)
    _patch_diagnostics(monkeypatch, HOST_REBOOT_DIAGNOSTICS)

    result = await _run_filler_check(ctx, enforcement=True)

    assert result.passed is True
    assert result.event.reason_code == Msg.FILLER_CRASHED.reason
    assert result.event.what_we_saw["death_kind"] == "host_reboot"
    assert result.event.what_we_saw["enforced"] is True
    assert strike_redis.calls == []


@pytest.mark.asyncio
async def test_rental_verification_filler_stopped_without_redis_fails_open(monkeypatch):
    """No strike storage is not evidence of a repeat kill: fail open, never penalize on our outage."""
    backend_client = _killed_filler_backend()
    ssh_client = FillerSSHClient(running=False)
    ctx = _filler_context(backend_client=backend_client, ssh_client=ssh_client)
    _patch_diagnostics(monkeypatch, STOPPED_DIAGNOSTICS)

    result = await _run_filler_check(ctx, enforcement=True)

    assert result.passed is True
    assert result.event.reason_code == Msg.FILLER_KILL_SUSPECTED.reason
    assert result.event.what_we_saw["kill_strikes"] is None


@pytest.mark.asyncio
async def test_rental_verification_filler_stopped_redis_error_fails_open(monkeypatch):
    """Redis raising mid-strike is treated the same as no store: no penalty, suspected only."""
    backend_client = _killed_filler_backend()
    ssh_client = FillerSSHClient(running=False)
    strike_redis = FakeStrikeRedis(strikes_to_return=99, raises=True)
    ctx = _filler_context(backend_client=backend_client, ssh_client=ssh_client, redis_service=strike_redis)
    _patch_diagnostics(monkeypatch, STOPPED_DIAGNOSTICS)

    result = await _run_filler_check(ctx, enforcement=True)

    assert result.passed is True
    assert result.event.reason_code == Msg.FILLER_KILL_SUSPECTED.reason
    assert result.event.what_we_saw["kill_strikes"] is None
    assert strike_redis.calls != []  # the strike attempt was made before Redis failed


@pytest.mark.parametrize(
    "diagnostics,expected_death_kind",
    [
        (CRASHED_DIAGNOSTICS, "self_crashed"),
        (OOM_DIAGNOSTICS, "oom_killed"),
        (NEVER_STARTED_DIAGNOSTICS, "never_started"),
        (ContainerDeathDiagnostics(), "unknown"),
    ],
)
@pytest.mark.asyncio
async def test_rental_verification_filler_self_death_never_punished(monkeypatch, diagnostics, expected_death_kind):
    """Crash / OOM / never-started / unknown are self-heal territory: pass even in enforcement, no strike."""
    backend_client = _killed_filler_backend()
    ssh_client = FillerSSHClient(running=False)
    strike_redis = FakeStrikeRedis(strikes_to_return=99)
    ctx = _filler_context(backend_client=backend_client, ssh_client=ssh_client, redis_service=strike_redis)
    _patch_diagnostics(monkeypatch, diagnostics)

    result = await _run_filler_check(ctx, enforcement=True)

    assert result.passed is True
    assert result.event.reason_code == Msg.FILLER_CRASHED.reason
    assert result.event.what_we_saw["death_kind"] == expected_death_kind
    assert strike_redis.calls == []


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
async def test_rental_verification_filler_ssh_transport_error_never_punished_even_in_enforcement():
    """A lost SSH connection is not evidence of a kill: fail open even under enforcement, no re-check."""
    backend_client = DummyBackendClient(
        response=ExecutorHealthCheckResponse(success=True, error=None, details={})
    )
    ssh_client = FillerSSHClient(raise_on_run=asyncssh.Error(code=1, reason="connection lost"))
    ctx = _filler_context(backend_client=backend_client, ssh_client=ssh_client)

    result = await _run_filler_check(ctx, enforcement=True)

    assert result.passed is True
    assert result.event.reason_code == Msg.FILLER_TRANSPORT_UNREACHABLE.reason
    assert result.event.what_we_saw["enforced"] is True
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
