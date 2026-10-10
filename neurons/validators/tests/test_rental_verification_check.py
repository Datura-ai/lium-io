from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import asyncssh
import pytest
from neurons.validators.src.core.docker_utils import ContainerDeathDiagnostics
from neurons.validators.src.protocol.vc_protocol.compute_requests import (
    CPU_QUOTA_EXCEEDS_HOST_REASON,
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


# ---------------------------------------------------------------- GPU-split nodes (DAH-2465 bundles)


# ---------------------------------------------------------------------------
# DAH-2614: the backend's probe must name the cards this cycle's scrape saw,
# instead of `--gpus all`, which never names one.
# ---------------------------------------------------------------------------


# --- DAH-2671 item 2b: send the advertised CPU count, quarantine a daemon-rejected count ---

_CPU_QUOTA_STDERR = (
    "docker: Error response from daemon: Range of CPUs is from 0.01 to 20.00, "
    "as there are only 20 CPUs available."
)


def _cpu_quota_response() -> ExecutorHealthCheckResponse:
    return ExecutorHealthCheckResponse(
        success=False,
        error=_CPU_QUOTA_STDERR,
        details={"docker_stderr": _CPU_QUOTA_STDERR},
        reason_code=CPU_QUOTA_EXCEEDS_HOST_REASON,
    )


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


# DAH-2703: a host reaper that removes the filler seconds after `docker run` leaves no container
# to probe, so the liveness check above never runs. The backend reports the kill streak instead.


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
