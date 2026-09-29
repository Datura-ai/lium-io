"""P320: a rented node whose own scrape finds the GPU runtime dead on the host is reset (marked inactive).

Node bcf6eb87 (28 Sep): NVML dead on the host, every cycle halted at the scrape with SCRAPE_FAILED_DRIVER, which
carried no reset, so the executor stayed active and the renter's pod kept billing. The cycle never reached
TenantEnforcementCheck, whose POD_NOT_RUNNING is the reset a rented node gets.

Each cycle runs the scrape, the fingerprint and the rented check through the real Pipeline, and the outcome goes
through the real ResultHandler into a RedisService on fakeredis, so the assertions read the reset the backend is sent
(RESET_VERIFIED_JOB_CHANNEL) — the publish that marks the executor inactive.
"""

import asyncio
import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest
from fakeredis import FakeServer
from fakeredis.aioredis import FakeRedis

from neurons.validators.src.services.task.checks import machine_spec_scrape
from neurons.validators.src.services.task.checks.gpu_fingerprint import GpuFingerprintCheck
from neurons.validators.src.services.task.checks.machine_spec_scrape import MachineSpecScrapeCheck
from neurons.validators.src.services.task.checks.rented_machine import TenantEnforcementCheck
from neurons.validators.src.services.task.messages import MachineSpecMessages as Msg
from neurons.validators.src.services.task.messages import TenantEnforcementMessages as TenantMsg
from neurons.validators.src.services.task.pipeline import Pipeline
from neurons.validators.src.services.task.result_handler import ResultHandler
from neurons.validators.src.services.task.runner import SSHCommandResult
from services.redis_service import RESET_VERIFIED_JOB_CHANNEL, VERIFIED_JOB_COUNT_KEY, RedisService
from test_rented_machine_check import (
    DummyBackendClient,
    DummyScoreCalculator,
    DummySSHClient,
    MockContainerCleanup,
    build_rented_data,
)

from helpers import FERNET_TOKEN, build_context_config, build_services, build_state

EXECUTOR = "executor-123"
POD_ID = "pod-1"
NVML_DRIVER_ERROR = "NVMLError_DriverNotLoaded('Driver Not Loaded')"
DRIVER_REPORT = json.dumps(
    {
        "error": "no_gpu_details",
        "data": {"data_gpu": {"gpu_count": 0, "gpu_details": []}, "gpu_scrape_error": NVML_DRIVER_ERROR},
    }
)
NO_GPU_REPORT = json.dumps({"error": "no_gpu_details", "data": {"data_gpu": {"gpu_count": 0, "gpu_details": []}}})
HEALTHY_SPECS = {
    "gpu": {"count": 1, "details": [{"name": "NVIDIA RTX 4090", "uuid": "GPU-abc123"}]},
    "gpu_processes": [],
}
# What a renter's own workload prints when it breaks inside the pod: the same words as a dead host, from the
# container, which decides nothing.
RENTER_CONTAINER_TEXT = (
    "nvidia-container-cli: initialization error: nvml error: driver not loaded\n"
    "Failed to initialize NVML: Driver/library version mismatch\n"
    "NVRM: Xid (PCI:0000:01:00): 79, GPU has fallen off the bus."
)


def _scrape_result(*, stdout: str, exit_code: int) -> SSHCommandResult:
    return SSHCommandResult(
        command="scrape.sh",
        command_id="cmd-1",
        exit_code=exit_code,
        stdout=stdout,
        stderr="",
        duration_ms=100,
        started_at=datetime.now(UTC),
        finished_at=datetime.now(UTC),
        success=exit_code == 0,
    )


class _Runner:
    def __init__(self, result: SSHCommandResult):
        self.result = result

    async def run(self, command, timeout=300, retryable=False, stdin_text=None):
        return self.result


class _Decrypt:
    def decrypt_payload(self, encrypt_key: str, payload: str) -> str:
        return json.dumps(HEALTHY_SPECS)


class _RenterPodSSH(DummySSHClient):
    """The renter's pod is running; anything run inside it answers with the renter's own GPU errors."""

    async def run(self, command: str):
        result = await super().run(command)
        if "docker exec" in command and "authorized_keys" not in command:
            result.stdout = RENTER_CONTAINER_TEXT
        return result


class _Sink:
    async def emit(self, event) -> None:
        return None


def _redis_service() -> RedisService:
    service = RedisService.__new__(RedisService)
    service.redis = FakeRedis(server=FakeServer())
    service.lock = asyncio.Lock()
    service.publish = AsyncMock()
    return service


async def _cycle(context_factory, service, *, scrape: SSHCommandResult, rented: bool = True, ssh=None, enabled=True):
    """One validation cycle as the service runs it: the record from Redis, the checks through the Pipeline, the
    outcome persisted by the ResultHandler."""
    verified = await service.get_verified_job_info(EXECUTOR)
    containers = [{"name": "tenant-123", "pod_id": POD_ID}] if rented else []
    ctx = context_factory(
        services=build_services(
            ssh=_Decrypt(),
            score_calculator=DummyScoreCalculator(actual_score=1.0, job_score=1.0, warning=""),
            container_cleanup=MockContainerCleanup(),
            backend=DummyBackendClient(active=True),
        ),
        config=build_context_config(machine_scrape_filename="scrape.sh", obfuscation_keys={}),
        state=build_state(
            remote_dir="/remote",
            rented_data=build_rented_data(EXECUTOR, {"containers": containers, "owner_flag": False}),
        ),
        runner=_Runner(scrape),
        ssh=ssh or DummySSHClient(pod_running=True),
        encrypt_key="test-encrypt-key",
        verified=verified,
        collateral_deposited=True,
        is_rental_succeed=True,
        contract_version="v1.0.0",
    )
    with patch.object(machine_spec_scrape.settings, "RENTED_HOST_GPU_FAULT_RESET_ENABLED", enabled):
        ok, events, ctx = await Pipeline(
            [MachineSpecScrapeCheck(), GpuFingerprintCheck(), TenantEnforcementCheck()], _Sink()
        ).run(ctx)
    await ResultHandler(redis_service=service, dry_run=False)._persist_verification_data(
        miner_hotkey="hk", executor_id=EXECUTOR, verified_job_info=verified, context=ctx, success=ok
    )
    return ok, events[-1], ctx


def _resets(service: RedisService) -> list[dict]:
    return [
        call.args[1] for call in service.publish.await_args_list if call.args[0] == RESET_VERIFIED_JOB_CHANNEL
    ]


async def _record(service: RedisService) -> dict:
    return json.loads(await service.redis.hget(VERIFIED_JOB_COUNT_KEY, EXECUTOR))


HEALTHY = _scrape_result(stdout=FERNET_TOKEN, exit_code=0)
DEAD_NVML = _scrape_result(stdout=DRIVER_REPORT, exit_code=1)


@pytest.mark.asyncio
async def test_dead_nvml_on_a_rented_node_marks_the_executor_inactive(context_factory):
    service = _redis_service()
    ok, _, _ = await _cycle(context_factory, service, scrape=HEALTHY)
    assert ok is True and _resets(service) == []

    ok, event, ctx = await _cycle(context_factory, service, scrape=DEAD_NVML)

    assert ok is False
    assert event.reason_code == Msg.SCRAPE_FAILED_DRIVER.reason
    assert event.check_id == MachineSpecScrapeCheck.check_id
    assert event.what_we_saw["host_gpu_fault_reset"] is True
    assert (ctx.score, ctx.job_score) == (0.0, 0.0)
    [reset] = _resets(service)
    assert reset["executor_uuid"] == EXECUTOR
    assert reset["reason_code"] == Msg.SCRAPE_FAILED_DRIVER.reason
    assert reset["check_id"] == MachineSpecScrapeCheck.check_id
    assert reset["evidence"] == {
        "pod_id": POD_ID,
        "rented_pod_ids": [POD_ID],
        "scrape_error": "no_gpu_details",
        "gpu_scrape_error": NVML_DRIVER_ERROR,
    }
    record = await _record(service)
    assert (record["count"], record["uuids"]) == (0, "GPU-abc123")


@pytest.mark.asyncio
async def test_nvml_listing_zero_gpus_on_a_rented_node_marks_it_inactive(context_factory):
    service = _redis_service()

    ok, event, _ = await _cycle(context_factory, service, scrape=_scrape_result(stdout=NO_GPU_REPORT, exit_code=1))

    assert ok is False and event.reason_code == Msg.SCRAPE_FAILED_NO_GPU.reason
    [reset] = _resets(service)
    assert reset["reason_code"] == Msg.SCRAPE_FAILED_NO_GPU.reason
    assert "gpu_scrape_error" not in reset["evidence"]


@pytest.mark.asyncio
async def test_a_renter_caused_container_failure_on_a_healthy_host_stays_active(context_factory):
    service = _redis_service()

    ok, event, ctx = await _cycle(context_factory, service, scrape=HEALTHY, ssh=_RenterPodSSH(pod_running=True))

    assert ok is True
    assert event.reason_code == TenantMsg.ALREADY_RENTED.reason
    assert ctx.clear_verified_job_info is False
    assert _resets(service) == []
    assert (await _record(service))["count"] == 1


@pytest.mark.asyncio
async def test_the_node_returns_through_normal_validation_once_the_host_is_healthy(context_factory):
    service = _redis_service()
    await _cycle(context_factory, service, scrape=HEALTHY)
    await _cycle(context_factory, service, scrape=DEAD_NVML)
    assert len(_resets(service)) == 1

    ok, event, ctx = await _cycle(context_factory, service, scrape=HEALTHY)

    assert ok is True
    assert event.reason_code == TenantMsg.ALREADY_RENTED.reason
    assert ctx.score == 1.0
    assert len(_resets(service)) == 1
    record = await _record(service)
    assert (record["count"], record["failed"], record["uuids"]) == (1, 0, "GPU-abc123")


@pytest.mark.asyncio
async def test_an_unrented_node_keeps_the_plain_scrape_halt(context_factory):
    service = _redis_service()

    ok, event, _ = await _cycle(context_factory, service, scrape=DEAD_NVML, rented=False)

    assert ok is False and event.reason_code == Msg.SCRAPE_FAILED_DRIVER.reason
    assert "host_gpu_fault_reset" not in event.what_we_saw
    assert _resets(service) == []
    assert (await _record(service))["failed"] == 1


@pytest.mark.asyncio
async def test_the_flag_off_keeps_the_plain_scrape_halt_on_a_rented_node(context_factory):
    service = _redis_service()

    ok, event, _ = await _cycle(context_factory, service, scrape=DEAD_NVML, enabled=False)

    assert ok is False and event.reason_code == Msg.SCRAPE_FAILED_DRIVER.reason
    assert _resets(service) == []


@pytest.mark.parametrize(
    "scrape",
    [
        pytest.param(_scrape_result(stdout="", exit_code=127), id="scrape-failed-on-host"),
        pytest.param(
            SSHCommandResult(
                command="scrape.sh", command_id="cmd-1", exit_code=-1, stdout="", stderr="", duration_ms=300_000,
                started_at=datetime.now(UTC), finished_at=datetime.now(UTC), success=False, error_type="timeout",
            ),
            id="scrape-timeout",
        ),
    ],
)
@pytest.mark.asyncio
async def test_a_scrape_failure_that_says_nothing_of_the_gpu_does_not_reset(context_factory, scrape):
    service = _redis_service()

    ok, _, _ = await _cycle(context_factory, service, scrape=scrape)

    assert ok is False
    assert _resets(service) == []


def test_the_reset_is_on_by_default():
    assert type(machine_spec_scrape.settings).model_fields["RENTED_HOST_GPU_FAULT_RESET_ENABLED"].default is True
