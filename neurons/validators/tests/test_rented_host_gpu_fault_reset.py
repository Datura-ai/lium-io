"""A rented node whose own scrape finds the GPU runtime dead on the host is reset (marked inactive).

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
from services.ssh_service import SSHService
from test_rented_machine_check import (
    DummyBackendClient,
    DummyScoreCalculator,
    DummySSHClient,
    MockContainerCleanup,
    build_rented_data,
)

from helpers import build_context_config, build_services, build_state

EXECUTOR = "executor-123"
# build_rented_data's owner of the rental
RENTAL_OWNER = "test-miner"
ENCRYPT_KEY = "test-encrypt-key"
POD_ID = "pod-1"
# repr() of the shipped scrape's NVMLError (machine_scrape.py): the class name and the nvml.h return code
NVML_DRIVER_ERROR = "NVMLError(9)"


def _sealed(report: str, key: str = ENCRYPT_KEY) -> str:
    # what the shipped scrape prints: the plain report, then its Fernet token under the cycle's key
    return f"{report}\n{SSHService()._encrypt(key, report)}"


def _plain_driver_report(gpu_scrape_error: str) -> str:
    return json.dumps(
        {
            "error": "no_gpu_details",
            "data": {"data_gpu": {"gpu_count": 0, "gpu_details": []}, "gpu_scrape_error": gpu_scrape_error},
        }
    )


def _driver_report(gpu_scrape_error: str) -> str:
    return _sealed(_plain_driver_report(gpu_scrape_error))


PLAIN_NO_GPU_REPORT = json.dumps({"error": "no_gpu_details", "data": {"data_gpu": {"gpu_count": 0, "gpu_details": []}}})
NO_GPU_REPORT = _sealed(PLAIN_NO_GPU_REPORT)
HEALTHY_SPECS = {
    "gpu": {"count": 1, "details": [{"name": "NVIDIA RTX 4090", "uuid": "GPU-abc123"}]},
    "gpu_processes": [],
}


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


class _Sink:
    async def emit(self, event) -> None:
        return None


def _redis_service() -> RedisService:
    service = RedisService.__new__(RedisService)
    service.redis = FakeRedis(server=FakeServer())
    service.lock = asyncio.Lock()
    service.publish = AsyncMock()
    return service


async def _cycle(
    context_factory,
    service,
    *,
    scrape: SSHCommandResult,
    rented: bool = True,
    enabled=True,
    miner_hotkey: str = RENTAL_OWNER,
):
    """One validation cycle as the service runs it: the record from Redis, the checks through the Pipeline, the
    outcome persisted by the ResultHandler."""
    verified = await service.get_verified_job_info(EXECUTOR)
    containers = [{"name": "tenant-123", "pod_id": POD_ID}] if rented else []
    ctx = context_factory(
        miner_hotkey=miner_hotkey,
        services=build_services(
            ssh=SSHService(),
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
        ssh=DummySSHClient(pod_running=True),
        encrypt_key=ENCRYPT_KEY,
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


HEALTHY = _scrape_result(stdout=SSHService()._encrypt(ENCRYPT_KEY, json.dumps(HEALTHY_SPECS)), exit_code=0)
DEAD_NVML = _scrape_result(stdout=_driver_report(NVML_DRIVER_ERROR), exit_code=1)


@pytest.mark.asyncio
async def test_dead_nvml_on_a_rented_node_marks_the_executor_inactive(context_factory):
    service = _redis_service()
    ok, _, _ = await _cycle(context_factory, service, scrape=HEALTHY)
    assert ok is True and _resets(service) == []

    ok, event, _ = await _cycle(context_factory, service, scrape=DEAD_NVML)

    assert ok is False
    assert event.reason_code == Msg.SCRAPE_FAILED_DRIVER.reason
    assert event.check_id == MachineSpecScrapeCheck.check_id
    assert event.what_we_saw["host_gpu_fault_reset"] is True
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


@pytest.mark.parametrize(
    "gpu_scrape_error", ["NVMLError(15)", "NVMLError(16)", "NVMLError(28)"], ids=["gpu-lost", "reset-required", "gpu-not-found"]
)
@pytest.mark.asyncio
async def test_every_dead_runtime_nvml_code_resets_a_rented_node(context_factory, gpu_scrape_error):
    service = _redis_service()

    await _cycle(context_factory, service, scrape=_scrape_result(stdout=_driver_report(gpu_scrape_error), exit_code=1))

    [reset] = _resets(service)
    assert reset["evidence"]["gpu_scrape_error"] == gpu_scrape_error


@pytest.mark.asyncio
async def test_nvml_listing_zero_gpus_on_a_rented_node_marks_it_inactive(context_factory):
    service = _redis_service()

    ok, event, _ = await _cycle(context_factory, service, scrape=_scrape_result(stdout=NO_GPU_REPORT, exit_code=1))

    assert ok is False and event.reason_code == Msg.SCRAPE_FAILED_NO_GPU.reason
    [reset] = _resets(service)
    assert reset["reason_code"] == Msg.SCRAPE_FAILED_NO_GPU.reason
    assert "gpu_scrape_error" not in reset["evidence"]


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


@pytest.mark.parametrize(
    ("scrape", "cycle_kwargs"),
    [
        pytest.param(DEAD_NVML, {"rented": False}, id="unrented-node"),
        pytest.param(DEAD_NVML, {"enabled": False}, id="flag-off"),
        # the report comes from the node's own host and the provider can seal it, so the shipped default is off
        pytest.param(
            DEAD_NVML,
            {
                "enabled": type(machine_spec_scrape.settings).model_fields[
                    "RENTED_HOST_GPU_FAULT_RESET_ENABLED"
                ].default
            },
            id="provider-forged-report-with-shipped-default",
        ),
        # miner A reports miner B's rented executor UUID with a sealed dead-NVML report from A's own host
        pytest.param(DEAD_NVML, {"miner_hotkey": "another-miner"}, id="rental-owned-by-another-miner"),
        pytest.param(
            _scrape_result(stdout=NO_GPU_REPORT, exit_code=1),
            {"miner_hotkey": "another-miner"},
            id="no-gpu-on-a-rental-owned-by-another-miner",
        ),
        # the scrape copies libnvidia-ml to a temp file before loading it; a full or read-only disk fails that write
        pytest.param(_scrape_result(stdout=_driver_report("NVMLError(12)"), exit_code=1), {}, id="library-not-loadable"),
        pytest.param(_scrape_result(stdout=_driver_report("NVMLError(18)"), exit_code=1), {}, id="driver-library-mismatch"),
        pytest.param(
            _scrape_result(stdout=_driver_report("OSError(28, 'No space left on device')"), exit_code=1), {}, id="not-nvml"
        ),
        pytest.param(_scrape_result(stdout="", exit_code=127), {}, id="scrape-failed-on-host"),
        # what a stray print or another cycle's token looks like: the report without its sealed copy, with a copy
        # sealed under another key, or with a replayed success payload
        pytest.param(
            _scrape_result(stdout=_plain_driver_report(NVML_DRIVER_ERROR), exit_code=1), {}, id="plain-driver-report"
        ),
        pytest.param(_scrape_result(stdout=PLAIN_NO_GPU_REPORT, exit_code=1), {}, id="plain-no-gpu-report"),
        pytest.param(_scrape_result(stdout='{"error": "no_gpu_details"}', exit_code=1), {}, id="bare-no-gpu-report"),
        pytest.param(
            _scrape_result(stdout=_sealed(_plain_driver_report(NVML_DRIVER_ERROR), key="another-key"), exit_code=1),
            {},
            id="sealed-under-another-key",
        ),
        pytest.param(
            _scrape_result(stdout=f"{_plain_driver_report(NVML_DRIVER_ERROR)}\n{HEALTHY.stdout}", exit_code=1),
            {},
            id="replayed-success-payload",
        ),
        pytest.param(
            SSHCommandResult(
                command="scrape.sh", command_id="cmd-1", exit_code=-1, stdout="", stderr="", duration_ms=300_000,
                started_at=datetime.now(UTC), finished_at=datetime.now(UTC), success=False, error_type="timeout",
            ),
            {},
            id="scrape-timeout",
        ),
    ],
)
@pytest.mark.asyncio
async def test_a_scrape_failure_that_does_not_show_a_dead_gpu_runtime_on_a_rented_node_does_not_reset(
    context_factory, scrape, cycle_kwargs
):
    service = _redis_service()

    ok, event, _ = await _cycle(context_factory, service, scrape=scrape, **cycle_kwargs)

    assert ok is False
    assert "host_gpu_fault_reset" not in event.what_we_saw
    assert _resets(service) == []
    assert (await _record(service))["failed"] == 1
