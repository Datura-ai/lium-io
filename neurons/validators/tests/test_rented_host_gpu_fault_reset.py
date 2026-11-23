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
RENTAL_OWNER = "miner-hotkey"
ENCRYPT_KEY = "test-encrypt-key"
POD_ID = "pod-1"
# repr() of the shipped scrape's NVMLError (machine_scrape.py): the class name and the nvml.h return code


def _driver_report(gpu_scrape_error: str) -> str:
    return json.dumps(
        {
            "error": "no_gpu_details",
            "data": {"data_gpu": {"gpu_count": 0, "gpu_details": []}, "gpu_scrape_error": gpu_scrape_error},
        }
    )


NO_GPU_REPORT = json.dumps({"error": "no_gpu_details", "data": {"data_gpu": {"gpu_count": 0, "gpu_details": []}}})
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
    persist: bool = True,
):
    """One validation cycle as the service runs it: the record from Redis, the checks through the Pipeline, the
    outcome persisted by the ResultHandler."""
    verified = await service.get_verified_job_info("hk", EXECUTOR)
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
    if not persist:
        return ok, events, ctx
    await ResultHandler(redis_service=service, dry_run=False)._persist_verification_data(
        miner_hotkey="hk", executor_id=EXECUTOR, verified_job_info=verified, context=ctx, success=ok
    )
    return ok, events[-1], ctx


def _resets(service: RedisService) -> list[dict]:
    return [
        call.args[1] for call in service.publish.await_args_list if call.args[0] == RESET_VERIFIED_JOB_CHANNEL
    ]


async def _record(service: RedisService) -> dict:
    return json.loads(await service.redis.hget(VERIFIED_JOB_COUNT_KEY, f"hk:{EXECUTOR}"))


HEALTHY = _scrape_result(stdout=SSHService()._encrypt(ENCRYPT_KEY, json.dumps(HEALTHY_SPECS)), exit_code=0)
DRIVER = Msg.SCRAPE_FAILED_DRIVER.reason


@pytest.mark.parametrize(
    ("stdout", "reason_code", "gpu_scrape_error"),
    [
        pytest.param(NO_GPU_REPORT, Msg.SCRAPE_FAILED_NO_GPU.reason, None, id="zero-gpus"),
        pytest.param(_driver_report("NVMLError(9)"), DRIVER, "NVMLError(9)", id="driver-not-loaded"),
        pytest.param(_driver_report("NVMLError(15)"), DRIVER, "NVMLError(15)", id="gpu-lost"),
        pytest.param(_driver_report("NVMLError(16)"), DRIVER, "NVMLError(16)", id="reset-required"),
        pytest.param(_driver_report("NVMLError(28)"), DRIVER, "NVMLError(28)", id="gpu-not-found"),
    ],
)
@pytest.mark.asyncio
async def test_a_dead_gpu_runtime_on_a_rented_node_marks_the_executor_inactive(
    context_factory, stdout, reason_code, gpu_scrape_error
):
    service = _redis_service()
    ok, _, _ = await _cycle(context_factory, service, scrape=HEALTHY)
    assert ok is True and _resets(service) == []

    ok, event, ctx = await _cycle(context_factory, service, scrape=_scrape_result(stdout=stdout, exit_code=1))

    assert ok is False
    assert event.reason_code == reason_code
    assert event.check_id == MachineSpecScrapeCheck.check_id
    assert event.what_we_saw["host_gpu_fault_reset"] is True
    [reset] = _resets(service)
    assert reset["executor_uuid"] == EXECUTOR
    assert reset["reason_code"] == reason_code
    assert reset["check_id"] == MachineSpecScrapeCheck.check_id
    evidence = {
        "pod_id": POD_ID,
        "rented_pod_ids": [POD_ID],
        "scrape_error": "no_gpu_details",
        "host_gpu_fault_reset": True,
        "executor_ip": ctx.executor.address,
        "executor_port": ctx.executor.port,
        "job_batch_id": ctx.config.job_batch_id,
    }
    if gpu_scrape_error is not None:
        evidence["gpu_scrape_error"] = gpu_scrape_error
    assert reset["evidence"] == evidence
    record = await _record(service)
    assert (record["count"], record["uuids"]) == (0, "GPU-abc123")




@pytest.mark.parametrize("stdout", [NO_GPU_REPORT, _driver_report("NVMLError(15)")], ids=["zero-gpus", "gpu-lost"])
@pytest.mark.asyncio
async def test_a_host_confirmed_gpu_fault_reset_is_sent_at_once_inside_a_rollout_window(context_factory, stdout):
    """r254 (DAH-6951): the scrape's reset for a rented node whose host confirms a dead GPU runtime carries a
    rollout failure reason (SCRAPE_FAILED_NO_GPU / _DRIVER), but it is a finding, not a read cut short: the wave
    sends it at once, inside a rollout window too, so billing stops as at base. The window still withholds the
    run's verdict."""
    from test_node_account_check import cycle_end, wave_create_task

    service = _redis_service()
    run = await _cycle(context_factory, service, scrape=_scrape_result(stdout=stdout, exit_code=1), persist=False)
    assert run[0] is False and run[1][-1].what_we_saw["host_gpu_fault_reset"] is True

    tasks, result = await wave_create_task(run, service)
    [reset] = _resets(service)
    assert reset["reason_code"] == run[1][-1].reason_code and tasks.held_resets == {}

    withheld = await cycle_end(tasks, service, result, "withheld")
    assert [w.result for w in withheld] == [result]
    assert len(_resets(service)) == 1


@pytest.mark.parametrize("pass_after", [True, False], ids=["same-cycle-pass", "no-later-write"])
@pytest.mark.asyncio
async def test_a_held_reset_sent_after_a_same_cycle_pass_keeps_the_pass_write(pass_after):
    """DAH-6951: a held reset is sent at the cycle end, after a same-cycle pass of the node (an express recheck)
    that re-anchored its GPUs. The reset happened first, so the record ends as it did when the reset went out at
    once: cleared, then the pass's write on top (count 1, the pass's anchor), never the failing run's stale
    snapshot, which would bring back the old anchor and fail the next cycle on a GPU mismatch."""
    from test_node_account_check import cycle_end

    from services.task.service import HeldReset, TaskService

    service = _redis_service()
    snapshot = {"count": 3, "failed": 0, "spec": "NVIDIA H100:8", "uuids": "GPU-old"}
    await service.redis.hset(VERIFIED_JOB_COUNT_KEY, f"hk:{EXECUTOR}", json.dumps(snapshot))
    tasks = TaskService.__new__(TaskService)
    tasks.open_hold_batch = "batch-1"
    tasks.held_resets = {
        "batch-1": [HeldReset(EXECUTOR, dict(miner_hotkey="hk", executor_id=EXECUTOR, prev_info=snapshot, evidence={}))]
    }
    if pass_after:
        # the pass read the same snapshot and re-anchored (ResultHandler's gpu_reanchor write)
        await service.set_verified_job_info(
            "hk", EXECUTOR, prev_info={**snapshot, "uuids": ""}, success=True, spec="NVIDIA H100:8", uuids="GPU-new"
        )

    await cycle_end(tasks, service, None, "stands")

    assert len(_resets(service)) == 1
    record = await _record(service)
    assert (record["count"], record["failed"], record["spec"]) == (1 if pass_after else 0, 0, "NVIDIA H100:8")
    assert record["uuids"] == ("GPU-new" if pass_after else "GPU-old")
