"""DAH-5635: an unrented node that failed with an availability error is published as soon as its
miner answers, while the cycle's share so far does not look like our own outage."""

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from datura.requests.miner_requests import ExecutorSSHInfo
from services.miner_service import MinerService
from services.task.availability import RunningOutageTally, silence_availability_errors_on_our_own_outage
from services.task.models import JobResult

BATCH = "2026-11-03 13:20:00"
SSH = {"reason_code": "EXECUTOR_SSH_UNREACHABLE", "reach_source": "validator"}
BY_MINER = {"reason_code": "EXECUTOR_UNREACHABLE_BY_MINER", "reach_source": "miner"}
GPU_LOST = {"reason_code": "GPU_LOST", "reach_source": "validator"}


def _result(errors, *, rented=False, passed=False, batch=BATCH) -> JobResult:
    return JobResult(
        spec={"gpu": {"count": 1}},
        executor_info=ExecutorSSHInfo(
            uuid=str(uuid4()), address="10.0.0.5", port=8080, ssh_username="root", ssh_port=2200,
            python_path="/usr/bin/python3", root_dir="/root/app",
        ),
        score=1.0 if passed else 0,
        job_score=1.0 if passed else 0,
        job_batch_id=batch,
        log_status="info" if passed else "error",
        log_text="x",
        gpu_model="NVIDIA H100 80GB HBM3" if passed else None,
        gpu_count=1 if passed else 0,
        is_rented=rented,
        availability_errors=errors,
    )


def _service(tally: RunningOutageTally | None) -> MinerService:
    service = MinerService.__new__(MinerService)
    service.outage_tally = tally
    service.publish_machine_specs = AsyncMock()
    return service


def _job(results) -> dict:
    return {"miner_hotkey": "miner-hotkey", "miner_coldkey": "miner-coldkey", "results": results}


def _healthy(count: int) -> list[JobResult]:
    return [_result([], passed=True) for _ in range(count)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("candidate", "published"),
    [
        (_result([SSH]), True),
        (_result([GPU_LOST]), True),
        (_result([SSH], rented=True), False),
        (_result([], passed=True), False),
        (_result([BY_MINER]), False),
        (_result(None), False),
        (_result([SSH], batch="2026-11-03 13:00:00"), False),
    ],
    ids=["ssh-unreachable", "gpu-lost", "rented", "success", "miner-only", "never-checked", "other-batch"],
)
async def test_only_an_unrented_validator_reading_that_failed_is_published_early(candidate, published):
    service = _service(RunningOutageTally(BATCH))

    await service.publish_unreachable_early(_job(_healthy(9) + [candidate]))

    if published:
        service.publish_machine_specs.assert_awaited_once()
        args, kwargs = service.publish_machine_specs.await_args
        assert args == ([candidate], "miner-hotkey", "miner-coldkey")
        assert kwargs == {"is_whole_miner_batch": False}
        assert candidate.scored_at is None
        assert service.outage_tally.published_early == {candidate.executor_info.uuid}
    else:
        service.publish_machine_specs.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("jobs", "published_by_job"),
    [
        # four checked results: the share carries no signal yet; the fifth makes it trusted
        ([[_result([SSH])] + _healthy(3), _healthy(1) + [_result([SSH])]], [0, 1]),
        # most of the cycle so far failed at the connect: our own outage, nothing goes early
        ([_healthy(2), [_result([SSH]) for _ in range(4)]], [0, 0]),
        # the share turns into an outage after an early publish: later failures wait for the cycle
        ([_healthy(8) + [_result([SSH])], [_result([SSH]) for _ in range(10)]], [1, 0]),
    ],
    ids=["below-smallest-fleet", "own-outage", "turns-into-outage"],
)
async def test_nothing_is_published_early_while_the_running_share_is_untrusted(jobs, published_by_job):
    service = _service(RunningOutageTally(BATCH))

    for results, expected in zip(jobs, published_by_job):
        service.publish_machine_specs.reset_mock()
        await service.publish_unreachable_early(_job(results))
        published = service.publish_machine_specs.await_args[0][0] if service.publish_machine_specs.await_args else []
        assert len(published) == expected


@pytest.mark.asyncio
async def test_no_tally_publishes_nothing_and_a_failing_publish_never_fails_the_wave():
    no_tally = _service(None)
    await no_tally.publish_unreachable_early(_job(_healthy(9) + [_result([SSH])]))
    no_tally.publish_machine_specs.assert_not_awaited()

    failing = _service(RunningOutageTally(BATCH))
    failing.publish_machine_specs = AsyncMock(side_effect=RuntimeError("redis down"))
    await failing.publish_unreachable_early(_job(_healthy(9) + [_result([SSH])]))


def test_the_cycle_end_silencer_relists_the_nodes_hidden_early_and_leaves_the_others_alone():
    """Nodes hidden early, then the cycle turns out to be our outage: [] lists them again (a GPU-fault
    entry stays); the rest keep None, so the backend leaves their stored errors alone."""
    tally = RunningOutageTally(BATCH)
    early_ssh, early_gpu = _result([SSH]), _result([SSH, GPU_LOST])
    tally.published_early = {early_ssh.executor_info.uuid, early_gpu.executor_info.uuid}
    later = [_result([SSH]) for _ in range(8)]
    cycle = [early_ssh, early_gpu, *later]

    assert silence_availability_errors_on_our_own_outage(cycle) == 10
    tally.relist_published_early(cycle)

    assert early_ssh.availability_errors == []
    assert early_gpu.availability_errors == [GPU_LOST]
    assert all(result.availability_errors is None for result in later)
