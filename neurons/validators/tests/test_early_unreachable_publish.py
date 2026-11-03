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
    ("candidate", "on_the_rented_list", "published"),
    [
        (_result([SSH]), False, True),
        (_result([GPU_LOST]), False, True),
        # the failure is built before the result knows its rentals (is_rented False): the cycle's rented list decides
        (_result([SSH]), True, False),
        (_result([], passed=True), False, False),
        (_result([BY_MINER]), False, False),
        (_result(None), False, False),
        (_result([SSH], batch="2026-11-03 13:00:00"), False, False),
    ],
    ids=["ssh-unreachable", "gpu-lost", "rented", "success", "miner-only", "never-checked", "other-batch"],
)
async def test_only_an_unrented_validator_reading_that_failed_is_published_early(
    candidate, on_the_rented_list, published
):
    rented = frozenset({str(candidate.executor_info.uuid).upper()}) if on_the_rented_list else frozenset()
    service = _service(RunningOutageTally(BATCH, frozenset(uuid.lower() for uuid in rented)))

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


def test_the_cycle_end_silencer_gives_nodes_published_early_their_state_from_before_the_cycle():
    """Nodes published early, then the cycle turns out to be our outage: each gets back what the last cycle
    published for it (a GPU-fault entry found now stays on top). One this process never published keeps None,
    as does every node not published early: the backend leaves their stored errors alone."""
    tally = RunningOutageTally(BATCH)
    listed, hidden, gpu, unknown = _result([SSH]), _result([SSH]), _result([SSH, GPU_LOST]), _result([SSH])
    tally.published_early = {r.executor_info.uuid for r in (listed, hidden, gpu, unknown)}
    before = {
        str(listed.executor_info.uuid).lower(): [],
        str(hidden.executor_info.uuid).lower(): [BY_MINER],
        str(gpu.executor_info.uuid).lower(): [],
    }
    later = [_result([SSH]) for _ in range(6)]
    cycle = [listed, hidden, gpu, unknown, *later]

    assert silence_availability_errors_on_our_own_outage(cycle) == 10
    tally.restore_published_early(cycle, before)

    assert listed.availability_errors == []
    assert hidden.availability_errors == [BY_MINER]
    assert gpu.availability_errors == [GPU_LOST]
    assert unknown.availability_errors is None
    assert all(result.availability_errors is None for result in later)


@pytest.mark.asyncio
@pytest.mark.parametrize(("whole_batch", "recorded"), [(True, True), (False, False)])
async def test_only_the_cycle_publish_records_what_it_sent(monkeypatch, whole_batch, recorded):
    service = MinerService.__new__(MinerService)
    service.cycle_published_availability = {}
    service.redis_service = AsyncMock()
    service._pod_states_capped_for_spec = lambda result, extra: None
    monkeypatch.setattr("services.miner_service.settings.DRY_RUN", False)
    node, unchecked = _result([SSH]), _result(None)

    await service.publish_machine_specs([node, unchecked], "hk", "ck", is_whole_miner_batch=whole_batch)

    expected = {str(node.executor_info.uuid).lower(): [SSH]} if recorded else {}
    assert service.cycle_published_availability == expected
