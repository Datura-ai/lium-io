"""DAH-6137: the wave checks a rented node once more when its pod was left on the old GPUs."""

from unittest.mock import AsyncMock

import pytest
from protocol.vc_protocol.compute_requests import RentedExecutorsResponse

import services.miner_service as miner_service
from services.task.messages import GpuLossHoldMessages as PodMsg
from services.task.messages import render_message
from tests.test_express_lane import _cycle_inputs, _job_result
from tests.test_rented_executor_not_listed import LISTED, make_payload, rented, request, rest_service  # noqa: F401


def _pod_gpus_gone(executor_id: str):
    result = _job_result(executor_id, score=0.0)
    result.availability_errors = [
        {"reason_code": "GPU_LOST", "what_we_saw": {"faults": ["pod_gpus_gone"], "missing_uuids": []}}
    ]
    return result


def _runs(rest_service, *results):  # noqa: F811
    queue = list(results)

    async def create_task(miner_info, executor_info, **_):
        return queue.pop(0)

    rest_service.task_service.create_task = AsyncMock(side_effect=create_task)
    rest_service.publish_machine_specs = AsyncMock()


@pytest.fixture(autouse=True)
def no_wait(monkeypatch):
    monkeypatch.setattr(miner_service, "POD_GPUS_GONE_RERUN_DELAY_SECONDS", 0)


@pytest.mark.asyncio
async def test_a_pod_left_on_old_gpus_is_checked_again_and_its_pass_published(rest_service):  # noqa: F811
    rest_service.miner_returns(LISTED)
    _runs(rest_service, _pod_gpus_gone(LISTED), _job_result(LISTED))

    out = await request(rest_service, RentedExecutorsResponse(executors={LISTED: rented()}))

    assert rest_service.task_service.create_task.await_count == 2
    assert [r.score for r in out["results"] if r.executor_info.uuid == LISTED] == [1.0]
    published = rest_service.publish_machine_specs.await_args.args[0]
    assert [r.executor_info.uuid for r in published] == [LISTED]


@pytest.mark.asyncio
async def test_a_rerun_that_fails_again_keeps_the_result_and_publishes_nothing(rest_service):  # noqa: F811
    rest_service.miner_returns(LISTED)
    _runs(rest_service, _pod_gpus_gone(LISTED), _pod_gpus_gone(LISTED))

    out = await request(rest_service, RentedExecutorsResponse(executors={LISTED: rented()}))

    assert rest_service.task_service.create_task.await_count == 2
    assert [r.score for r in out["results"] if r.executor_info.uuid == LISTED] == [0.0]
    rest_service.publish_machine_specs.assert_not_awaited()


@pytest.mark.asyncio
async def test_other_failures_and_a_spent_budget_are_not_rerun(rest_service, monkeypatch):  # noqa: F811
    rest_service.miner_returns(LISTED)
    other = _job_result(LISTED, score=0.0)
    other.availability_errors = [{"reason_code": "GPU_LOST", "what_we_saw": {"faults": ["nvml_error"]}}]
    _runs(rest_service, other)
    await request(rest_service, RentedExecutorsResponse(executors={LISTED: rented()}))
    assert rest_service.task_service.create_task.await_count == 1

    monkeypatch.setattr(miner_service, "POD_GPUS_GONE_RERUN_MIN_SECONDS", 10_000)
    _runs(rest_service, _pod_gpus_gone(LISTED))
    await request(rest_service, RentedExecutorsResponse(executors={LISTED: rented()}))
    assert rest_service.task_service.create_task.await_count == 1


@pytest.mark.asyncio
async def test_an_express_recheck_waits_for_the_moved_pod_and_returns_its_pass(rest_service):  # noqa: F811
    # a provider's recheck right after a GPU swap: its own result is the pass, published by the express lane
    rest_service.miner_returns(LISTED)
    _runs(rest_service, _pod_gpus_gone(LISTED), _job_result(LISTED))

    out = await rest_service.request_job_to_miner(
        payload=make_payload(),
        encrypted_files=_cycle_inputs().encrypted_files,
        rented_data=RentedExecutorsResponse(executors={LISTED: rented()}),
        default_docker_image_digests={},
        executor_id=LISTED,
    )

    assert rest_service.task_service.create_task.await_count == 2
    assert [r.score for r in out["results"] if r.executor_info.uuid == LISTED] == [1.0]
    rest_service.publish_machine_specs.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_failed_rerun_is_the_result_not_the_stale_first_one(rest_service):  # noqa: F811
    # DAH-6546: the rerun read every GPU and failed on another check; the stale pod_gpus_gone was published instead
    rest_service.miner_returns(LISTED)
    other = _job_result(LISTED, score=0.0)
    other.availability_errors = [{"reason_code": "GPU_LOST", "what_we_saw": {"faults": ["nvml_error"]}}]
    _runs(rest_service, _pod_gpus_gone(LISTED), other)

    out = await request(rest_service, RentedExecutorsResponse(executors={LISTED: rented()}))

    (result,) = [r for r in out["results"] if r.executor_info.uuid == LISTED]
    assert result is other
    rest_service.publish_machine_specs.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("rerun", ["timeout", "errored", "session_lost", "uid_unread"])
async def test_a_rerun_with_no_verdict_keeps_the_first(rest_service, rerun):  # noqa: F811
    # review 4601760468: create_task turns an exception into a log_status "error" result, which has no verdict either
    rest_service.miner_returns(LISTED)
    first = _pod_gpus_gone(LISTED)
    errored = _job_result(LISTED, score=0.0)
    errored.log_status = "error"
    # review 4602277079: a rerun whose SSH session died is a warning result whose later checks never ran
    session_lost = _job_result(LISTED, score=0.0)
    session_lost.log_status = "warning"
    session_lost.failure_reason_code = "EXECUTOR_TRANSPORT_UNREACHABLE"
    # a rerun whose login uid could not be read stopped before any check
    uid_unread = _job_result(LISTED, score=0.0)
    uid_unread.log_status = "warning"
    uid_unread.failure_reason_code = "LOGIN_USER_UNREAD"
    queue = [first]

    async def create_task(miner_info, executor_info, **_):
        if queue:
            return queue.pop(0)
        if rerun == "errored":
            return errored
        if rerun == "session_lost":
            return session_lost
        if rerun == "uid_unread":
            return uid_unread
        raise TimeoutError

    rest_service.task_service.create_task = AsyncMock(side_effect=create_task)
    rest_service.publish_machine_specs = AsyncMock()

    out = await request(rest_service, RentedExecutorsResponse(executors={LISTED: rented()}))

    assert rest_service.task_service.create_task.await_count == 2
    (result,) = [r for r in out["results"] if r.executor_info.uuid == LISTED]
    assert result.availability_errors[0]["what_we_saw"]["faults"] == ["pod_gpus_gone"]


@pytest.mark.asyncio
async def test_a_pod_whose_container_could_not_be_read_is_checked_again(rest_service):  # noqa: F811
    rest_service.miner_returns(LISTED)
    unread = _job_result(LISTED, score=0.0)
    unread.validation_event = render_message(PodMsg.POD_UNREAD, ctx=None, check_id="gpu.validate.loss_hold", what={"faults": ["pod_unread"]})
    _runs(rest_service, unread, _job_result(LISTED))

    out = await request(rest_service, RentedExecutorsResponse(executors={LISTED: rented()}))

    assert rest_service.task_service.create_task.await_count == 2
    assert [r.score for r in out["results"] if r.executor_info.uuid == LISTED] == [1.0]


@pytest.mark.asyncio
async def test_an_express_recheck_with_a_short_deadline_skips_the_rerun(rest_service):  # noqa: F811
    rest_service.miner_returns(LISTED)
    _runs(rest_service, _pod_gpus_gone(LISTED))
    too_short = (
        miner_service.CYCLE_JOB_WAIT_SECONDS
        + miner_service.PROBE_TAIL_MARGIN_SECONDS
        + miner_service.POD_GPUS_GONE_RERUN_MIN_SECONDS
        - 1
    )

    await rest_service.request_job_to_miner(
        payload=make_payload(),
        encrypted_files=_cycle_inputs().encrypted_files,
        rented_data=RentedExecutorsResponse(executors={LISTED: rented()}),
        default_docker_image_digests={},
        executor_id=LISTED,
        job_timeout=too_short,
    )

    assert rest_service.task_service.create_task.await_count == 1
