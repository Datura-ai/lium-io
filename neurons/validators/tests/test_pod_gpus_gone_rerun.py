"""DAH-6137: the wave checks a rented node once more when its pod was left on the old GPUs."""

from unittest.mock import AsyncMock

import pytest
from protocol.vc_protocol.compute_requests import RentedExecutorsResponse

import services.miner_service as miner_service
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
