from __future__ import annotations

import pytest

from neurons.validators.src.services.task.checks.port_count import PortCountCheck
from neurons.validators.src.services.task.messages import PortCountMessages as Msg
from protocol.vc_protocol.compute_requests import (
    RentedExecutor,
    RentedExecutorsResponse,
    RentedPod,
)
from services.const import MIN_PORT_COUNT

from tests.helpers import build_state




@pytest.mark.asyncio
async def test_port_count_sufficient_passes(context_factory):
    """Check passes when port count >= MIN_PORT_COUNT."""
    ctx = context_factory(state=build_state(verified_port_count=MIN_PORT_COUNT + 1))

    # Act
    result = await PortCountCheck().run(ctx)

    # Assert
    assert result.passed is True
    assert result.event.reason_code == Msg.PORT_COUNT_RECORDED.reason
    assert result.updates["port_count"] == MIN_PORT_COUNT + 1


@pytest.mark.asyncio
@pytest.mark.parametrize("port_count", [0, MIN_PORT_COUNT - 1])
async def test_port_count_insufficient_fails_when_not_rented(context_factory, port_count):
    """Check fails when port count < MIN_PORT_COUNT and executor is not rented."""
    ctx = context_factory(state=build_state(verified_port_count=port_count))

    result = await PortCountCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.INSUFFICIENT_PORTS.reason
    assert result.updates["port_count"] == port_count
    assert result.updates["state"].specs["available_port_count"] == port_count


@pytest.mark.asyncio
async def test_port_count_insufficient_passes_when_rented(context_factory):
    """Check passes when port count < MIN_PORT_COUNT but executor is rented."""
    rented_data = RentedExecutorsResponse(
        executors={
            "executor-123": RentedExecutor(
                miner_hotkey="test-miner",
                executor_ip_address="127.0.0.1",
                executor_ip_port="22",
                pods=[RentedPod(pod_id="pod-1", container_name="container_test", rented_ports=[8080, 8081])],
            )
        },
    )
    ctx = context_factory(state=build_state(verified_port_count=MIN_PORT_COUNT - 1, rented_data=rented_data))

    result = await PortCountCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.PORT_COUNT_RECORDED.reason
    assert result.updates["port_count"] == MIN_PORT_COUNT - 1


@pytest.mark.asyncio
async def test_insufficient_ports_names_the_orphaned_container_holding_them(context_factory):
    """DAH-2991: the shortfall caused by an orphan the cleanup could not remove is named, not a bare count."""
    name = "pod_11655dc5-53ba-4a8d-a341-fe6c9d12bda7"
    ctx = context_factory(state=build_state(verified_port_count=2, orphaned_containers=[name]))

    result = await PortCountCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.INSUFFICIENT_PORTS.reason
    assert result.event.what_we_saw["held_by_orphaned_containers"] == [name]
    assert name in result.event.remediation


EXECUTOR_UUID = "executor-123"
ANSWERED_PAIRS = [(9001, 40001), (9002, 40002)]
BACKGROUND_JOB_PORTS = [40010, 40011]


def _background_job_data(
    *,
    owner: str | None = "lium",
    filler_ports: list[int] | None = None,
    executor_uuid: str = EXECUTOR_UUID,
    renter_pods: list[RentedPod] | None = None,
) -> RentedExecutorsResponse:
    executors = {}
    if renter_pods is not None:
        executors[EXECUTOR_UUID] = RentedExecutor(
            miner_hotkey="test-miner",
            executor_ip_address="127.0.0.1",
            executor_ip_port="22",
            pods=renter_pods,
        )
    return RentedExecutorsResponse(
        executors=executors,
        filler_ports_by_executor={executor_uuid: BACKGROUND_JOB_PORTS if filler_ports is None else filler_ports},
        default_job_owner_by_executor={executor_uuid: owner} if owner is not None else {},
    )


def _state(rented_data: RentedExecutorsResponse | None, pairs=ANSWERED_PAIRS):
    return build_state(verified_port_count=len(pairs), verified_port_pairs=list(pairs), rented_data=rented_data)


@pytest.mark.asyncio
async def test_background_job_ports_lift_an_unrented_host_to_the_floor(context_factory):
    ctx = context_factory(state=_state(_background_job_data()))

    result = await PortCountCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.PORT_COUNT_RECORDED.reason
    assert result.event.what_we_saw["held_by_preemptible_background_jobs"] == len(BACKGROUND_JOB_PORTS)
    # the published figures stay the answered count: the platform adds these ports itself
    assert result.updates["port_count"] == len(ANSWERED_PAIRS)
    assert result.updates["state"].specs["available_port_count"] == len(ANSWERED_PAIRS)


@pytest.mark.asyncio
async def test_background_job_ports_below_the_floor_still_fail(context_factory):
    ctx = context_factory(state=_state(_background_job_data(), pairs=[]))

    result = await PortCountCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.INSUFFICIENT_PORTS.reason
    assert result.event.what_we_saw["held_by_preemptible_background_jobs"] == len(BACKGROUND_JOB_PORTS)


@pytest.mark.asyncio
async def test_a_port_both_answered_and_listed_counts_once(context_factory):
    answered_external = ANSWERED_PAIRS[0][1]
    ctx = context_factory(state=_state(_background_job_data(filler_ports=[answered_external])))

    result = await PortCountCheck().run(ctx)

    assert result.passed is False
    assert result.event.what_we_saw["held_by_preemptible_background_jobs"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rented_data",
    [
        pytest.param(None, id="no-backend-snapshot"),
        pytest.param(_background_job_data(owner="miner"), id="miner-default-job"),
        pytest.param(_background_job_data(owner=None), id="owner-not-reported"),
        pytest.param(_background_job_data(owner="future-owner"), id="unknown-owner"),
        pytest.param(_background_job_data(executor_uuid="another-executor"), id="another-executors-jobs"),
        # the backend drops stale and terminal filler rows before it lists ports (daos/filler_run.py)
        pytest.param(_background_job_data(filler_ports=[]), id="no-live-job-rows"),
    ],
)
async def test_ports_not_provably_held_by_a_platform_job_keep_mains_result(context_factory, rented_data):
    ctx = context_factory(state=_state(rented_data))

    result = await PortCountCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.INSUFFICIENT_PORTS.reason
    assert result.updates["state"].specs["available_port_count"] == len(ANSWERED_PAIRS)


@pytest.mark.asyncio
async def test_a_renter_pod_keeps_mains_result_and_counts_no_background_job_ports(context_factory):
    renter = RentedPod(pod_id="pod-1", container_name="container_test", rented_ports=[40020])
    ctx = context_factory(state=_state(_background_job_data(renter_pods=[renter])))

    result = await PortCountCheck().run(ctx)

    # main passes a rented host whatever its count; the background-job ports are never added to it
    assert result.passed is True
    assert result.event.what_we_saw["held_by_preemptible_background_jobs"] == 0
    assert result.updates["port_count"] == len(ANSWERED_PAIRS)


@pytest.mark.asyncio
async def test_an_executor_entry_without_pods_is_not_rented(context_factory):
    ctx = context_factory(state=_state(_background_job_data(renter_pods=[])))

    result = await PortCountCheck().run(ctx)

    assert result.passed is True
    assert result.event.what_we_saw["held_by_preemptible_background_jobs"] == len(BACKGROUND_JOB_PORTS)
