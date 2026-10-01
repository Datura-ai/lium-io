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


def _background_jobs(
    filler_ports: list[int], *, owner: str | None = "lium", executor_uuid: str = EXECUTOR_UUID
) -> RentedExecutorsResponse:
    return RentedExecutorsResponse(
        executors={},
        filler_ports_by_executor={executor_uuid: filler_ports},
        default_job_owner_by_executor={executor_uuid: owner} if owner is not None else {},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("rented_data", "pairs", "held"),
    [
        # one port answers, two sit behind Lium's job: not probed this cycle, so not counted
        pytest.param(
            _background_jobs([40010, 40011]),
            ANSWERED_PAIRS[:1],
            2,
            id="1-answered-plus-2-held-fails",
        ),
        pytest.param(_background_jobs([40010, 40011, 40012]), [], 3, id="none-answered"),
        pytest.param(None, ANSWERED_PAIRS, 0, id="no-backend-snapshot"),
        pytest.param(
            _background_jobs([40010], owner="miner"), ANSWERED_PAIRS, 0, id="miner-default-job"
        ),
        pytest.param(
            _background_jobs([40010], owner=None), ANSWERED_PAIRS, 0, id="owner-not-reported"
        ),
        pytest.param(
            _background_jobs([40010], executor_uuid="another-executor"),
            ANSWERED_PAIRS,
            0,
            id="another-node",
        ),
    ],
)
async def test_background_job_ports_are_reported_on_the_floor_failure_and_never_counted(
    context_factory, rented_data, pairs, held
):
    state = build_state(
        verified_port_count=len(pairs), verified_port_pairs=list(pairs), rented_data=rented_data
    )
    ctx = context_factory(state=state)

    result = await PortCountCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.INSUFFICIENT_PORTS.reason
    assert result.event.what_we_saw["held_by_preemptible_background_jobs"] == held
    assert result.updates["port_count"] == len(pairs)
