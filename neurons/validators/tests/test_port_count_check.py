from __future__ import annotations

import pytest
from datura.requests.miner_requests import ExecutorSSHInfo

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
async def test_port_count_records_inventory_when_filler_active(context_factory):
    rented_data = RentedExecutorsResponse(
        executors={},
        filler_containers_by_executor={"executor-123": "filler_active"},
    )
    specs = {
        "verified_ports": [20000, 20001, 20002],
        "available_port_count": 3,
        "port_mappings": "[[20000, 20000]]",
    }
    ctx = context_factory(
        state=build_state(
            specs=specs,
            verified_port_count=MIN_PORT_COUNT + 1,
            rented_data=rented_data,
        )
    )

    result = await PortCountCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.PORT_COUNT_RECORDED.reason
    assert result.updates["port_count"] == MIN_PORT_COUNT + 1
    assert result.updates["state"].specs["available_port_count"] == MIN_PORT_COUNT + 1
