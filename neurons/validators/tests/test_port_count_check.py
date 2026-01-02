from __future__ import annotations

import pytest
from datura.requests.miner_requests import ExecutorSSHInfo

from neurons.validators.src.services.task.checks.port_count import PortCountCheck
from neurons.validators.src.services.task.messages import PortCountMessages as Msg
from services.const import MIN_PORT_COUNT

from tests.helpers import build_context_config, build_services, build_state


@pytest.mark.parametrize(
    "verified_port_count,expected_count",
    [
        (100, 100),
        (50, 50),
        (0, 0),
        (1, 1),
    ],
)
@pytest.mark.asyncio
async def test_port_count_check_reads_from_state(
    verified_port_count,
    expected_count,
    context_factory,
):
    # PortCountCheck now reads verified_port_count from ctx.state
    services = build_services()
    config = build_context_config()
    state = build_state(verified_port_count=verified_port_count)
    executor = ExecutorSSHInfo(
        uuid="executor-123",
        address="127.0.0.1",
        port=22,
        ssh_username="root",
        ssh_port=22,
        python_path="/usr/bin/python",
        root_dir="/root/app",
        port_range="40000-40100",
        port_mappings="[[46681, 56681], [46682, 56682]]",
    )

    ctx = context_factory(services=services, config=config, state=state, executor=executor)

    # Act
    result = await PortCountCheck().run(ctx)

    # Assert
    assert result.passed is True
    assert result.event.reason_code == Msg.PORT_COUNT_RECORDED.reason
    assert result.updates["port_count"] == expected_count
    assert result.updates["state"].specs["available_port_count"] == expected_count
    assert result.updates["state"].specs["port_range"] == "40000-40100"
    assert result.updates["state"].specs["port_mappings"] == "[[46681, 56681], [46682, 56682]]"
