from __future__ import annotations

import pytest

from neurons.validators.src.services.task.checks.port_count import PortCountCheck
from neurons.validators.src.services.task.messages import PortCountMessages as Msg

from helpers import build_context_config, build_services, build_state


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
    """Port count should be read from ctx.state.verified_port_count."""
    # Arrange
    services = build_services()
    config = build_context_config()
    state = build_state(verified_port_count=verified_port_count)

    ctx = context_factory(services=services, config=config, state=state)

    # Act
    result = await PortCountCheck().run(ctx)

    # Assert
    assert result.passed is True
    assert result.event.reason_code == Msg.PORT_COUNT_RECORDED.reason
    assert result.updates["port_count"] == expected_count
    assert result.updates["state"].specs["available_port_count"] == expected_count
