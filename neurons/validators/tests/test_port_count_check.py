from __future__ import annotations

import pytest

from neurons.validators.src.services.task.checks.port_count import PortCountCheck
from neurons.validators.src.services.task.messages import PortCountMessages as Msg
from services.const import MIN_PORT_COUNT

from tests.helpers import build_context_config, build_services, build_state


@pytest.mark.parametrize(
    "port_count",
    [
        MIN_PORT_COUNT + 1,
        MIN_PORT_COUNT - 1,
        0,
    ],
)
@pytest.mark.asyncio
async def test_port_count_check(
    port_count,
    context_factory,
):
    # PortCountCheck now reads verified_port_count from ctx.state
    services = build_services()
    config = build_context_config()
    state = build_state(verified_port_count=port_count)

    ctx = context_factory(services=services, config=config, state=state)

    result = await PortCountCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.PORT_COUNT_RECORDED.reason
    assert result.updates["port_count"] == port_count
