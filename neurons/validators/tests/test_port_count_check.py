from __future__ import annotations

import pytest

from neurons.validators.src.services.task.checks.port_count import PortCountCheck
from neurons.validators.src.services.task.messages import PortCountMessages as Msg
from services.const import MIN_PORT_COUNT
from services.redis_service import AVAILABLE_PORT_MAPS_PREFIX

from tests.helpers import build_context_config, build_services, build_state


class DummyPortMapping:
    def __init__(self, *, count: int):
        self.count = count
        self.called_with: list[str] = []

    async def get_successful_ports_count(self, executor_uuid: str) -> int:
        self.called_with.append(executor_uuid)
        return self.count


class DummyRedis:
    def __init__(self, values: list[bytes]):
        self.values = values
        self.keys: list[str] = []

    async def lrange(self, key: str) -> list[bytes]:
        self.keys.append(key)
        return self.values



@pytest.mark.parametrize(
    "port_count,redis_values,expected_count,expect_redis_call",
    [
        (MIN_PORT_COUNT + 1, [], MIN_PORT_COUNT + 1, False),
        (MIN_PORT_COUNT - 1, [b"3000,3001", b"3002,3003"], 2, True),
        (0, [b"4000,4001"], 1, True),
    ],
)
@pytest.mark.asyncio
async def test_port_count_check(
    port_count,
    redis_values,
    expected_count,
    expect_redis_call,
    context_factory,
):
    port_mapping = DummyPortMapping(count=port_count)
    redis_service = DummyRedis(redis_values)
    services = build_services(port_mapping=port_mapping, redis=redis_service)
    config = build_context_config()
    state = build_state()

    ctx = context_factory(services=services, config=config, state=state)

    result = await PortCountCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.PORT_COUNT_RECORDED.reason
    assert result.updates["port_count"] == expected_count

    if expect_redis_call:
        expected_key = f"{AVAILABLE_PORT_MAPS_PREFIX}:{ctx.miner_hotkey}:{ctx.executor.uuid}"
        assert redis_service.keys == [expected_key]
    else:
        assert redis_service.keys == []
