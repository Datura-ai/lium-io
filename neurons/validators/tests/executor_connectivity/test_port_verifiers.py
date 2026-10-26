import asyncio

import pytest

from services.executor_connectivity.models import ContainerStartResult, PortPair
from services.executor_connectivity.port_verifiers import SemiBatchVerifier


CONTAINER_STARTED = ContainerStartResult(ok=True, container_id="c", status="Up")


@pytest.mark.asyncio
async def test_semi_batch_fails_a_chunk_that_does_not_answer_in_time(mocker):
    ports = [PortPair(9000 + i, 9000 + i) for i in range(20)]

    async def slow_then_ok(_ssh, name, *_args):
        if runner.run.await_count == 1:
            await asyncio.sleep(5)
        return CONTAINER_STARTED

    runner = mocker.Mock()
    runner.run = mocker.AsyncMock(side_effect=slow_then_ok)
    runner.cleanup = mocker.AsyncMock()
    port_tester = mocker.Mock()
    port_tester.test_many = mocker.AsyncMock(side_effect=lambda _s, _h, chunk, *_a: (chunk, []))
    verifier = SemiBatchVerifier(port_tester, runner)
    mocker.patch.object(SemiBatchVerifier, "CHUNK_TIMEOUT_SEC", 0.1)

    successful, failed = await verifier.verify(ports, ssh_client=mocker.Mock(), host="1.2.3.4")

    assert failed == ports[:10]
    assert successful == ports[10:]
