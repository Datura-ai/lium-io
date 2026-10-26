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


def _batch_verifier(mocker, test_many):
    from services.executor_connectivity.port_verifiers import BatchVerifier

    runner = mocker.Mock()
    runner.run = mocker.AsyncMock(return_value=CONTAINER_STARTED)
    runner.cleanup = mocker.AsyncMock()
    port_tester = mocker.Mock()
    port_tester.test_many = mocker.AsyncMock(side_effect=test_many)
    mocker.patch.object(BatchVerifier, "REPROBE_DELAY_SEC", 0)
    return BatchVerifier(port_tester, runner), port_tester


@pytest.mark.asyncio
async def test_batch_reprobes_missed_ports_once_when_most_fail(mocker):
    ports = [PortPair(9000 + i, 9000 + i) for i in range(10)]
    calls = []

    def late_listeners(_s, _h, probed, *_a):
        calls.append(list(probed))
        return (probed[:1], probed[1:]) if len(calls) == 1 else (probed, [])

    verifier, _ = _batch_verifier(mocker, late_listeners)

    result = await verifier.verify(ports, ssh_client=mocker.Mock(), host="1.2.3.4")

    assert calls == [ports, ports[1:]]
    assert sorted(result.successful, key=lambda p: p.internal) == ports
    assert result.failed == []


@pytest.mark.asyncio
async def test_batch_does_not_reprobe_when_most_ports_pass(mocker):
    ports = [PortPair(9000 + i, 9000 + i) for i in range(10)]
    verifier, port_tester = _batch_verifier(mocker, lambda _s, _h, probed, *_a: (probed[:8], probed[8:]))

    result = await verifier.verify(ports, ssh_client=mocker.Mock(), host="1.2.3.4")

    assert port_tester.test_many.await_count == 1
    assert result.failed == ports[8:]
