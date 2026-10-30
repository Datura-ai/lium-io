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


def _batch_verifier(mocker, test_many, external=None):
    from services.executor_connectivity.port_verifiers import BatchVerifier

    runner = mocker.Mock()
    runner.run = mocker.AsyncMock(return_value=CONTAINER_STARTED)
    runner.cleanup = mocker.AsyncMock()
    port_tester = mocker.Mock()
    port_tester.test_many = mocker.AsyncMock(side_effect=test_many)
    mocker.patch.object(BatchVerifier, "REPROBE_DELAY_SEC", 0)
    return BatchVerifier(port_tester, runner, external), port_tester


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


@pytest.mark.asyncio
async def test_batch_late_bind_with_enforced_outside_prober_is_not_externally_blocked(mocker):
    from core.config import settings

    mocker.patch.object(settings, "EXTERNAL_PORT_PROBE_ENFORCE", True)
    ports = [PortPair(9000 + i, 9000 + i) for i in range(10)]
    calls = []

    def late_listeners(_s, _h, probed, *_a):
        calls.append(list(probed))
        return (probed[:1], probed[1:]) if len(calls) == 1 else (probed, [])

    external = mocker.Mock()
    # the outside prober is refused: the listeners were not bound yet
    external.reached = mocker.AsyncMock(return_value=[])
    sample_late_ports = lambda ps, k: ps[1 : 1 + k]  # noqa: E731
    mocker.patch("services.executor_connectivity.port_verifiers.random.sample", sample_late_ports)
    verifier, _ = _batch_verifier(mocker, late_listeners, external)

    result = await verifier.verify(ports, ssh_client=mocker.Mock(), host="1.2.3.4")

    assert result.external != "blocked"
    assert sorted(result.successful, key=lambda p: p.internal) == ports


@pytest.mark.asyncio
async def test_batch_reprobes_two_of_three_below_the_port_floor(mocker):
    ports = [PortPair(9000 + i, 9000 + i) for i in range(3)]
    calls = []

    def late_listeners(_s, _h, probed, *_a):
        calls.append(list(probed))
        return (probed[:2], probed[2:]) if len(calls) == 1 else (probed, [])

    verifier, _ = _batch_verifier(mocker, late_listeners)

    result = await verifier.verify(ports, ssh_client=mocker.Mock(), host="1.2.3.4")

    assert calls == [ports, ports[2:]]
    assert sorted(result.successful, key=lambda p: p.internal) == ports


@pytest.mark.asyncio
async def test_batch_reprobe_gives_up_after_its_budget(mocker):
    from services.executor_connectivity.port_verifiers import BatchVerifier

    ports = [PortPair(9000 + i, 9000 + i) for i in range(3)]
    verifier, port_tester = _batch_verifier(mocker, lambda _s, _h, probed, *_a: ([], probed))

    result = await verifier.verify(ports, ssh_client=mocker.Mock(), host="1.2.3.4")

    # each of the 2 attempts: one probe plus REPROBE_ROUNDS re-probes, then the attempt fails
    assert port_tester.test_many.await_count == 2 * (1 + BatchVerifier.REPROBE_ROUNDS)
    assert result.successful == [] and result.failed == ports
