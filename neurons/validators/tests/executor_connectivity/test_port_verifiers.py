import pytest

from services.executor_connectivity.models import ContainerStartResult, PortPair
from services.executor_connectivity.port_verifiers import SemiBatchVerifier


CONTAINER_STARTED = ContainerStartResult(ok=True, container_id="c", status="Up")
CONTAINER_START_FAILED = ContainerStartResult(
    ok=False, container_id=None, status="port is already allocated"
)


@pytest.mark.asyncio
async def test_semi_batch_publishes_a_chunk_in_one_container(mocker):
    ports = [PortPair(9000, 9000), PortPair(9001, 9001), PortPair(9002, 9002)]

    runner = mocker.Mock()
    runner.run = mocker.AsyncMock(return_value=CONTAINER_STARTED)
    runner.cleanup = mocker.AsyncMock()

    port_tester = mocker.Mock()
    port_tester.test_many = mocker.AsyncMock(return_value=(ports, []))

    verifier = SemiBatchVerifier(port_tester, runner)

    successful, failed = await verifier.verify(ports, ssh_client=mocker.Mock(), host="1.2.3.4")

    assert successful == ports
    assert failed == []
    runner.run.assert_called_once()
    runner.cleanup.assert_called_once()
    publish_flags = runner.run.call_args.args[3]
    assert publish_flags == "-p 9000:9000 -p 9001:9001 -p 9002:9002"
    port_tester.test_many.assert_awaited_once()


@pytest.mark.asyncio
async def test_semi_batch_maps_external_to_internal(mocker):
    ports = [PortPair(22, 40009), PortPair(8000, 40010)]

    runner = mocker.Mock()
    runner.run = mocker.AsyncMock(return_value=CONTAINER_STARTED)
    runner.cleanup = mocker.AsyncMock()

    port_tester = mocker.Mock()
    port_tester.test_many = mocker.AsyncMock(return_value=(ports, []))

    verifier = SemiBatchVerifier(port_tester, runner)

    await verifier.verify(ports, ssh_client=mocker.Mock(), host="1.2.3.4")

    publish_flags = runner.run.call_args.args[3]
    assert publish_flags == "-p 40009:22 -p 40010:8000"


@pytest.mark.asyncio
async def test_semi_batch_caps_at_max_ports(mocker):
    ports = [PortPair(9000 + i, 9000 + i) for i in range(80)]

    runner = mocker.Mock()
    runner.run = mocker.AsyncMock(return_value=CONTAINER_STARTED)
    runner.cleanup = mocker.AsyncMock()

    port_tester = mocker.Mock()
    port_tester.test_many = mocker.AsyncMock(side_effect=lambda session, host, tested, token, log_ctx=None: (tested, []))

    verifier = SemiBatchVerifier(port_tester, runner)

    successful, _ = await verifier.verify(ports, ssh_client=mocker.Mock(), host="1.2.3.4", max_ports=50)

    assert len(successful) == 50
    assert runner.run.await_count == 50 / SemiBatchVerifier.CHUNK_SIZE
    tested_ports = [port for call in port_tester.test_many.call_args_list for port in call.args[2]]
    assert tested_ports == ports[:50]


@pytest.mark.asyncio
async def test_semi_batch_returns_empty_when_no_chunk_can_start(mocker):
    ports = [PortPair(40000 + i, 40000 + i) for i in range(25)]

    runner = mocker.Mock()
    runner.run = mocker.AsyncMock(return_value=CONTAINER_START_FAILED)
    runner.cleanup = mocker.AsyncMock()

    port_tester = mocker.Mock()
    port_tester.test_many = mocker.AsyncMock()

    verifier = SemiBatchVerifier(port_tester, runner)

    successful, failed = await verifier.verify(ports, ssh_client=mocker.Mock(), host="1.2.3.4")

    assert successful == []
    assert failed == ports
    port_tester.test_many.assert_not_called()
    runner.cleanup.assert_not_called()


@pytest.mark.asyncio
async def test_semi_batch_busy_port_fails_only_its_own_chunk(mocker):
    # DAH-2527: docker refuses the whole `run` over one already-allocated port, and idle fillers
    # take the same low ports this tier probes. One busy port must cost one chunk, not the tier.
    ports = [PortPair(40000 + i, 40000 + i) for i in range(25)]

    runner = mocker.Mock()
    runner.run = mocker.AsyncMock(
        side_effect=lambda ssh_client, name, script, publish_flags, timeout: (
            CONTAINER_START_FAILED if "-p 40012:40012" in publish_flags else CONTAINER_STARTED
        )
    )
    runner.cleanup = mocker.AsyncMock()

    port_tester = mocker.Mock()
    port_tester.test_many = mocker.AsyncMock(side_effect=lambda session, host, tested, token, log_ctx=None: (tested, []))

    verifier = SemiBatchVerifier(port_tester, runner)

    successful, failed = await verifier.verify(ports, ssh_client=mocker.Mock(), host="1.2.3.4")

    assert successful == ports[:10] + ports[20:]
    assert failed == ports[10:20]


@pytest.mark.asyncio
async def test_semi_batch_returns_empty_when_run_raises(mocker):
    ports = [PortPair(9000, 9000), PortPair(9001, 9001)]

    runner = mocker.Mock()
    runner.run = mocker.AsyncMock(side_effect=ConnectionError("ssh channel closed"))
    runner.cleanup = mocker.AsyncMock()

    port_tester = mocker.Mock()
    port_tester.test_many = mocker.AsyncMock()

    verifier = SemiBatchVerifier(port_tester, runner)

    successful, failed = await verifier.verify(ports, ssh_client=mocker.Mock(), host="1.2.3.4")

    assert successful == []
    assert failed == ports
    port_tester.test_many.assert_not_called()
    runner.cleanup.assert_not_called()


@pytest.mark.asyncio
async def test_semi_batch_returns_empty_when_test_many_raises(mocker):
    ports = [PortPair(9000, 9000), PortPair(9001, 9001)]

    runner = mocker.Mock()
    runner.run = mocker.AsyncMock(return_value=CONTAINER_STARTED)
    runner.cleanup = mocker.AsyncMock()

    port_tester = mocker.Mock()
    port_tester.test_many = mocker.AsyncMock(side_effect=RuntimeError("probe blew up"))

    verifier = SemiBatchVerifier(port_tester, runner)

    successful, failed = await verifier.verify(ports, ssh_client=mocker.Mock(), host="1.2.3.4")

    assert successful == []
    assert failed == ports
    runner.cleanup.assert_called_once()


@pytest.mark.asyncio
async def test_semi_batch_survives_cleanup_failure(mocker):
    ports = [PortPair(9000, 9000), PortPair(9001, 9001)]

    runner = mocker.Mock()
    runner.run = mocker.AsyncMock(return_value=CONTAINER_STARTED)
    runner.cleanup = mocker.AsyncMock(side_effect=ConnectionError("ssh dropped"))

    port_tester = mocker.Mock()
    port_tester.test_many = mocker.AsyncMock(return_value=(ports, []))

    verifier = SemiBatchVerifier(port_tester, runner)

    successful, failed = await verifier.verify(ports, ssh_client=mocker.Mock(), host="1.2.3.4")

    assert successful == ports
    assert failed == []
