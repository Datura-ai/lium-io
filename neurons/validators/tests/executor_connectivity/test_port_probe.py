import pytest

from services.executor_connectivity.models import BatchResult, PortPair
from services.executor_connectivity.port_probe import PortProbe


@pytest.mark.asyncio
async def test_port_probe_uses_batch_when_successful(mocker):
    ports = [PortPair(9000, 9000), PortPair(9001, 9001)]

    batch = mocker.Mock()
    batch.verify = mocker.AsyncMock(return_value=BatchResult(ports, []))
    semi = mocker.Mock()
    semi.verify = mocker.AsyncMock()
    fallback = mocker.Mock()
    fallback.verify = mocker.AsyncMock()

    probe = PortProbe(batch, semi, fallback)

    result = await probe.probe(ports, ssh_client=mocker.Mock(), host="127.0.0.1")

    assert list(result.successful) == ports
    assert list(result.failed) == []
    batch.verify.assert_called_once()
    semi.verify.assert_not_called()
    fallback.verify.assert_not_called()


@pytest.mark.asyncio
async def test_port_probe_uses_semi_batch_when_batch_empty(mocker):
    ports = [PortPair(9000, 9000), PortPair(9001, 9001)]

    batch = mocker.Mock()
    batch.verify = mocker.AsyncMock(return_value=BatchResult([], ports))
    semi = mocker.Mock()
    semi.verify = mocker.AsyncMock(return_value=(ports, []))
    fallback = mocker.Mock()
    fallback.verify = mocker.AsyncMock()

    probe = PortProbe(batch, semi, fallback)

    result = await probe.probe(ports, ssh_client=mocker.Mock(), host="127.0.0.1")

    assert list(result.successful) == ports
    assert list(result.failed) == []
    batch.verify.assert_called_once()
    semi.verify.assert_called_once()
    fallback.verify.assert_not_called()


@pytest.mark.asyncio
async def test_port_probe_keeps_partial_semi_batch_result(mocker):
    # DAH-2527: with chunked publication a busy port leaves the tier partially successful — that
    # result must be carried forward, not thrown away for the 10-port sequential tier.
    ports = [PortPair(9000 + i, 9000 + i) for i in range(3)]

    batch = mocker.Mock()
    batch.verify = mocker.AsyncMock(return_value=BatchResult([], ports))
    semi = mocker.Mock()
    semi.verify = mocker.AsyncMock(return_value=(ports[:2], ports[2:]))
    fallback = mocker.Mock()
    fallback.verify = mocker.AsyncMock()

    probe = PortProbe(batch, semi, fallback)

    result = await probe.probe(ports, ssh_client=mocker.Mock(), host="127.0.0.1")

    assert list(result.successful) == ports[:2]
    assert list(result.failed) == ports[2:]
    fallback.verify.assert_not_called()


@pytest.mark.asyncio
async def test_port_probe_falls_through_to_fallback_when_batch_and_semi_empty(mocker):
    ports = [PortPair(9000, 9000), PortPair(9001, 9001)]

    batch = mocker.Mock()
    batch.verify = mocker.AsyncMock(return_value=BatchResult([], ports))
    semi = mocker.Mock()
    semi.verify = mocker.AsyncMock(return_value=([], ports))
    fallback = mocker.Mock()
    fallback.verify = mocker.AsyncMock(return_value=(ports[:1], ports[1:]))

    probe = PortProbe(batch, semi, fallback)

    result = await probe.probe(ports, ssh_client=mocker.Mock(), host="127.0.0.1")

    assert list(result.successful) == ports[:1]
    assert list(result.failed) == ports[1:]
    batch.verify.assert_called_once()
    semi.verify.assert_called_once()
    fallback.verify.assert_called_once()


@pytest.mark.asyncio
async def test_port_probe_fallback_tests_every_port_a_split_node_needs(mocker):
    ports = [PortPair(9000 + i, 9000 + i) for i in range(30)]
    batch = mocker.Mock()
    batch.verify = mocker.AsyncMock(return_value=BatchResult([], ports))
    semi = mocker.Mock()
    semi.verify = mocker.AsyncMock(return_value=([], ports))
    fallback = mocker.Mock()
    fallback.verify = mocker.AsyncMock(return_value=(ports[:24], []))

    await PortProbe(batch, semi, fallback).probe(ports, ssh_client=mocker.Mock(), host="127.0.0.1", min_ports=24)

    assert batch.verify.call_args.kwargs["min_ports"] == 24
    assert fallback.verify.call_args.kwargs["max_ports"] == 24


@pytest.mark.asyncio
async def test_port_probe_runs_next_tier_while_a_split_node_is_short(mocker):
    # a node that needs 24 ports (24 one-GPU splits): 3 host-network successes must not stop the published-port tiers
    ports = [PortPair(9000 + i, 9000 + i) for i in range(30)]
    batch = mocker.Mock()
    batch.verify = mocker.AsyncMock(return_value=BatchResult(ports[:3], ports[3:]))
    semi = mocker.Mock()
    semi.verify = mocker.AsyncMock(return_value=(ports[:10], ports[10:]))
    fallback = mocker.Mock()
    fallback.verify = mocker.AsyncMock(return_value=(ports[:24], ports[24:]))

    result = await PortProbe(batch, semi, fallback).probe(
        ports, ssh_client=mocker.Mock(), host="127.0.0.1", min_ports=24
    )

    assert list(result.successful) == ports[:24]
    assert result.tier == "fallback"
    semi.verify.assert_called_once()


@pytest.mark.asyncio
async def test_port_probe_keeps_the_best_tier_for_a_short_split_node(mocker):
    ports = [PortPair(9000 + i, 9000 + i) for i in range(30)]
    batch = mocker.Mock()
    batch.verify = mocker.AsyncMock(return_value=BatchResult(ports[:5], ports[5:]))
    semi = mocker.Mock()
    semi.verify = mocker.AsyncMock(return_value=(ports[:2], ports[2:]))
    fallback = mocker.Mock()
    fallback.verify = mocker.AsyncMock(return_value=([], ports))

    result = await PortProbe(batch, semi, fallback).probe(
        ports, ssh_client=mocker.Mock(), host="127.0.0.1", min_ports=24
    )

    assert list(result.successful) == ports[:5]
    assert result.tier == "batch"


@pytest.mark.asyncio
async def test_port_probe_stops_once_every_selected_port_answered(mocker):
    # fewer ports selected than a split node needs (a rental holds the rest): all 21 answered, so no later tier runs
    ports = [PortPair(9000 + i, 9000 + i) for i in range(21)]
    batch = mocker.Mock()
    batch.verify = mocker.AsyncMock(return_value=BatchResult(ports, []))
    semi = mocker.Mock()
    semi.verify = mocker.AsyncMock()
    fallback = mocker.Mock()
    fallback.verify = mocker.AsyncMock()

    result = await PortProbe(batch, semi, fallback).probe(
        ports, ssh_client=mocker.Mock(), host="127.0.0.1", min_ports=24
    )

    assert list(result.successful) == ports
    assert result.tier == "batch"
    semi.verify.assert_not_called()
    fallback.verify.assert_not_called()
