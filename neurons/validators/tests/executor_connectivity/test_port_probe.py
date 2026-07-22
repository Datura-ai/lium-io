import pytest

from services.executor_connectivity.models import PortPair
from services.executor_connectivity.port_probe import PortProbe


@pytest.mark.asyncio
async def test_port_probe_uses_batch_when_successful(mocker):
    ports = [PortPair(9000, 9000), PortPair(9001, 9001)]

    batch = mocker.Mock()
    batch.verify = mocker.AsyncMock(return_value=(ports, []))
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
    batch.verify = mocker.AsyncMock(return_value=([], ports))
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
async def test_port_probe_falls_through_to_fallback_when_batch_and_semi_empty(mocker):
    ports = [PortPair(9000, 9000), PortPair(9001, 9001)]

    batch = mocker.Mock()
    batch.verify = mocker.AsyncMock(return_value=([], ports))
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
