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


def _hung_probe(mocker, ports):
    batch = mocker.Mock()
    batch.verify = mocker.AsyncMock(return_value=BatchResult([], ports, start_hung=True))
    semi = mocker.Mock()
    semi.verify = mocker.AsyncMock(return_value=([], ports))
    fallback = mocker.Mock()
    fallback.verify = mocker.AsyncMock(return_value=([], ports))
    return PortProbe(batch, semi, fallback), semi, fallback


@pytest.mark.asyncio
async def test_port_probe_skips_later_tiers_after_a_hung_start_when_the_flag_is_on(mocker):
    ports = [PortPair(9000, 9000)]
    mocker.patch("services.executor_connectivity.port_probe.settings.PORT_PROBE_SKIP_TIERS_ON_HUNG_START", True)
    probe, semi, fallback = _hung_probe(mocker, ports)

    result = await probe.probe(ports, ssh_client=mocker.Mock(), host="127.0.0.1")

    assert list(result.failed) == ports
    assert result.tier == "batch"
    semi.verify.assert_not_called()
    fallback.verify.assert_not_called()


@pytest.mark.asyncio
async def test_port_probe_only_logs_the_skip_after_a_hung_start_when_the_flag_is_off(mocker):
    ports = [PortPair(9000, 9000)]
    mocker.patch("services.executor_connectivity.port_probe.settings.PORT_PROBE_SKIP_TIERS_ON_HUNG_START", False)
    log = mocker.patch("services.executor_connectivity.port_probe.logger.info")
    probe, semi, fallback = _hung_probe(mocker, ports)

    result = await probe.probe(ports, ssh_client=mocker.Mock(), host="127.0.0.1")

    assert result.tier == "fallback"
    semi.verify.assert_called_once()
    fallback.verify.assert_called_once()
    message = log.call_args.args[0]
    assert message.message == "port check would skip later tiers"
    assert message.extra["later_tiers_found_ports"] is False
    assert message.extra["later_tiers_port_count"] == 0
    assert message.extra["later_tiers_seconds"] >= 0
