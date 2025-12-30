import pytest

from services.executor_connectivity.models import DindProbeResult, PortPair, PortProbeResult
from services.executor_connectivity.orchestrator import ConnectivityOrchestrator
from services.executor_connectivity.dind_probe import DindProbe
from services.executor_connectivity.port_probe import PortProbe
from services.executor_connectivity.port_selector import PortSelector


@pytest.mark.asyncio
async def test_orchestrator_success(sample_executor_info, mock_ssh_client, mocker):
    selected_ports = [PortPair(9000, 9000), PortPair(9001, 9001)]

    port_selector = mocker.Mock(spec=PortSelector)
    port_selector.select.return_value = selected_ports

    port_probe = mocker.Mock(spec=PortProbe)
    port_probe.probe = mocker.AsyncMock(
        return_value=PortProbeResult(successful=tuple(selected_ports), failed=tuple())
    )

    dind_probe = mocker.Mock(spec=DindProbe)
    dind_probe.verify = mocker.AsyncMock(
        return_value=DindProbeResult(
            success=True,
            sysbox_runtime=True,
            port=selected_ports[0],
            log_text="ok",
        )
    )

    orchestrator = ConnectivityOrchestrator(
        port_selector,
        port_probe,
        dind_probe,
    )

    result = await orchestrator.verify(
        executor_info=sample_executor_info,
        miner_hotkey="miner",
        sysbox_runtime=False,
        rented_ports=[],
        ssh_client=mock_ssh_client,
    )

    assert result.status == "ok"
    assert result.dind_ok is True
    assert result.sysbox_runtime is True
    assert len(result.successful_ports) == 2

    port_selector.select.assert_called_once()
    port_probe.probe.assert_called_once()
    dind_probe.verify.assert_called_once()


@pytest.mark.asyncio
async def test_orchestrator_no_ports(sample_executor_info, mock_ssh_client, mocker):
    port_selector = mocker.Mock(spec=PortSelector)
    port_selector.select.return_value = []

    port_probe = mocker.Mock(spec=PortProbe)
    port_probe.probe = mocker.AsyncMock()

    dind_probe = mocker.Mock(spec=DindProbe)
    dind_probe.verify = mocker.AsyncMock()

    orchestrator = ConnectivityOrchestrator(
        port_selector,
        port_probe,
        dind_probe,
    )

    result = await orchestrator.verify(
        executor_info=sample_executor_info,
        miner_hotkey="miner",
        sysbox_runtime=False,
        rented_ports=[],
        ssh_client=mock_ssh_client,
    )

    assert result.status == "no_ports"
    assert result.successful_ports == tuple()
    assert result.failed_ports == tuple()
    assert result.dind_port is None
    assert result.dind_ok is False

    port_selector.select.assert_called_once()
    port_probe.probe.assert_not_called()
    dind_probe.verify.assert_not_called()


@pytest.mark.asyncio
async def test_orchestrator_probe_fails_dind_ok(sample_executor_info, mock_ssh_client, mocker):
    selected_ports = [PortPair(9000, 9000), PortPair(9001, 9001)]

    port_selector = mocker.Mock(spec=PortSelector)
    port_selector.select.return_value = selected_ports

    port_probe = mocker.Mock(spec=PortProbe)
    port_probe.probe = mocker.AsyncMock(
        return_value=PortProbeResult(successful=tuple(), failed=tuple(selected_ports))
    )

    dind_probe = mocker.Mock(spec=DindProbe)
    dind_probe.verify = mocker.AsyncMock(
        return_value=DindProbeResult(
            success=True,
            sysbox_runtime=True,
            port=selected_ports[0],
            log_text="ok",
        )
    )

    orchestrator = ConnectivityOrchestrator(
        port_selector,
        port_probe,
        dind_probe,
    )

    result = await orchestrator.verify(
        executor_info=sample_executor_info,
        miner_hotkey="miner",
        sysbox_runtime=False,
        rented_ports=[],
        ssh_client=mock_ssh_client,
    )

    assert result.status == "ok"
    assert result.dind_ok is True
    assert len(result.successful_ports) == 1
    assert result.successful_ports[0] in result.selected_ports

    port_selector.select.assert_called_once()
    port_probe.probe.assert_called_once()
    dind_probe.verify.assert_called_once()


@pytest.mark.asyncio
async def test_orchestrator_probe_ok_dind_fails(sample_executor_info, mock_ssh_client, mocker):
    selected_ports = [PortPair(9000, 9000), PortPair(9001, 9001)]

    port_selector = mocker.Mock(spec=PortSelector)
    port_selector.select.return_value = selected_ports

    port_probe = mocker.Mock(spec=PortProbe)
    port_probe.probe = mocker.AsyncMock(
        return_value=PortProbeResult(successful=tuple(selected_ports), failed=tuple())
    )

    dind_probe = mocker.Mock(spec=DindProbe)
    dind_probe.verify = mocker.AsyncMock(
        return_value=DindProbeResult(
            success=False,
            sysbox_runtime=False,
            port=selected_ports[0],
            log_text="fail",
        )
    )

    orchestrator = ConnectivityOrchestrator(
        port_selector,
        port_probe,
        dind_probe,
    )

    result = await orchestrator.verify(
        executor_info=sample_executor_info,
        miner_hotkey="miner",
        sysbox_runtime=True,
        rented_ports=[],
        ssh_client=mock_ssh_client,
    )

    assert result.status == "ok"
    assert result.dind_ok is False
    assert result.sysbox_runtime is False
    assert len(result.failed_ports) == 1
    assert result.failed_ports[0] in result.selected_ports

    port_selector.select.assert_called_once()
    port_probe.probe.assert_called_once()
    dind_probe.verify.assert_called_once()
