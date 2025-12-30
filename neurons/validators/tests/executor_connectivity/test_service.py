import pytest

from services.executor_connectivity_service import ExecutorConnectivityService, PortPair
from services.executor_connectivity.models import PortVerificationResult


@pytest.mark.asyncio
async def test_verify_ports_successful_flow(
    mock_ssh_client,
    sample_executor_info,
    mocker,
):
    miner_hotkey = "test_miner"
    rented_ports = [8000, 8001]
    rented_pod_names = ["pod_1", "pod_2"]

    selected = (PortPair(9000, 9000), PortPair(9001, 9001), PortPair(9002, 9002))
    successful = (PortPair(9001, 9001), PortPair(9002, 9002))

    cleanup_service = mocker.Mock()
    cleanup_service.cleanup = mocker.AsyncMock()

    orchestrator = mocker.Mock()
    orchestrator.verify = mocker.AsyncMock(
        return_value=PortVerificationResult(
            selected_ports=selected,
            successful_ports=successful,
            failed_ports=tuple(),
            dind_port=selected[0],
            dind_ok=True,
            sysbox_runtime=False,
            status="ok",
        )
    )

    persister = mocker.Mock()
    persister.save = mocker.AsyncMock()

    executor_service = ExecutorConnectivityService(
        orchestrator=orchestrator,
        persister=persister,
        cleanup_service=cleanup_service,
    )

    result = await executor_service.verify_ports(
        mock_ssh_client,
        miner_hotkey,
        sample_executor_info,
        rented_ports=rented_ports,
        rented_pod_names=rented_pod_names,
    )

    assert result.status == "ok"
    assert result.elapsed_sec is not None

    cleanup_service.cleanup.assert_called_once_with(
        mock_ssh_client,
        rented_pod_names,
        "container_",
    )
    orchestrator.verify.assert_called_once()
    persister.save.assert_called_once()


@pytest.mark.asyncio
async def test_verify_ports_non_ok_status_skips_persist(
    mock_ssh_client,
    sample_executor_info,
    mocker,
):
    miner_hotkey = "test_miner"

    selected = (PortPair(9000, 9000),)

    cleanup_service = mocker.Mock()
    cleanup_service.cleanup = mocker.AsyncMock()

    orchestrator = mocker.Mock()
    orchestrator.verify = mocker.AsyncMock(
        return_value=PortVerificationResult(
            selected_ports=selected,
            successful_ports=tuple(),
            failed_ports=selected,
            dind_port=selected[0],
            dind_ok=False,
            sysbox_runtime=False,
            status="no_working_ports",
        )
    )

    persister = mocker.Mock()
    persister.save = mocker.AsyncMock()

    executor_service = ExecutorConnectivityService(
        orchestrator=orchestrator,
        persister=persister,
        cleanup_service=cleanup_service,
    )

    result = await executor_service.verify_ports(
        mock_ssh_client,
        miner_hotkey,
        sample_executor_info,
    )

    assert result.status == "no_working_ports"
    assert result.elapsed_sec is not None

    cleanup_service.cleanup.assert_called_once_with(
        mock_ssh_client,
        [],
        "container_",
    )
    orchestrator.verify.assert_called_once()
    persister.save.assert_not_called()
