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

    executor_service = ExecutorConnectivityService(
        orchestrator=orchestrator,
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

    orchestrator.verify.assert_called_once()


@pytest.mark.asyncio
async def test_verify_ports_non_ok_status_skips_persist(
    mock_ssh_client,
    sample_executor_info,
    mocker,
):
    miner_hotkey = "test_miner"

    selected = (PortPair(9000, 9000),)

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

    executor_service = ExecutorConnectivityService(
        orchestrator=orchestrator,
    )

    result = await executor_service.verify_ports(
        mock_ssh_client,
        miner_hotkey,
        sample_executor_info,
    )

    assert result.status == "no_working_ports"
    assert result.elapsed_sec is not None

    orchestrator.verify.assert_called_once()


@pytest.mark.asyncio
async def test_verify_ports_merges_rented_and_filler_ports(
    mock_ssh_client,
    sample_executor_info,
    mocker,
):
    # DAH-2527: filler ports arrive separately so rented_ports keeps meaning "has a rental",
    # but the orchestrator below must receive them as one set of taken ports
    orchestrator = mocker.Mock()
    orchestrator.verify = mocker.AsyncMock(
        return_value=PortVerificationResult(
            selected_ports=tuple(),
            successful_ports=tuple(),
            failed_ports=tuple(),
            dind_port=None,
            dind_ok=False,
            sysbox_runtime=False,
            status="no_ports",
        )
    )

    executor_service = ExecutorConnectivityService(orchestrator=orchestrator)

    await executor_service.verify_ports(
        mock_ssh_client,
        "test_miner",
        sample_executor_info,
        rented_ports=[40100],
        filler_ports=[40001, 40003],
    )

    passed = orchestrator.verify.call_args.kwargs
    assert set(passed["unavailable_ports"]) == {40001, 40003, 40100}
    assert "rented_ports" not in passed
