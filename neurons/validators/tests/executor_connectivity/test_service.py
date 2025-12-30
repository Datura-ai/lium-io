import pytest

from services.executor_connectivity import (
    ContainerCleanupProtocol,
    OrchestratorProtocol,
    ResultPersisterProtocol,
)
from services.executor_connectivity_service import ExecutorConnectivityService, PortPair
from services.executor_connectivity.models import PortVerificationResult


class FakeCleanup(ContainerCleanupProtocol):
    def __init__(self):
        self.calls = []

    async def cleanup(self, ssh, active_pod_names, container_name_prefix):
        self.calls.append((ssh, active_pod_names, container_name_prefix))


class FakeOrchestrator(OrchestratorProtocol):
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def verify(
        self,
        *,
        ssh_client,
        executor_info,
        miner_hotkey: str,
        sysbox_runtime: bool,
        rented_ports: list[int] | None,
    ):
        self.calls.append(
            {
                "ssh_client": ssh_client,
                "executor_info": executor_info,
                "miner_hotkey": miner_hotkey,
                "sysbox_runtime": sysbox_runtime,
                "rented_ports": rented_ports,
            }
        )
        return self.result


class FakePersister(ResultPersisterProtocol):
    def __init__(self):
        self.calls = []

    async def save(self, result, executor_id, miner_hotkey):
        self.calls.append((result, executor_id, miner_hotkey))


@pytest.fixture
def fake_cleanup():
    return FakeCleanup()


@pytest.fixture
def mock_ssh_service():
    return object()


@pytest.fixture
def fake_persister():
    return FakePersister()


@pytest.fixture
def orchestrator_factory():
    def _factory(result):
        return FakeOrchestrator(result)

    return _factory


@pytest.mark.asyncio
async def test_verify_ports_successful_flow(
    mock_redis_service,
    port_mapping_dao,
    mock_ssh_service,
    mock_ssh_client,
    sample_executor_info,
    fake_cleanup,
    fake_persister,
    orchestrator_factory,
):
    """Test verify_ports uses injected services and returns a result."""
    miner_hotkey = "test_miner"
    rented_ports = [8000, 8001]
    rented_pod_names = ["pod_1", "pod_2"]

    selected = (PortPair(9000, 9000), PortPair(9001, 9001), PortPair(9002, 9002))
    successful = (PortPair(9001, 9001), PortPair(9002, 9002))

    fake_orchestrator = orchestrator_factory(
        PortVerificationResult(
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
        mock_redis_service,
        port_mapping_dao,
        mock_ssh_service,
        orchestrator=fake_orchestrator,
        persister=fake_persister,
        cleanup_service=fake_cleanup,
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
    assert fake_cleanup.calls == [(mock_ssh_client, rented_pod_names, "container_")]
    assert fake_orchestrator.calls[0]["rented_ports"] == rented_ports
    assert fake_orchestrator.calls[0]["miner_hotkey"] == miner_hotkey
    assert fake_persister.calls[0][1] == sample_executor_info.uuid


@pytest.mark.asyncio
async def test_verify_ports_non_ok_status_skips_persist(
    mock_redis_service,
    port_mapping_dao,
    mock_ssh_service,
    mock_ssh_client,
    sample_executor_info,
    fake_cleanup,
    fake_persister,
    orchestrator_factory,
):
    """Test verify_ports does not persist when status is not ok."""
    miner_hotkey = "test_miner"

    selected = (PortPair(9000, 9000),)

    fake_orchestrator = orchestrator_factory(
        PortVerificationResult(
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
        mock_redis_service,
        port_mapping_dao,
        mock_ssh_service,
        orchestrator=fake_orchestrator,
        persister=fake_persister,
        cleanup_service=fake_cleanup,
    )

    result = await executor_service.verify_ports(
        mock_ssh_client,
        miner_hotkey,
        sample_executor_info,
    )

    assert result.status == "no_working_ports"
    assert result.elapsed_sec is not None
    assert fake_cleanup.calls == [(mock_ssh_client, [], "container_")]
    assert fake_persister.calls == []
