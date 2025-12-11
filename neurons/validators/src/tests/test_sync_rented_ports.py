"""
Tests for sync_rented_ports service.

TDD approach: tests written first, then implementation.
"""

import pytest
from uuid import uuid4, UUID

from protocol.vc_protocol.compute_requests import (
    RentedContainer,
    RentedMachine,
    RentedMachineResponse,
)
from services.sync_rented_ports import SyncRentedPortsResult, sync_rented_ports


class DummyPortMappingDao:
    """Mock DAO for port mapping operations."""

    def __init__(
        self,
        rented_pod_ids: set[UUID] | None = None,
        release_counts: dict[UUID, int] | None = None,
    ):
        self.rented_pod_ids = rented_pod_ids or set()
        self.release_counts = release_counts or {}
        self.released_pods: list[UUID] = []

    async def get_rented_pod_ids_older_than(self, minutes: int = 10) -> set[UUID]:
        return self.rented_pod_ids

    async def release_ports_for_pod(self, pod_id: UUID) -> int:
        self.released_pods.append(pod_id)
        return self.release_counts.get(pod_id, 0)


def make_rented_machine_response(
    pod_ids: list[str] | None = None,
    banned_guids: list[str] | None = None,
) -> RentedMachineResponse:
    """Helper to create RentedMachineResponse with given pod_ids."""
    machines = []
    if pod_ids:
        containers = [
            RentedContainer(name=f"container_{pid[:8]}", pod_id=pid)
            for pid in pod_ids
        ]
        machines.append(
            RentedMachine(
                miner_hotkey="5DHgdom...",
                executor_id="executor-123",
                executor_ip_address="192.168.1.1",
                executor_ip_port="4000",
                containers=containers,
                owner_flag=False,
            )
        )
    return RentedMachineResponse(machines=machines, banned_guids=banned_guids or [])


@pytest.mark.asyncio
async def test_sync_no_stale_pods():
    """When all local pods are in backend response, nothing should be released."""
    # Arrange
    pod_id_1 = uuid4()
    pod_id_2 = uuid4()
    backend_response = make_rented_machine_response(pod_ids=[str(pod_id_1), str(pod_id_2)])
    dao = DummyPortMappingDao(
        rented_pod_ids={pod_id_1, pod_id_2},
        release_counts={},
    )

    # Act
    result = await sync_rented_ports(backend_response, dao)

    # Assert
    # No ports released because all local pods exist on backend
    assert result.released_port_count == 0
    assert result.stale_pod_ids == set()
    assert result.unknown_pod_ids == set()
    assert dao.released_pods == []


@pytest.mark.asyncio
async def test_sync_releases_stale_pods():
    """Pods in local DB but not in backend should be released."""
    # Arrange
    pod_id_active = uuid4()
    pod_id_stale_1 = uuid4()
    pod_id_stale_2 = uuid4()
    backend_response = make_rented_machine_response(pod_ids=[str(pod_id_active)])
    dao = DummyPortMappingDao(
        rented_pod_ids={pod_id_active, pod_id_stale_1, pod_id_stale_2},
        release_counts={pod_id_stale_1: 5, pod_id_stale_2: 3},
    )

    # Act
    result = await sync_rented_ports(backend_response, dao)

    # Assert
    # 8 ports released (5 + 3) for 2 stale pods
    assert result.released_port_count == 8
    assert result.stale_pod_ids == {pod_id_stale_1, pod_id_stale_2}
    assert result.unknown_pod_ids == set()
    assert set(dao.released_pods) == {pod_id_stale_1, pod_id_stale_2}


@pytest.mark.asyncio
async def test_sync_warns_about_unknown_pods():
    """Pods in backend response but not in local DB should be in unknown_pod_ids."""
    # Arrange
    pod_id_known = uuid4()
    pod_id_unknown = uuid4()
    backend_response = make_rented_machine_response(pod_ids=[str(pod_id_known), str(pod_id_unknown)])
    dao = DummyPortMappingDao(rented_pod_ids={pod_id_known})

    # Act
    result = await sync_rented_ports(backend_response, dao)

    # Assert
    # No releases, but unknown pod detected (backend has pod we don't know about)
    assert result.released_port_count == 0
    assert result.stale_pod_ids == set()
    assert result.unknown_pod_ids == {pod_id_unknown}


@pytest.mark.asyncio
async def test_sync_handles_stale_and_unknown_pods():
    """Both stale (local only) and unknown (backend only) pods handled correctly."""
    # Arrange
    pod_id_synced = uuid4()
    pod_id_stale = uuid4()
    pod_id_unknown = uuid4()
    backend_response = make_rented_machine_response(pod_ids=[str(pod_id_synced), str(pod_id_unknown)])
    dao = DummyPortMappingDao(
        rented_pod_ids={pod_id_synced, pod_id_stale},
        release_counts={pod_id_stale: 10},
    )

    # Act
    result = await sync_rented_ports(backend_response, dao)

    # Assert
    # Stale pod released, unknown pod detected
    assert result.released_port_count == 10
    assert result.stale_pod_ids == {pod_id_stale}
    assert result.unknown_pod_ids == {pod_id_unknown}
    assert dao.released_pods == [pod_id_stale]


@pytest.mark.asyncio
async def test_sync_empty_backend_releases_all():
    """When backend returns no pods, all local pods should be released."""
    # Arrange
    pod_id_1 = uuid4()
    pod_id_2 = uuid4()
    backend_response = make_rented_machine_response(pod_ids=[])
    dao = DummyPortMappingDao(
        rented_pod_ids={pod_id_1, pod_id_2},
        release_counts={pod_id_1: 2, pod_id_2: 3},
    )

    # Act
    result = await sync_rented_ports(backend_response, dao)

    # Assert
    # All local pods are stale because backend has none
    assert result.released_port_count == 5
    assert result.stale_pod_ids == {pod_id_1, pod_id_2}


@pytest.mark.asyncio
async def test_sync_empty_local_db():
    """When local DB has no rented pods, only unknown pods should be detected."""
    # Arrange
    pod_id_unknown = uuid4()
    backend_response = make_rented_machine_response(pod_ids=[str(pod_id_unknown)])
    dao = DummyPortMappingDao(rented_pod_ids=set())

    # Act
    result = await sync_rented_ports(backend_response, dao)

    # Assert
    # No releases, but unknown pod detected
    assert result.released_port_count == 0
    assert result.stale_pod_ids == set()
    assert result.unknown_pod_ids == {pod_id_unknown}


@pytest.mark.asyncio
async def test_sync_custom_threshold():
    """Threshold parameter should be passed to DAO."""
    # Arrange
    backend_response = make_rented_machine_response(pod_ids=[])
    threshold_received = None

    class TrackingDao(DummyPortMappingDao):
        async def get_rented_pod_ids_older_than(self, minutes: int = 10) -> set[UUID]:
            nonlocal threshold_received
            threshold_received = minutes
            return set()

    dao = TrackingDao()

    # Act
    await sync_rented_ports(backend_response, dao, threshold_minutes=30)

    # Assert
    # Custom threshold was passed to DAO
    assert threshold_received == 30


@pytest.mark.asyncio
async def test_sync_multiple_containers_per_machine():
    """Multiple containers on same executor should all be counted."""
    # Arrange
    pod_id_1 = uuid4()
    pod_id_2 = uuid4()
    pod_id_3 = uuid4()
    pod_id_stale = uuid4()
    containers = [
        RentedContainer(name="container_1", pod_id=str(pod_id_1)),
        RentedContainer(name="container_2", pod_id=str(pod_id_2)),
        RentedContainer(name="container_3", pod_id=str(pod_id_3)),
    ]
    machine = RentedMachine(
        miner_hotkey="5DHgdom...",
        executor_id="executor-123",
        executor_ip_address="192.168.1.1",
        executor_ip_port="4000",
        containers=containers,
        owner_flag=False,
    )
    backend_response = RentedMachineResponse(machines=[machine], banned_guids=[])
    dao = DummyPortMappingDao(
        rented_pod_ids={pod_id_1, pod_id_2, pod_id_3, pod_id_stale},
        release_counts={pod_id_stale: 7},
    )

    # Act
    result = await sync_rented_ports(backend_response, dao)

    # Assert
    # Only stale pod released, all 3 backend pods recognized
    assert result.released_port_count == 7
    assert result.stale_pod_ids == {pod_id_stale}
    assert result.unknown_pod_ids == set()


@pytest.mark.asyncio
async def test_sync_multiple_machines():
    """Pods from multiple machines should all be considered."""
    # Arrange
    pod_id_1 = uuid4()
    pod_id_2 = uuid4()
    pod_id_stale = uuid4()
    machine_1 = RentedMachine(
        miner_hotkey="miner-1",
        executor_id="executor-1",
        executor_ip_address="192.168.1.1",
        executor_ip_port="4000",
        containers=[RentedContainer(name="c1", pod_id=str(pod_id_1))],
        owner_flag=False,
    )
    machine_2 = RentedMachine(
        miner_hotkey="miner-2",
        executor_id="executor-2",
        executor_ip_address="192.168.1.2",
        executor_ip_port="4001",
        containers=[RentedContainer(name="c2", pod_id=str(pod_id_2))],
        owner_flag=False,
    )
    backend_response = RentedMachineResponse(machines=[machine_1, machine_2], banned_guids=[])
    dao = DummyPortMappingDao(
        rented_pod_ids={pod_id_1, pod_id_2, pod_id_stale},
        release_counts={pod_id_stale: 4},
    )

    # Act
    result = await sync_rented_ports(backend_response, dao)

    # Assert
    # Pods from both machines recognized, only stale pod released
    assert result.released_port_count == 4
    assert result.stale_pod_ids == {pod_id_stale}
