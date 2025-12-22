from datetime import datetime
from uuid import uuid4

import pytest

from daos.port_mapping_dao import PortMappingDao
from models.port_mapping import PortMapping
from .factories import create_port_mappings_batch


@pytest.fixture
def port_mapping_dao():
    """Create PortMappingDao instance for testing."""
    return PortMappingDao()


@pytest.mark.asyncio
async def test_get_successful_ports_returns_dict_with_external_port_keys(
    port_mapping_dao: PortMappingDao, test_db_session, mock_async_session_maker
):
    """Test that get_successful_ports returns dict with external_port as keys."""
    executor_id = uuid4()

    successful_ports = create_port_mappings_batch(
        count=3, executor_id=executor_id, is_successful=True, base_port=9000
    )
    failed_ports = create_port_mappings_batch(
        count=2, executor_id=executor_id, is_successful=False, base_port=9100
    )

    test_db_session.add_all(successful_ports + failed_ports)
    await test_db_session.commit()

    result = await port_mapping_dao.get_successful_ports(executor_id)

    # Should return only successful ports as dict
    assert len(result) == 3
    assert set(result.keys()) == {9000, 9001, 9002}

    # All values should be PortMapping objects
    for external_port, port_mapping in result.items():
        assert isinstance(port_mapping, PortMapping)
        assert port_mapping.external_port == external_port
        assert port_mapping.is_successful is True
        assert port_mapping.executor_id == executor_id


@pytest.mark.asyncio
async def test_upsert_port_results_creates_new_ports(
    port_mapping_dao: PortMappingDao, test_db_session, mock_async_session_maker
):
    """Test that upsert_port_results creates new ports."""
    executor_id = uuid4()

    port_results = create_port_mappings_batch(count=3, executor_id=executor_id, base_port=20000)
    await port_mapping_dao.upsert_port_results(port_results)

    # Verify ports are saved in database using get_successful_ports
    ports_dict = await port_mapping_dao.get_successful_ports(executor_id)
    assert len(ports_dict) == 3
    assert set(ports_dict.keys()) == {20000, 20001, 20002}


@pytest.mark.asyncio
async def test_upsert_port_results_updates_existing_ports(
    port_mapping_dao: PortMappingDao, test_db_session, mock_async_session_maker
):
    """Test that upsert_port_results updates existing ports with same external_port."""
    executor_id = uuid4()

    # Create initial ports with older timestamp
    older_time = datetime(2025, 1, 1, 10, 0, 0)
    initial_ports = create_port_mappings_batch(count=2, executor_id=executor_id, base_port=20000)
    for port in initial_ports:
        port.verification_time = older_time
        port.is_successful = False  # Different success status

    await port_mapping_dao.upsert_port_results(initial_ports)

    # Update same ports with newer timestamp and different status
    newer_time = datetime(2025, 1, 1, 11, 0, 0)
    updated_ports = create_port_mappings_batch(count=2, executor_id=executor_id, base_port=20000)
    for port in updated_ports:
        port.verification_time = newer_time
        port.is_successful = True

    await port_mapping_dao.upsert_port_results(updated_ports)

    # Verify ports were updated
    ports_dict = await port_mapping_dao.get_successful_ports(executor_id)
    assert len(ports_dict) == 2  # Should still be 2 ports

    for port in ports_dict.values():
        assert port.verification_time == newer_time
        assert port.is_successful is True


@pytest.mark.asyncio
async def test_upsert_port_results_different_executors_isolated(
    port_mapping_dao: PortMappingDao, test_db_session, mock_async_session_maker
):
    """Test that upsert_port_results for different executors don't interfere."""
    executor_id1 = uuid4()
    executor_id2 = uuid4()

    # Save ports for first executor
    ports1 = create_port_mappings_batch(count=2, executor_id=executor_id1, base_port=20000)
    await port_mapping_dao.upsert_port_results(ports1)

    # Save ports for second executor (same port numbers)
    ports2 = create_port_mappings_batch(count=2, executor_id=executor_id2, base_port=20000)
    await port_mapping_dao.upsert_port_results(ports2)

    # Verify each executor has their own ports
    dict1 = await port_mapping_dao.get_successful_ports(executor_id1)
    dict2 = await port_mapping_dao.get_successful_ports(executor_id2)

    assert len(dict1) == 2
    assert len(dict2) == 2

    # All ports for executor1 should have executor_id1
    for port in dict1.values():
        assert port.executor_id == executor_id1

    # All ports for executor2 should have executor_id2
    for port in dict2.values():
        assert port.executor_id == executor_id2


# Note: get_successful_ports is tested indirectly through the DockerService tests
# which provide comprehensive coverage of the functionality in realistic scenarios.


@pytest.mark.asyncio
async def test_get_ports_for_pod_returns_reserved_ports(
    port_mapping_dao: PortMappingDao, test_db_session, mock_async_session_maker
):
    """Test that get_ports_for_pod returns only ports reserved for specific pod."""
    executor_id = uuid4()
    pod_id_1 = uuid4()
    pod_id_2 = uuid4()

    # Create ports for pod_1
    pod1_ports = create_port_mappings_batch(
        count=3, executor_id=executor_id, is_successful=True, base_port=9000
    )
    for port in pod1_ports:
        port.rented_for_pod_id = pod_id_1

    # Create ports for pod_2
    pod2_ports = create_port_mappings_batch(
        count=2, executor_id=executor_id, is_successful=True, base_port=9100
    )
    for port in pod2_ports:
        port.rented_for_pod_id = pod_id_2

    # Create unreserved ports
    free_ports = create_port_mappings_batch(
        count=2, executor_id=executor_id, is_successful=True, base_port=9200
    )

    test_db_session.add_all(pod1_ports + pod2_ports + free_ports)
    await test_db_session.commit()

    # Act - get ports for pod_1
    result = await port_mapping_dao.get_ports_for_pod(pod_id_1)

    # Assert
    assert len(result) == 3
    assert set(result.keys()) == {9000, 9001, 9002}

    for external_port, port_mapping in result.items():
        assert port_mapping.rented_for_pod_id == pod_id_1
        assert port_mapping.external_port == external_port


@pytest.mark.asyncio
async def test_reserve_ports_for_pod_sets_pod_id(
    port_mapping_dao: PortMappingDao, test_db_session, mock_async_session_maker
):
    """Test that reserve_ports_for_pod reserves new ports and releases old ones."""
    executor_id = uuid4()
    pod_id = uuid4()

    # Create ports - some will be reserved for pod, some will stay free
    all_ports = create_port_mappings_batch(
        count=6, executor_id=executor_id, is_successful=True, base_port=9000
    )

    # Initially reserve ports 9000, 9001, 9002 for this pod
    for port in all_ports[:3]:
        port.rented_for_pod_id = pod_id

    test_db_session.add_all(all_ports)
    await test_db_session.commit()

    # Act - Reserve new set of ports: 9001, 9002, 9003, 9004
    # This should:
    # - Keep 9001, 9002 (already reserved for this pod)
    # - Release 9000 (was reserved but not in new list)
    # - Reserve 9003, 9004 (new ports)
    new_external_ports = [9001, 9002, 9003, 9004]
    # Create mappings as (docker_port, internal_port, external_port)
    new_mappings = [(port, port, port) for port in new_external_ports]
    await port_mapping_dao.reserve_ports_for_pod(executor_id, new_mappings, pod_id)

    # Assert - verify correct ports are reserved
    result = await port_mapping_dao.get_ports_for_pod(pod_id)
    assert len(result) == 4

    # Check that correct ports are reserved
    reserved_external_ports = {port.external_port for port in result.values()}
    assert reserved_external_ports == {9001, 9002, 9003, 9004}

    # Verify all reserved ports belong to this pod
    for port in result.values():
        assert port.rented_for_pod_id == pod_id

    # Verify port 9000 was released (get all ports for executor and check)
    all_executor_ports = await port_mapping_dao.get_successful_ports(executor_id)
    port_9000 = all_executor_ports[9000]
    assert port_9000.rented_for_pod_id is None


@pytest.mark.asyncio
async def test_release_ports_for_pod_clears_pod_id(
    port_mapping_dao: PortMappingDao, test_db_session, mock_async_session_maker
):
    """Test that release_ports_for_pod clears rented_for_pod_id."""
    executor_id = uuid4()
    pod_id = uuid4()

    # Create reserved ports
    ports = create_port_mappings_batch(
        count=4, executor_id=executor_id, is_successful=True, base_port=9000
    )
    for port in ports:
        port.rented_for_pod_id = pod_id

    test_db_session.add_all(ports)
    await test_db_session.commit()

    # Verify ports are reserved
    before_release = await port_mapping_dao.get_ports_for_pod(pod_id)
    assert len(before_release) == 4

    # Act
    released_count = await port_mapping_dao.release_ports_for_pod(pod_id)

    # Assert
    assert released_count == 4

    # Verify ports are no longer reserved
    after_release = await port_mapping_dao.get_ports_for_pod(pod_id)
    assert len(after_release) == 0


@pytest.mark.asyncio
async def test_get_available_ports_excluding_rented_filters_correctly(
    port_mapping_dao: PortMappingDao, test_db_session, mock_async_session_maker
):
    """Test that get_available_ports_excluding_rented returns only free successful ports."""
    executor_id = uuid4()

    # Create successful unreserved ports (should be returned)
    free_successful = create_port_mappings_batch(
        count=3, executor_id=executor_id, is_successful=True, base_port=9000
    )

    # Create successful reserved ports (should NOT be returned)
    rented_successful = create_port_mappings_batch(
        count=2, executor_id=executor_id, is_successful=True, base_port=9100
    )
    for port in rented_successful:
        port.rented_for_pod_id = uuid4()

    # Create failed unreserved ports (should NOT be returned)
    free_failed = create_port_mappings_batch(
        count=2, executor_id=executor_id, is_successful=False, base_port=9200
    )

    test_db_session.add_all(free_successful + rented_successful + free_failed)
    await test_db_session.commit()

    # Act
    result = await port_mapping_dao.get_available_ports_excluding_rented(executor_id)

    # Assert - should return only free successful ports
    assert len(result) == 3
    assert set(result.keys()) == {9000, 9001, 9002}

    for port in result.values():
        assert port.is_successful is True
        assert port.rented_for_pod_id is None


@pytest.mark.asyncio
async def test_get_available_ports_excluding_rented_respects_limit(
    port_mapping_dao: PortMappingDao, test_db_session, mock_async_session_maker
):
    """Test that get_available_ports_excluding_rented respects limit parameter."""
    executor_id = uuid4()

    # Create 10 free successful ports
    ports = create_port_mappings_batch(
        count=10, executor_id=executor_id, is_successful=True, base_port=9000
    )
    test_db_session.add_all(ports)
    await test_db_session.commit()

    # Act - request only 5 ports
    result = await port_mapping_dao.get_available_ports_excluding_rented(executor_id, limit=5)

    # Assert
    assert len(result) == 5
