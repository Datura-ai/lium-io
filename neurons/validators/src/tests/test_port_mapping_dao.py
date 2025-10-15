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
