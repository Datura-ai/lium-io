from unittest.mock import AsyncMock, Mock
from uuid import uuid4, UUID
from datetime import datetime

import pytest
import pytest_asyncio

from services.docker_service import DockerService
from models.port_mapping import PortMapping


@pytest.fixture
def mock_dependencies():
    """Mock all DockerService dependencies."""
    ssh_service = Mock()
    redis_service = Mock()
    port_mapping_dao = Mock()
    return ssh_service, redis_service, port_mapping_dao


@pytest_asyncio.fixture
async def docker_service(mock_dependencies):
    """Create DockerService instance with mocked dependencies."""
    ssh_service, redis_service, port_mapping_dao = mock_dependencies
    service = DockerService(
        ssh_service=ssh_service,
        redis_service=redis_service,
    )
    service.port_mapping_dao = port_mapping_dao
    return service


@pytest.mark.asyncio
async def test_generate_portMappings_exact_matches(docker_service):
    """Test port mappings with exact docker_port == external_port matches."""
    executor_id = str(uuid4())
    miner_hotkey = "test_miner"
    docker_ports = [22, 20000, 20001]

    # Mock database response with exact matches for all requested ports
    mock_ports = {
        22: PortMapping(
            miner_hotkey=miner_hotkey,
            executor_id=UUID(executor_id),
            internal_port=22,
            external_port=22,
            is_successful=True,
            verification_time=datetime.utcnow()
        ),
        20000: PortMapping(
            miner_hotkey=miner_hotkey,
            executor_id=UUID(executor_id),
            internal_port=20000,
            external_port=20000,
            is_successful=True,
            verification_time=datetime.utcnow()
        ),
        20001: PortMapping(
            miner_hotkey=miner_hotkey,
            executor_id=UUID(executor_id),
            internal_port=20001,
            external_port=20001,
            is_successful=True,
            verification_time=datetime.utcnow()
        )
    }

    docker_service.port_mapping_dao.get_successful_ports = AsyncMock(return_value=mock_ports)

    # Test
    result = await docker_service.generate_portMappings(miner_hotkey, executor_id, docker_ports)

    # Verify
    assert len(result) == 3
    assert (22, 22, 22) in result
    assert (20000, 20000, 20000) in result
    assert (20001, 20001, 20001) in result
    docker_service.port_mapping_dao.get_successful_ports.assert_called_once_with(UUID(executor_id))


@pytest.mark.asyncio
async def test_generate_portMappings_mixed_scenario(docker_service):
    """Test port mappings with both exact matches and random selection."""
    executor_id = str(uuid4())
    miner_hotkey = "test_miner"
    docker_ports = [22, 20000, 20001]

    # Mock database response: exact match for 22, random for others
    mock_ports = {
        22: PortMapping(
            miner_hotkey=miner_hotkey,
            executor_id=UUID(executor_id),
            internal_port=22,
            external_port=22,
            is_successful=True,
            verification_time=datetime.utcnow()
        ),
        8080: PortMapping(
            miner_hotkey=miner_hotkey,
            executor_id=UUID(executor_id),
            internal_port=8080,
            external_port=8080,
            is_successful=True,
            verification_time=datetime.utcnow()
        ),
        9090: PortMapping(
            miner_hotkey=miner_hotkey,
            executor_id=UUID(executor_id),
            internal_port=9090,
            external_port=9090,
            is_successful=True,
            verification_time=datetime.utcnow()
        )
    }

    docker_service.port_mapping_dao.get_successful_ports = AsyncMock(return_value=mock_ports)

    # Test
    result = await docker_service.generate_portMappings(miner_hotkey, executor_id, docker_ports)

    # Verify
    assert len(result) == 3
    assert (22, 22, 22) in result  # Exact match
    # Other docker ports should get random available ports
    other_mappings = [m for m in result if m[0] != 22]
    assert len(other_mappings) == 2
    external_ports_used = {m[2] for m in other_mappings}
    assert external_ports_used.issubset({8080, 9090})  # Should be from available set


@pytest.mark.asyncio
async def test_generate_portMappings_empty_database_fallback(docker_service):
    """Test port mappings when database returns no successful ports, fallback to Redis."""
    executor_id = str(uuid4())
    miner_hotkey = "test_miner"
    docker_ports = [22, 20000]

    # Mock empty database response
    docker_service.port_mapping_dao.get_successful_ports = AsyncMock(return_value={})

    # Mock Redis fallback to return empty list
    docker_service.generate_port_mapping_from_redis = AsyncMock(return_value=[])

    # Test
    result = await docker_service.generate_portMappings(miner_hotkey, executor_id, docker_ports)

    # Verify - should fallback to Redis when DB returns empty
    assert result == []
    docker_service.port_mapping_dao.get_successful_ports.assert_called_once_with(UUID(executor_id))
    docker_service.generate_port_mapping_from_redis.assert_called_once()


@pytest.mark.asyncio
async def test_generate_portMappings_insufficient_ports(docker_service):
    """Test behavior when database has insufficient ports (returns only what's available)."""
    executor_id = str(uuid4())
    miner_hotkey = "test_miner"
    docker_ports = [22, 20000, 20001]  # Request 3 ports

    # Mock database response with only 1 available port
    mock_ports = {
        22: PortMapping(
            miner_hotkey=miner_hotkey,
            executor_id=UUID(executor_id),
            internal_port=22,
            external_port=22,
            is_successful=True,
            verification_time=datetime.utcnow()
        )
    }

    docker_service.port_mapping_dao.get_successful_ports = AsyncMock(return_value=mock_ports)

    # Test
    result = await docker_service.generate_portMappings(miner_hotkey, executor_id, docker_ports)

    # Verify - should return only one mapping (for port 22), skip others
    assert len(result) == 1
    assert (22, 22, 22) in result
    docker_service.port_mapping_dao.get_successful_ports.assert_called_once_with(UUID(executor_id))


@pytest.mark.asyncio
async def test_generate_portMappings_database_error_fallback(docker_service):
    """Test fallback to Redis when database throws exception."""
    executor_id = str(uuid4())
    miner_hotkey = "test_miner"
    docker_ports = [22, 20000]

    # Mock database error
    docker_service.port_mapping_dao.get_successful_ports = AsyncMock(side_effect=Exception("Database error"))

    # Mock Redis fallback
    expected_redis_result = [(22, 22, 22), (20000, 20000, 20000)]
    docker_service.generate_port_mapping_from_redis = AsyncMock(return_value=expected_redis_result)

    # Test
    result = await docker_service.generate_portMappings(miner_hotkey, executor_id, docker_ports)

    # Verify fallback was called
    assert result == expected_redis_result
    docker_service.generate_port_mapping_from_redis.assert_called_once_with(
        executor_id, docker_ports, miner_hotkey
    )


@pytest.mark.asyncio
async def test_generate_portMappings_default_preferred_ports(docker_service):
    """Test that method uses PREFERED_POD_PORTS when internal_ports is None."""
    executor_id = str(uuid4())
    miner_hotkey = "test_miner"

    # Mock database response with all preferred ports available
    from services.const import PREFERRED_POD_PORTS

    mock_ports = {}
    for port in PREFERRED_POD_PORTS:
        mock_ports[port] = PortMapping(
            miner_hotkey=miner_hotkey,
            executor_id=UUID(executor_id),
            internal_port=port,
            external_port=port,
            is_successful=True,
            verification_time=datetime.utcnow()
        )

    docker_service.port_mapping_dao.get_successful_ports = AsyncMock(return_value=mock_ports)

    # Test with internal_ports=None (should use PREFERED_POD_PORTS)
    result = await docker_service.generate_portMappings(miner_hotkey, executor_id, None)

    # Verify that it used PREFERED_POD_PORTS
    assert len(result) == len(PREFERRED_POD_PORTS)
    for port in PREFERRED_POD_PORTS:
        assert (port, port, port) in result
    docker_service.port_mapping_dao.get_successful_ports.assert_called_once_with(UUID(executor_id))