from unittest.mock import AsyncMock, Mock
from uuid import uuid4, UUID
from datetime import datetime

import pytest
import pytest_asyncio

from services.docker_service import DockerService
from models.port_mapping import PortMapping
from tests.factories import create_port_mapping, create_port_mappings_batch


def create_mock_port_dict(
    ports: list[int],
    miner_hotkey: str,
    executor_id: UUID
) -> dict[int, PortMapping]:
    """Helper to create mock port dictionary from list of ports."""
    return {
        port: create_port_mapping(
            miner_hotkey=miner_hotkey,
            executor_id=executor_id,
            internal_port=port,
            external_port=port,
            is_successful=True
        )
        for port in ports
    }


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
        port_mapping_dao=port_mapping_dao
    )
    return service


@pytest.fixture
def test_executor_id():
    """Fixture for test executor ID."""
    return str(uuid4())


@pytest.fixture
def test_miner_hotkey():
    """Fixture for test miner hotkey."""
    return "test_miner"


@pytest.mark.asyncio
async def test_generate_portMappings_exact_matches(docker_service, test_executor_id, test_miner_hotkey):
    """Test port mappings with exact docker_port == external_port matches."""
    docker_ports = [22, 20000, 20001]

    # Mock database response with exact matches for all requested ports
    mock_ports = create_mock_port_dict(docker_ports, test_miner_hotkey, UUID(test_executor_id))
    docker_service.port_mapping_dao.get_successful_ports = AsyncMock(return_value=mock_ports)

    # Act
    result = await docker_service.generate_portMappings(test_miner_hotkey, test_executor_id, docker_ports)

    # Assert
    # Expect exact matches for all ports
    assert len(result) == 3
    assert (22, 22, 22) in result
    assert (20000, 20000, 20000) in result
    assert (20001, 20001, 20001) in result
    docker_service.port_mapping_dao.get_successful_ports.assert_called_once_with(UUID(test_executor_id))


@pytest.mark.asyncio
async def test_generate_portMappings_mixed_scenario(docker_service, test_executor_id, test_miner_hotkey):
    """Test port mappings with both exact matches and random selection."""
    docker_ports = [22, 20000, 20001]

    # Mock database response: exact match for 22, random for others
    mock_ports = create_mock_port_dict([22, 8080, 9090], test_miner_hotkey, UUID(test_executor_id))
    docker_service.port_mapping_dao.get_successful_ports = AsyncMock(return_value=mock_ports)

    # Act
    result = await docker_service.generate_portMappings(test_miner_hotkey, test_executor_id, docker_ports)

    # Assert
    # Expect exact match for 22, random selection for others
    assert len(result) == 3
    assert (22, 22, 22) in result  # Exact match

    # Other docker ports should get random available ports from {8080, 9090}
    other_mappings = [m for m in result if m[0] != 22]
    assert len(other_mappings) == 2
    external_ports_used = {m[2] for m in other_mappings}
    assert external_ports_used.issubset({8080, 9090})


@pytest.mark.parametrize("scenario,db_response,docker_ports,expected_result", [
    (
        "empty_database",
        {},
        [22, 20000],
        []
    ),
    (
        "insufficient_ports",
        "create_mock",  # Special marker to create 2 ports
        [22, 20000, 20001],
        [(22, 22, 22), (20000, 20000, 20000), (20001, 20001, 20001)]
    ),
    (
        "database_error",
        Exception("Database error"),
        [22, 20000],
        [(22, 22, 22), (20000, 20000, 20000)]
    )
])
@pytest.mark.asyncio
async def test_generate_portMappings_fallback_to_redis(
    docker_service, test_executor_id, test_miner_hotkey, scenario, db_response, docker_ports, expected_result
):
    """Test fallback to Redis on various database failures."""
    # Arrange
    if db_response == "create_mock":
        # Create insufficient ports (2 ports when MIN_PORTS=3)
        db_response = create_mock_port_dict([22, 20000], test_miner_hotkey, UUID(test_executor_id))

    if isinstance(db_response, Exception):
        docker_service.port_mapping_dao.get_successful_ports = AsyncMock(side_effect=db_response)
    else:
        docker_service.port_mapping_dao.get_successful_ports = AsyncMock(return_value=db_response)

    docker_service.generate_port_mapping_from_redis = AsyncMock(return_value=expected_result)

    # Act
    result = await docker_service.generate_portMappings(test_miner_hotkey, test_executor_id, docker_ports)

    # Assert
    # Expect fallback to Redis in all scenarios
    assert result == expected_result
    docker_service.generate_port_mapping_from_redis.assert_called_once_with(
        test_executor_id, docker_ports, test_miner_hotkey
    )


@pytest.mark.asyncio
async def test_generate_portMappings_default_preferred_ports(docker_service, test_executor_id, test_miner_hotkey):
    """Test that method uses PREFERED_POD_PORTS when internal_ports is None."""
    from services.const import PREFERRED_POD_PORTS

    # Mock database response with all preferred ports available
    mock_ports = create_mock_port_dict(PREFERRED_POD_PORTS, test_miner_hotkey, UUID(test_executor_id))
    docker_service.port_mapping_dao.get_successful_ports = AsyncMock(return_value=mock_ports)

    # Act - internal_ports=None should use PREFERRED_POD_PORTS
    result = await docker_service.generate_portMappings(test_miner_hotkey, test_executor_id, None)

    # Assert
    # Expect exact matches for all PREFERRED_POD_PORTS
    assert len(result) == len(PREFERRED_POD_PORTS)
    for port in PREFERRED_POD_PORTS:
        assert (port, port, port) in result
    docker_service.port_mapping_dao.get_successful_ports.assert_called_once_with(UUID(test_executor_id))


@pytest.mark.asyncio
async def test_no_exact_match_custom_ports_uses_random_selection(docker_service, test_executor_id, test_miner_hotkey):
    """Test random selection when no exact matches found with custom internal_ports."""
    custom_internal_ports = [8080, 8081, 8082]

    # Available ports don't match requested ports
    mock_ports = create_mock_port_dict([9000, 9001, 9002], test_miner_hotkey, UUID(test_executor_id))
    docker_service.port_mapping_dao.get_successful_ports = AsyncMock(return_value=mock_ports)

    # Act
    result = await docker_service.generate_portMappings(test_miner_hotkey, test_executor_id, custom_internal_ports)

    # Assert
    # Expect random selection: docker ports from custom list, external ports from available set
    assert len(result) == 3
    docker_ports_used = {m[0] for m in result}
    assert docker_ports_used == {8080, 8081, 8082}

    external_ports_used = {m[2] for m in result}
    assert external_ports_used == {9000, 9001, 9002}

    # Verify mapping structure
    for docker_port, internal_port, external_port in result:
        assert docker_port in custom_internal_ports
        assert external_port in {9000, 9001, 9002}
        assert internal_port == external_port

    docker_service.port_mapping_dao.get_successful_ports.assert_called_once_with(UUID(test_executor_id))


@pytest.mark.asyncio
async def test_no_exact_match_preferred_ports_uses_min_selection(docker_service, test_executor_id, test_miner_hotkey):
    """Test min selection when no exact matches found with PREFERRED_POD_PORTS."""
    from services.const import PREFERRED_POD_PORTS

    # Available ports don't overlap with PREFERRED_POD_PORTS - create sequential ports starting from 9000
    available_ports = list(range(9000, 9000 + len(PREFERRED_POD_PORTS)))
    mock_ports = create_mock_port_dict(available_ports, test_miner_hotkey, UUID(test_executor_id))
    docker_service.port_mapping_dao.get_successful_ports = AsyncMock(return_value=mock_ports)

    # Act - internal_ports=None means use PREFERRED_POD_PORTS
    result = await docker_service.generate_portMappings(test_miner_hotkey, test_executor_id, None)

    # Assert
    # Expect min selection: external ports selected in ascending order
    assert len(result) == len(PREFERRED_POD_PORTS)

    external_ports_used = [m[2] for m in result]
    for i in range(len(PREFERRED_POD_PORTS)):
        assert external_ports_used[i] == 9000 + i

    docker_service.port_mapping_dao.get_successful_ports.assert_called_once_with(UUID(test_executor_id))


@pytest.mark.asyncio
async def test_partial_exact_match_preferred_ports_uses_min_for_missing(docker_service, test_executor_id, test_miner_hotkey):
    """Test combination of exact match and min selection with PREFERRED_POD_PORTS."""
    from services.const import PREFERRED_POD_PORTS

    # Available ports: exact match for port 22, rest don't overlap with PREFERRED_POD_PORTS
    available_ports = [22] + list(range(9000, 9000 + len(PREFERRED_POD_PORTS)))
    mock_ports = create_mock_port_dict(available_ports, test_miner_hotkey, UUID(test_executor_id))
    docker_service.port_mapping_dao.get_successful_ports = AsyncMock(return_value=mock_ports)

    # Act - internal_ports=None means use PREFERRED_POD_PORTS
    result = await docker_service.generate_portMappings(test_miner_hotkey, test_executor_id, None)

    # Assert
    # Expect exact match for 22, min selection for remaining
    assert len(result) == len(PREFERRED_POD_PORTS)
    assert (22, 22, 22) in result

    # Remaining mappings should use min selection
    remaining_mappings = [m for m in result if m[0] != 22]
    external_ports_used = [m[2] for m in remaining_mappings]
    for i in range(len(remaining_mappings)):
        assert external_ports_used[i] == 9000 + i

    docker_service.port_mapping_dao.get_successful_ports.assert_called_once_with(
        UUID(test_executor_id)
    )


@pytest.mark.asyncio
async def test_flexible_mode_deviates_from_preferred_when_unavailable(
    docker_service, test_executor_id, test_miner_hotkey
):
    """Test that FLEXIBLE mode allows deviation from PREFERRED_POD_PORTS when preferred ports unavailable.

    In flexible mode (internal_ports=None), when no exact matches exist:
    - First element (docker_port) = external_port (NOT from PREFERRED_POD_PORTS)
    - This allows using any available port for both docker and external
    """
    from services.const import PREFERRED_POD_PORTS

    # Available ports completely different from PREFERRED_POD_PORTS
    available_ports = list(range(9000, 9000 + len(PREFERRED_POD_PORTS)))
    mock_ports = create_mock_port_dict(available_ports, test_miner_hotkey, UUID(test_executor_id))
    docker_service.port_mapping_dao.get_successful_ports = AsyncMock(return_value=mock_ports)

    # Act - internal_ports=None means use PREFERRED_POD_PORTS in flexible mode
    result = await docker_service.generate_portMappings(test_miner_hotkey, test_executor_id, None)

    # Assert
    assert len(result) == len(PREFERRED_POD_PORTS)

    # In flexible mode with no exact matches, docker_port should equal external_port
    # (both come from available_ports, NOT from PREFERRED_POD_PORTS)
    for docker_port, internal_port, external_port in result:
        # KEY ASSERTION: docker_port equals external_port (both from available set)
        assert docker_port == external_port
        # Both should be from available_ports range, not PREFERRED_POD_PORTS
        assert docker_port in available_ports
        assert docker_port not in PREFERRED_POD_PORTS

    # Verify ports are selected in ascending order (min strategy)
    docker_ports_used = [m[0] for m in result]
    assert docker_ports_used == sorted(available_ports)[: len(PREFERRED_POD_PORTS)]

    docker_service.port_mapping_dao.get_successful_ports.assert_called_once_with(
        UUID(test_executor_id)
    )
