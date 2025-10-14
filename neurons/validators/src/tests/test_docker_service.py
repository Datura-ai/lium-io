from unittest.mock import AsyncMock, Mock
from uuid import uuid4, UUID
from datetime import datetime

import pytest
import pytest_asyncio

from services.docker_service import DockerService
from models.port_mapping import PortMapping
from .factories import create_port_mapping, create_port_mappings_batch


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


@pytest.mark.parametrize(
    "available_ports,expected_mappings,initial_port_count",
    [
        # Exact match with PREFERRED_POD_PORTS
        (
            [22, 20000, 20001],
            [(22, 22, 22), (20000, 20000, 20000), (20001, 20001, 20001)],
            None,
        ),
        # Simple available ports - SSH missing, gets max port
        (
            [20000, 20001, 20002],
            [(22, 20002, 20002), (20000, 20000, 20000), (20001, 20001, 20001)],
            None,
        ),
        # Available ports don't match PREFERRED_POD_PORTS - flexible mode assigns SSH to max port
        (
            [9000, 9001, 9002],
            [(22, 9002, 9002), (9000, 9000, 9000), (9001, 9001, 9001)],
            None,
        ),
        # many ports available, only 1 initial_port_count
        (
            [r for r in range(20000, 20100)],
            [(22, 20099, 20099), (20000, 20000, 20000)],
            1,
        ),
        # many ports available, 50 initial_port_count
        (
            [r for r in range(20000, 20100)],
            [(22, 20099, 20099)] + [(port, port, port) for port in range(20000, 20050)],
            50,
        ),
    ],
)
@pytest.mark.asyncio
async def test_flexible_mode_port_mappings(
    docker_service, test_executor_id, test_miner_hotkey, available_ports, expected_mappings, initial_port_count, monkeypatch
):
    """Test FLEXIBLE mode with various available port scenarios.

    In flexible mode (internal_ports=None):
    - If exact matches exist, use them
    - If no exact matches, docker_port = external_port from available set
    - SSH port (22) gets special handling: max port if not available
    """
    # Mock PREFERRED_POD_PORTS to a shorter list for easier testing
    monkeypatch.setattr("services.docker_service.PREFERRED_POD_PORTS", [22, 20000, 20001])

    # Mock database response
    mock_ports = create_mock_port_dict(available_ports, test_miner_hotkey, UUID(test_executor_id))
    docker_service.port_mapping_dao.get_successful_ports = AsyncMock(return_value=mock_ports)

    # Act - internal_ports=None triggers flexible mode
    result = await docker_service.generate_portMappings(test_miner_hotkey, test_executor_id, None, initial_port_count)

    # Assert
    assert len(result) == len(expected_mappings)
    assert set(result) == set(expected_mappings)
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


@pytest.mark.parametrize("initial_port_count,expected_length,expected_first_port,should_have_extra_ports", [
    # No initial count - returns all PREFERRED_POD_PORTS
    (None, 11, 22, False),
    # Less than PREFERRED_POD_PORTS length - returns limited list
    (0, 1, 22, False),
    (2, 3, 22, False),  # +1 for SSH port = 3 total
    (5, 6, 22, False),  # +1 for SSH port = 6 total
    # More than PREFERRED_POD_PORTS length - returns PREFERRED_POD_PORTS + extra
    (11, 12, 22, True),  # +1 for SSH port = 12 total, 1 extra port needed
    (15, 16, 22, True),  # +1 for SSH port = 16 total, 5 extra ports needed
])
def test_get_preferred_ports(
    docker_service,
    initial_port_count,
    expected_length,
    expected_first_port,
    should_have_extra_ports,
    monkeypatch
):
    """Test get_prefered_ports method with various initial_port_count scenarios.

    The method adds 1 to initial_port_count for SSH port and returns:
    - All PREFERRED_POD_PORTS if initial_port_count is None/0
    - Limited PREFERRED_POD_PORTS if initial_port_count < len(PREFERRED_POD_PORTS)
    - PREFERRED_POD_PORTS + extra ports if initial_port_count > len(PREFERRED_POD_PORTS)
    """
    # Arrange - Mock PREFERRED_POD_PORTS to a known list
    mock_preferred_ports = [22, 20000, 20001, 20002, 20003, 20004, 20005, 20006, 20007, 20008, 20009]
    monkeypatch.setattr("services.docker_service.PREFERRED_POD_PORTS", mock_preferred_ports)

    # Act
    result = docker_service._get_preferred_ports(initial_port_count)

    # Assert
    assert len(result) == expected_length
    assert result[0] == expected_first_port

    # Verify all ports are from PREFERRED_POD_PORTS or are extra sequential ports
    if should_have_extra_ports:
        # Check that first 11 ports are from PREFERRED_POD_PORTS
        assert result[:11] == mock_preferred_ports
        # Check that extra ports are sequential after max preferred port
        max_preferred = max(mock_preferred_ports)
        extra_ports = result[11:]
        for i, port in enumerate(extra_ports):
            assert port == max_preferred + i
    else:
        # All ports should be from PREFERRED_POD_PORTS
        for port in result:
            assert port in mock_preferred_ports
