from unittest.mock import AsyncMock, Mock
from uuid import uuid4, UUID
from datetime import datetime

import pytest
import pytest_asyncio

from services.docker_service import DockerService
from models.port_mapping import PortMapping
from .factories import create_port_mapping, create_port_mappings_batch
from payload_models.payloads import PayloadPortMapping
from datura.requests.miner_requests import ExecutorSSHInfo
from payload_models.payloads import ContainerStartRequest


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
    attestation_service = Mock()

    # Mock the async context manager for Redis lock
    lock_mock = AsyncMock()
    lock_mock.__aenter__ = AsyncMock(return_value=lock_mock)
    lock_mock.__aexit__ = AsyncMock(return_value=None)
    redis_service.acquire_executor_lock = Mock(return_value=lock_mock)

    return ssh_service, redis_service, port_mapping_dao, attestation_service


@pytest_asyncio.fixture
async def docker_service(mock_dependencies):
    """Create DockerService instance with mocked dependencies."""
    ssh_service, redis_service, port_mapping_dao, attestation_service = mock_dependencies
    service = DockerService(
        ssh_service=ssh_service,
        redis_service=redis_service,
        port_mapping_dao=port_mapping_dao,
        attestation_service=attestation_service
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
    docker_service.port_mapping_dao.get_available_ports_excluding_rented = AsyncMock(return_value=mock_ports)
    docker_service.port_mapping_dao.get_ports_for_pod = AsyncMock(return_value={})
    docker_service.port_mapping_dao.reserve_ports_for_pod = AsyncMock()

    # Act
    result = await docker_service.generate_portMappings(test_miner_hotkey, test_executor_id, UUID(test_executor_id), docker_ports)
    result = result[0]

    # Assert
    # Expect exact matches for all ports
    assert len(result) == 3
    assert (22, 22, 22) in result
    assert (20000, 20000, 20000) in result
    assert (20001, 20001, 20001) in result
    docker_service.port_mapping_dao.get_available_ports_excluding_rented.assert_called_once_with(UUID(test_executor_id))
    docker_service.port_mapping_dao.reserve_ports_for_pod.assert_called_once()


@pytest.mark.asyncio
async def test_generate_portMappings_mixed_scenario(docker_service, test_executor_id, test_miner_hotkey):
    """Test port mappings with both exact matches and random selection."""
    docker_ports = [22, 20000, 20001]

    # Mock database response: exact match for 22, random for others
    mock_ports = create_mock_port_dict([22, 8080, 9090], test_miner_hotkey, UUID(test_executor_id))
    docker_service.port_mapping_dao.get_available_ports_excluding_rented = AsyncMock(return_value=mock_ports)
    docker_service.port_mapping_dao.get_ports_for_pod = AsyncMock(return_value={})
    docker_service.port_mapping_dao.reserve_ports_for_pod = AsyncMock()

    # Act
    result = await docker_service.generate_portMappings(test_miner_hotkey, test_executor_id, UUID(test_executor_id), docker_ports)
    result = result[0]

    # Assert
    # Expect exact match for 22, random selection for others
    assert len(result) == 3
    assert (22, 22, 22) in result  # Exact match

    # Other docker ports should get random available ports from {8080, 9090}
    other_mappings = [m for m in result if m[0] != 22]
    assert len(other_mappings) == 2
    external_ports_used = {m[2] for m in other_mappings}
    assert external_ports_used.issubset({8080, 9090})


@pytest.mark.parametrize(
    "available_ports,expected_mappings,initial_port_count,enable_jupyter",
    [
        # Exact match with PREFERRED_POD_PORTS
        (
            [22, 20000, 20001],
            [(22, 22, 22), (20000, 20000, 20000), (20001, 20001, 20001)],
            None,
            False,
        ),
        # Simple available ports - SSH missing, gets max port
        (
            [20000, 20001, 20002],
            [(22, 20002, 20002), (20000, 20000, 20000), (20001, 20001, 20001)],
            None,
            False,
        ),
        # Available ports don't match PREFERRED_POD_PORTS - flexible mode assigns SSH to max port
        (
            [9000, 9001, 9002],
            [(22, 9002, 9002), (9000, 9000, 9000), (9001, 9001, 9001)],
            None,
            False,
        ),
        # many ports available, only 1 initial_port_count
        (
            [r for r in range(20000, 20100)],
            [(22, 20099, 20099), (20000, 20000, 20000)],
            1,
            False,
        ),
        # many ports available, 50 initial_port_count
        (
            [r for r in range(20000, 20100)],
            [(22, 20099, 20099)] + [(port, port, port) for port in range(20000, 20050)],
            50,
            False,
        ),
        # case - we have a small amount of ports available, but big initial_port_count
        (
            [r for r in range(20000, 20005)],
            [(22, 20004, 20004)]  + [(port, port, port) for port in range(20000, 20004)],
            50,
            False,
        ),
        # enable_jupyter=True, 8888 available - exact match
        (
            [22, 8888, 20000, 20001],
            [(22, 22, 22), (8888, 8888, 8888), (20000, 20000, 20000), (20001, 20001, 20001)],
            None,
            True,
        ),
        # enable_jupyter=True, 8888 not available - SSH gets max, Jupyter gets next available
        (
            [9000, 9001, 9002, 9003],
            [(22, 9003, 9003), (8888, 9002, 9002), (9000, 9000, 9000), (9001, 9001, 9001)],
            None,
            True,
        ),
        # enable_jupyter=True with initial_port_count - SSH, Jupyter, then 2 more ports
        (
            [r for r in range(20000, 20100)],
            [(22, 20099, 20099), (8888, 20098, 20098), (20000, 20000, 20000), (20001, 20001, 20001)],
            2,
            True,
        ),
    ],
)
@pytest.mark.asyncio
async def test_flexible_mode_port_mappings(
    docker_service, test_executor_id, test_miner_hotkey, available_ports, expected_mappings, initial_port_count, enable_jupyter, monkeypatch
):
    """Test FLEXIBLE mode with various available port scenarios.

    In flexible mode (internal_ports=None):
    - If exact matches exist, use them
    - If no exact matches, docker_port = external_port from available set
    - SSH port (22) gets special handling: max port if not available
    - When enable_jupyter=True, Jupyter port (8888) is inserted at position 1
    """
    # Mock PREFERRED_POD_PORTS to a shorter list for easier testing
    monkeypatch.setattr("services.docker_service.PREFERRED_POD_PORTS", [20000, 20001, 20002, 20003])

    # Mock database response
    mock_ports = create_mock_port_dict(available_ports, test_miner_hotkey, UUID(test_executor_id))
    docker_service.port_mapping_dao.get_available_ports_excluding_rented = AsyncMock(return_value=mock_ports)
    docker_service.port_mapping_dao.get_ports_for_pod = AsyncMock(return_value={})
    docker_service.port_mapping_dao.reserve_ports_for_pod = AsyncMock()

    # Act - internal_ports=None triggers flexible mode
    result = await docker_service.generate_portMappings(
        test_miner_hotkey, test_executor_id, UUID(test_executor_id), None, initial_port_count, enable_jupyter=enable_jupyter
    )
    result = result[0]

    # Assert
    assert len(result) == len(expected_mappings)
    assert set(result) == set(expected_mappings)
    docker_service.port_mapping_dao.get_available_ports_excluding_rented.assert_called_once_with(UUID(test_executor_id))


@pytest.mark.asyncio
async def test_no_exact_match_custom_ports_uses_random_selection(docker_service, test_executor_id, test_miner_hotkey):
    """Test random selection when no exact matches found with custom internal_ports."""
    custom_internal_ports = [8080, 8081, 8082]

    # Available ports don't match requested ports
    mock_ports = create_mock_port_dict([9000, 9001, 9002], test_miner_hotkey, UUID(test_executor_id))
    docker_service.port_mapping_dao.get_available_ports_excluding_rented = AsyncMock(return_value=mock_ports)
    docker_service.port_mapping_dao.get_ports_for_pod = AsyncMock(return_value={})
    docker_service.port_mapping_dao.reserve_ports_for_pod = AsyncMock()

    # Act
    result = await docker_service.generate_portMappings(test_miner_hotkey, test_executor_id, UUID(test_executor_id), custom_internal_ports)
    result = result[0]

    # Assert
    # Expect random selection: docker ports from custom list, external ports from available set
    assert len(result) == 3
    docker_ports_used = {m[0] for m in result}
    assert docker_ports_used == {8080, 8081, 22}

    external_ports_used = {m[2] for m in result}
    assert external_ports_used == {9000, 9001, 9002}
    possible_internal_ports = custom_internal_ports + [22]

    # Verify mapping structure
    for docker_port, internal_port, external_port in result:
        assert docker_port in possible_internal_ports
        assert external_port in {9000, 9001, 9002}
        assert internal_port == external_port

    docker_service.port_mapping_dao.get_available_ports_excluding_rented.assert_called_once_with(UUID(test_executor_id))


@pytest.mark.parametrize("initial_port_count,expected_length,expected_first_port,should_have_extra_ports", [
    # No initial count - returns all PREFERRED_POD_PORTS
    (None, 11, 22, False),
    (2, 2, 22, False),  # +1 for SSH port = 3 total
    (5, 5, 22, False),  # +1 for SSH port = 6 total
    # More than PREFERRED_POD_PORTS length - returns PREFERRED_POD_PORTS + extra
    (11, 11, 22, True),  # +1 for SSH port = 12 total, 1 extra port needed
    (15, 15, 22, True),  # +1 for SSH port = 16 total, 5 extra ports needed
])
def test_get_preferred_ports(
    docker_service,
    initial_port_count,
    expected_length,
    expected_first_port,
    should_have_extra_ports,
    monkeypatch
):
    """Test get_preferred_ports method with various initial_port_count scenarios.

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


@pytest.mark.parametrize(
    "enable_jupyter,internal_ports,available_ports,expected_jupyter_in_mappings,jupyter_port_position",
    [
        # Scenario 1: enable_jupyter=True, STRICT mode, 8888 available (exact match)
        (True, [22, 8080, 8888], [22, 8080, 8888, 9000], True, 1),
        # Scenario 2: enable_jupyter=True, STRICT mode, 8888 not available (random assignment)
        (True, [22, 8080, 8888], [22, 8080, 9000, 9100], True, 1),
        # Scenario 3: enable_jupyter=True, FLEXIBLE mode, 8888 available
        (True, None, [22, 8888, 20000, 20001], True, 1),
        # Scenario 4: enable_jupyter=True, FLEXIBLE mode, 8888 not available
        (True, None, [9000, 9001, 9002, 9003], True, 1),
        # Scenario 5: enable_jupyter=False, should not include jupyter port
        (False, [22, 8080, 9000], [22, 8080, 9000], False, None),
    ],
)
@pytest.mark.asyncio
async def test_enable_jupyter_feature(
    docker_service,
    test_executor_id,
    test_miner_hotkey,
    enable_jupyter,
    internal_ports,
    available_ports,
    expected_jupyter_in_mappings,
    jupyter_port_position,
    monkeypatch,
):
    """Test enable_jupyter feature in various scenarios.

    Covers:
    - STRICT mode (with internal_ports): jupyter port 8888 inserted at position 1
    - FLEXIBLE mode (without internal_ports): jupyter port 8888 inserted at position 1
    - Disabled jupyter: no jupyter port in mappings, jupyter_port_map is None
    """
    # Arrange
    monkeypatch.setattr("services.docker_service.PREFERRED_POD_PORTS", [20000, 20001, 20002, 20003])
    mock_ports = create_mock_port_dict(available_ports, test_miner_hotkey, UUID(test_executor_id))
    docker_service.port_mapping_dao.get_available_ports_excluding_rented = AsyncMock(return_value=mock_ports)
    docker_service.port_mapping_dao.get_ports_for_pod = AsyncMock(return_value={})
    docker_service.port_mapping_dao.reserve_ports_for_pod = AsyncMock()

    # Act
    mappings, jupyter_port_map = await docker_service.generate_portMappings(
        test_miner_hotkey, test_executor_id, UUID(test_executor_id), internal_ports, enable_jupyter=enable_jupyter
    )

    # Assert
    # SSH port should always be first
    assert mappings[0][0] == 22

    if expected_jupyter_in_mappings:
        # Jupyter port should be at specified position
        assert mappings[jupyter_port_position][0] == 8888
        # Jupyter port map should be returned
        assert jupyter_port_map is not None
        assert jupyter_port_map[0] == 8888
        assert jupyter_port_map[1] in available_ports
    else:
        # Jupyter port should not be in mappings
        jupyter_ports = [m for m in mappings if m[0] == 8888]
        assert len(jupyter_ports) == 0
        # Jupyter port map should be None
        assert jupyter_port_map is None


@pytest.mark.asyncio
async def test_pod_mapping_reuse(docker_service, test_executor_id, test_miner_hotkey):
    """Test that existing pod mappings are reused when pod_id is provided."""
    pod_id = uuid4()

    # Existing pod mappings for ports 22, 8080, 8081 (keyed by docker_port)
    existing_pod_mapping = {
        22: create_port_mapping(
            miner_hotkey=test_miner_hotkey,
            executor_id=UUID(test_executor_id),
            internal_port=20000,
            external_port=20000,
            rented_for_pod_id=pod_id,
            docker_port=22,
        ),
        8080: create_port_mapping(
            miner_hotkey=test_miner_hotkey,
            executor_id=UUID(test_executor_id),
            internal_port=20001,
            external_port=20001,
            rented_for_pod_id=pod_id,
            docker_port=8080,
        ),
        8081: create_port_mapping(
            miner_hotkey=test_miner_hotkey,
            executor_id=UUID(test_executor_id),
            internal_port=20002,
            external_port=20002,
            rented_for_pod_id=pod_id,
            docker_port=8081,
        ),
    }

    # Available ports (not used in this test since we have pod_mapping)
    available_ports = create_mock_port_dict(
        [9000, 9001, 9002],
        test_miner_hotkey,
        UUID(test_executor_id)
    )

    docker_service.port_mapping_dao.get_available_ports_excluding_rented = AsyncMock(return_value=available_ports)
    docker_service.port_mapping_dao.get_ports_for_pod = AsyncMock(return_value=existing_pod_mapping)
    docker_service.port_mapping_dao.reserve_ports_for_pod = AsyncMock()

    # Act - request same ports that are in pod_mapping
    requested_ports = [22, 8080, 8081]
    result, _ = await docker_service.generate_portMappings(
        test_miner_hotkey, test_executor_id, pod_id, requested_ports
    )

    # Assert - should reuse existing pod mappings
    assert len(result) == 3
    # Check that we got the exact mappings from pod_mapping (note: internal_port from DB is 20000, 20001, 20002)
    assert (22, 20000, 20000) in result
    assert (8080, 20001, 20001) in result
    assert (8081, 20002, 20002) in result

    # Verify reserve_ports_for_pod was called with correct parameters
    docker_service.port_mapping_dao.reserve_ports_for_pod.assert_called_once()
    call_args = docker_service.port_mapping_dao.reserve_ports_for_pod.call_args
    assert call_args[0][0] == UUID(test_executor_id)
    # Second argument is now mappings (list of tuples), not external_ports
    mappings = call_args[0][1]
    external_ports_from_mappings = {m[2] for m in mappings}
    assert external_ports_from_mappings == {20000, 20001, 20002}
    assert call_args[0][2] == pod_id


@pytest.mark.asyncio
async def test_min_port_count_validation(docker_service, test_executor_id, test_miner_hotkey, monkeypatch):
    """Test that generate_portMappings returns empty when MIN_PORT_COUNT is not met."""
    # Set MIN_PORT_COUNT to 3
    monkeypatch.setattr("services.docker_service.MIN_PORT_COUNT", 3)

    # Only 2 available ports (less than MIN_PORT_COUNT)
    available_ports = create_mock_port_dict(
        [9000, 9001],
        test_miner_hotkey,
        UUID(test_executor_id)
    )

    docker_service.port_mapping_dao.get_available_ports_excluding_rented = AsyncMock(return_value=available_ports)
    docker_service.port_mapping_dao.get_ports_for_pod = AsyncMock(return_value={})

    # Act
    result, jupyter_map = await docker_service.generate_portMappings(
        test_miner_hotkey, test_executor_id, UUID(test_executor_id), [22, 8080, 8081]
    )

    # Assert - should return empty result
    assert result == []
    assert jupyter_map is None


@pytest.mark.asyncio
async def test_reserve_ports_called_with_correct_external_ports(docker_service, test_executor_id, test_miner_hotkey):
    """Test that reserve_ports_for_pod is called with all external ports from mappings."""
    # Available ports with different external port numbers
    available_ports = {
        22: create_port_mapping(
            miner_hotkey=test_miner_hotkey,
            executor_id=UUID(test_executor_id),
            internal_port=20000,
            external_port=20000,
        ),
        8080: create_port_mapping(
            miner_hotkey=test_miner_hotkey,
            executor_id=UUID(test_executor_id),
            internal_port=20001,
            external_port=20001,
        ),
        9999: create_port_mapping(
            miner_hotkey=test_miner_hotkey,
            executor_id=UUID(test_executor_id),
            internal_port=20002,
            external_port=20002,
        ),
    }

    docker_service.port_mapping_dao.get_available_ports_excluding_rented = AsyncMock(return_value=available_ports)
    docker_service.port_mapping_dao.get_ports_for_pod = AsyncMock(return_value={})
    docker_service.port_mapping_dao.reserve_ports_for_pod = AsyncMock()

    pod_id = uuid4()

    # Act
    result, _ = await docker_service.generate_portMappings(
        test_miner_hotkey, test_executor_id, pod_id, [22, 8080, 8081]
    )

    # Assert
    assert len(result) == 3

    # Verify reserve_ports_for_pod was called with correct mappings
    docker_service.port_mapping_dao.reserve_ports_for_pod.assert_called_once()
    call_args = docker_service.port_mapping_dao.reserve_ports_for_pod.call_args
    mappings_reserved = call_args[0][1]  # Second argument is now mappings, not external_ports

    # Verify mappings match the result
    assert mappings_reserved == result
    assert call_args[0][2] == pod_id


@pytest.mark.asyncio
async def test_pod_mapping_partial_reuse(docker_service, test_executor_id, test_miner_hotkey):
    """Test that some ports are reused from pod_mapping and some are allocated from available."""
    pod_id = uuid4()

    # Existing pod mappings only for port 22
    existing_pod_mapping = {
        22: create_port_mapping(
            miner_hotkey=test_miner_hotkey,
            executor_id=UUID(test_executor_id),
            internal_port=20000,
            external_port=20000,
            rented_for_pod_id=pod_id,
        ),
    }

    # Available ports for the rest
    available_ports = create_mock_port_dict(
        [8080, 8081, 9000],
        test_miner_hotkey,
        UUID(test_executor_id)
    )

    docker_service.port_mapping_dao.get_available_ports_excluding_rented = AsyncMock(return_value=available_ports)
    docker_service.port_mapping_dao.get_ports_for_pod = AsyncMock(return_value=existing_pod_mapping)
    docker_service.port_mapping_dao.reserve_ports_for_pod = AsyncMock()

    # Act - request ports: 22 (in pod_mapping), 8080, 8081 (from available)
    requested_ports = [22, 8080, 8081]
    result, _ = await docker_service.generate_portMappings(
        test_miner_hotkey, test_executor_id, pod_id, requested_ports
    )

    # Assert
    assert len(result) == 3
    # Port 22 should be reused from pod_mapping
    assert (22, 20000, 20000) in result
    # Ports 8080 and 8081 should be allocated from available
    assert (8080, 8080, 8080) in result
    assert (8081, 8081, 8081) in result


# =============================================================================
# Tests for _convert_payload_ports and backend data flow
# =============================================================================

@pytest.mark.parametrize("available_raw,pod_raw,expected_available_keys,expected_pod_keys", [
    # Only available ports
    (
        [PayloadPortMapping(internal_port=p, external_port=p, docker_port=None) for p in [20000, 20001, 20002]],
        [],
        {20000, 20001, 20002},
        set(),
    ),
    # Pod mapping with docker_port as key
    (
        [],
        [PayloadPortMapping(internal_port=20000, external_port=20000, docker_port=22),
         PayloadPortMapping(internal_port=20001, external_port=20001, docker_port=8080)],
        set(),
        {22, 8080},
    ),
    # Pod mapping fallback to external_port when docker_port is None
    (
        [],
        [PayloadPortMapping(internal_port=20000, external_port=20000, docker_port=None)],
        set(),
        {20000},
    ),
])
def test_convert_payload_ports(docker_service, available_raw, pod_raw, expected_available_keys, expected_pod_keys):
    """Test _convert_payload_ports conversion logic."""
    available, pod = docker_service._convert_payload_ports(available_raw, pod_raw)
    assert set(available.keys()) == expected_available_keys
    assert set(pod.keys()) == expected_pod_keys


@pytest.mark.asyncio
async def test_generate_portMappings_uses_backend_data(docker_service, test_executor_id, test_miner_hotkey):
    """Test backend data is used instead of DB when provided."""
    available_raw = [PayloadPortMapping(internal_port=p, external_port=p, docker_port=None) for p in [22, 8080, 8081]]
    docker_service.port_mapping_dao.get_available_ports_excluding_rented = AsyncMock()
    docker_service.port_mapping_dao.get_ports_for_pod = AsyncMock()
    docker_service.port_mapping_dao.reserve_ports_for_pod = AsyncMock()

    result, _ = await docker_service.generate_portMappings(
        test_miner_hotkey, test_executor_id, uuid4(), [22, 8080, 8081],
        available_ports_raw=available_raw, pod_mapping_raw=[],
    )

    assert set(result) == {(22, 22, 22), (8080, 8080, 8080), (8081, 8081, 8081)}
    docker_service.port_mapping_dao.get_available_ports_excluding_rented.assert_not_called()
    docker_service.port_mapping_dao.get_ports_for_pod.assert_not_called()


@pytest.mark.asyncio
async def test_generate_portMappings_falls_back_to_db(docker_service, test_executor_id, test_miner_hotkey):
    """Test fallback to DB when backend data is None."""
    mock_ports = create_mock_port_dict([22, 8080, 8081], test_miner_hotkey, UUID(test_executor_id))
    docker_service.port_mapping_dao.get_available_ports_excluding_rented = AsyncMock(return_value=mock_ports)
    docker_service.port_mapping_dao.get_ports_for_pod = AsyncMock(return_value={})
    docker_service.port_mapping_dao.reserve_ports_for_pod = AsyncMock()

    result, _ = await docker_service.generate_portMappings(
        test_miner_hotkey, test_executor_id, uuid4(), [22, 8080, 8081],
    )

    assert len(result) == 3
    docker_service.port_mapping_dao.get_available_ports_excluding_rented.assert_called_once()
    docker_service.port_mapping_dao.get_ports_for_pod.assert_called_once()


@pytest.mark.asyncio
async def test_generate_portMappings_with_backend_pod_mapping(docker_service, test_executor_id, test_miner_hotkey):
    """Test pod mappings from backend are applied correctly."""
    available_raw = [PayloadPortMapping(internal_port=p, external_port=p, docker_port=None) for p in [9000, 9001]]
    pod_raw = [PayloadPortMapping(internal_port=20000, external_port=20000, docker_port=22)]
    docker_service.port_mapping_dao.reserve_ports_for_pod = AsyncMock()

    result, _ = await docker_service.generate_portMappings(
        test_miner_hotkey, test_executor_id, uuid4(), [22, 8080, 8081],
        available_ports_raw=available_raw, pod_mapping_raw=pod_raw,
    )

    assert len(result) == 3
    assert (22, 20000, 20000) in result  # from backend pod_mapping


# =============================================================================
# Tests for clean_existing_containers
# =============================================================================


def _make_ssh_run_result(stdout: str):
    """Helper to create a mock SSH run result."""
    mock_result = Mock()
    mock_result.stdout = stdout
    return mock_result


class DummySSHConnectionManager:
    def __init__(self, ssh_client):
        self.ssh_client = ssh_client

    async def __aenter__(self):
        return self.ssh_client

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None


@pytest.fixture
def retry_ssh_mock(monkeypatch):
    """Mock retry_ssh_command to capture SSH commands without executing."""
    import services.docker_service as ds_module
    mock = AsyncMock()
    monkeypatch.setattr(ds_module, "retry_ssh_command", mock)
    return mock


def _make_ssh_command_result(exit_status: int = 0, stdout: str = "", stderr: str = ""):
    result = Mock()
    result.exit_status = exit_status
    result.stdout = stdout
    result.stderr = stderr
    return result


@pytest.mark.asyncio
async def test_install_ssh_service_creates_bootstrap_script_inside_container(docker_service):
    """SSH bootstrap script is written directly in-container before execution."""
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(return_value=_make_ssh_command_result(exit_status=0))
    docker_service.execute_and_stream_logs = AsyncMock(return_value=(True, ""))

    result = await docker_service.install_open_ssh_server_and_start_ssh_service(
        ssh_client=ssh_client,
        container_name="pod_test",
        log_tag="test_log",
        log_extra={"pod_id": "pod-id"},
    )

    assert result is True
    ssh_client.run.assert_awaited_once()
    create_command = ssh_client.run.await_args_list[0].args[0]
    assert create_command.startswith('/usr/bin/docker exec -i pod_test sh -c ')
    assert "cat > /tmp/lium-ssh-bootstrap.sh" in create_command
    assert "chmod +x /tmp/lium-ssh-bootstrap.sh" in create_command
    assert "<< '__LIUM_SSHD_BOOTSTRAP_" in create_command
    assert "/usr/bin/docker cp" not in create_command

    assert docker_service.execute_and_stream_logs.await_count == 1
    commands = [call.kwargs["command"] for call in docker_service.execute_and_stream_logs.await_args_list]
    assert commands[0].endswith(" sh /tmp/lium-ssh-bootstrap.sh")
    assert all("/usr/bin/docker cp" not in command for command in commands)


@pytest.mark.asyncio
async def test_install_ssh_service_returns_false_when_in_container_script_creation_fails(docker_service):
    """Script creation failure returns False and skips bootstrap execution."""
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(return_value=_make_ssh_command_result(exit_status=1, stderr="permission denied"))
    docker_service.execute_and_stream_logs = AsyncMock(return_value=(True, ""))
    docker_service.stream_log = AsyncMock()

    result = await docker_service.install_open_ssh_server_and_start_ssh_service(
        ssh_client=ssh_client,
        container_name="pod_test",
        log_tag="test_log",
        log_extra={"pod_id": "pod-id"},
    )

    assert result is False
    ssh_client.run.assert_awaited_once()
    docker_service.execute_and_stream_logs.assert_not_awaited()


def test_ssh_bootstrap_script_supports_multi_distro_install(docker_service):
    """The injected SSH bootstrap script supports the expected package managers."""
    script = docker_service._ssh_bootstrap_script_path().read_text()

    assert "apt-get update" in script
    assert "apt-get install -y openssh-server" in script
    assert "apk add --no-cache openssh" in script
    assert "dnf install -y openssh-server" in script
    assert "yum install -y openssh-server" in script


def test_ssh_bootstrap_script_uses_single_watchdog_with_30_second_sleep(docker_service):
    """The watchdog loop is single-instance and checks sshd every 30 seconds."""
    script = docker_service._ssh_bootstrap_script_path().read_text()

    assert 'WATCHDOG_PIDFILE="/run/sshd-watchdog.pid"' in script
    assert 'WATCHDOG_LOG="/tmp/sshd-watchdog.log"' in script
    assert 'SLEEP_SECONDS=30' in script
    assert 'kill -0 "$watchdog_pid"' in script
    assert 'nohup sh "$SCRIPT_PATH" --watchdog-loop' in script
    assert 'sleep "$SLEEP_SECONDS"' in script


@pytest.mark.asyncio
async def test_start_container_restarts_ssh_after_docker_start(docker_service, monkeypatch):
    """start_container reruns the SSH bootstrap helper after docker start."""
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock()
    monkeypatch.setattr("services.docker_service.asyncssh.import_private_key", Mock(return_value="pkey"))
    monkeypatch.setattr(
        "services.docker_service.asyncssh.connect",
        Mock(return_value=DummySSHConnectionManager(ssh_client)),
    )

    docker_service.ssh_service.decrypt_payload.return_value = "private-key"
    docker_service._prepare_known_hosts_policy = AsyncMock(return_value=None)
    docker_service.install_open_ssh_server_and_start_ssh_service = AsyncMock(return_value=True)

    payload = ContainerStartRequest(
        miner_hotkey="miner-hotkey",
        miner_address="127.0.0.1",
        miner_port=8000,
        executor_id=str(uuid4()),
        pod_id="pod-id",
        container_name="pod_test",
    )
    executor_info = ExecutorSSHInfo(
        uuid=str(uuid4()),
        address="127.0.0.1",
        port=8001,
        ssh_username="root",
        ssh_port=2200,
        python_path="/usr/bin/python3",
        root_dir="/root/app",
    )
    keypair = Mock(ss58_address="validator-hotkey")

    await docker_service.start_container(payload, executor_info, keypair, "encrypted-private-key")

    ssh_client.run.assert_awaited_once_with("/usr/bin/docker start pod_test")
    docker_service.install_open_ssh_server_and_start_ssh_service.assert_awaited_once_with(
        ssh_client=ssh_client,
        container_name="pod_test",
        log_tag="start_container_pod-id",
        log_extra={
            "miner_hotkey": "miner-hotkey",
            "executor_uuid": payload.executor_id,
            "executor_ip_address": "127.0.0.1",
            "executor_port": 8001,
            "executor_ssh_username": "root",
            "executor_ssh_port": 2200,
        },
    )


@pytest.mark.asyncio
async def test_start_container_logs_ssh_bootstrap_failure_and_keeps_starting(docker_service, monkeypatch):
    """A failed SSH bootstrap does not interrupt docker start."""
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock()
    monkeypatch.setattr("services.docker_service.asyncssh.import_private_key", Mock(return_value="pkey"))
    monkeypatch.setattr(
        "services.docker_service.asyncssh.connect",
        Mock(return_value=DummySSHConnectionManager(ssh_client)),
    )

    docker_service.ssh_service.decrypt_payload.return_value = "private-key"
    docker_service._prepare_known_hosts_policy = AsyncMock(return_value=None)
    docker_service.install_open_ssh_server_and_start_ssh_service = AsyncMock(return_value=False)

    payload = ContainerStartRequest(
        miner_hotkey="miner-hotkey",
        miner_address="127.0.0.1",
        miner_port=8000,
        executor_id=str(uuid4()),
        pod_id="pod-id",
        container_name="pod_test",
    )
    executor_info = ExecutorSSHInfo(
        uuid=str(uuid4()),
        address="127.0.0.1",
        port=8001,
        ssh_username="root",
        ssh_port=2200,
        python_path="/usr/bin/python3",
        root_dir="/root/app",
    )
    keypair = Mock(ss58_address="validator-hotkey")
    logger_mock = Mock()
    monkeypatch.setattr("services.docker_service.logger.warning", logger_mock)

    await docker_service.start_container(payload, executor_info, keypair, "encrypted-private-key")

    ssh_client.run.assert_awaited_once_with("/usr/bin/docker start pod_test")
    docker_service.install_open_ssh_server_and_start_ssh_service.assert_awaited_once()
    assert logger_mock.called


@pytest.mark.asyncio
async def test_clean_containers_active_siblings_not_removed(docker_service, retry_ssh_mock):
    """Active sibling containers listed in active_container_names must be preserved."""
    # Arrange
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(return_value=_make_ssh_run_result(
        "pod_abc123\npod_sibling1\npod_sibling2\n"
    ))

    # Act
    await docker_service.clean_existing_containers(
        ssh_client=ssh_client,
        default_extra={},
        pod_name="pod_abc123",
        active_container_names=["pod_sibling1", "pod_sibling2"],
    )

    # Assert — only target pod removed, active siblings preserved
    rm_command = retry_ssh_mock.call_args_list[0][0][1]
    assert "pod_abc123" in rm_command
    assert "pod_sibling1" not in rm_command
    assert "pod_sibling2" not in rm_command


@pytest.mark.asyncio
async def test_clean_containers_stale_pods_removed(docker_service, retry_ssh_mock):
    """Stale pod_ containers not in active list should be cleaned up."""
    # Arrange
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(return_value=_make_ssh_run_result(
        "pod_target\npod_stale_orphan\npod_active_sibling\nsome_other_container\n"
    ))

    # Act
    await docker_service.clean_existing_containers(
        ssh_client=ssh_client,
        default_extra={},
        pod_name="pod_target",
        active_container_names=["pod_active_sibling"],
    )

    # Assert — target + stale orphan removed; active sibling and non-pod container kept
    rm_command = retry_ssh_mock.call_args_list[0][0][1]
    assert "pod_target" in rm_command
    assert "pod_stale_orphan" in rm_command
    assert "pod_active_sibling" not in rm_command
    # non-pod containers are never touched
    assert "some_other_container" not in rm_command


@pytest.mark.asyncio
async def test_clean_containers_none_fallback_removes_all_pods(docker_service, retry_ssh_mock):
    """When active_container_names is None (old backend), all pod_ containers are removed."""
    # Arrange
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(return_value=_make_ssh_run_result(
        "pod_target\npod_other_pod\nsome_container\n"
    ))

    # Act
    await docker_service.clean_existing_containers(
        ssh_client=ssh_client,
        default_extra={},
        pod_name="pod_target",
        active_container_names=None,
    )

    # Assert — old behavior: all pod_ containers removed, non-pod containers untouched
    rm_command = retry_ssh_mock.call_args_list[0][0][1]
    assert "pod_target" in rm_command
    assert "pod_other_pod" in rm_command
    assert "some_container" not in rm_command


@pytest.mark.asyncio
async def test_clean_containers_targeted_volume_rm(docker_service, retry_ssh_mock):
    """Volume cleanup removes volumes for all removed containers, not prune -af."""
    # Arrange
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(return_value=_make_ssh_run_result(
        "pod_target\npod_stale\npod_active_sibling\n"
    ))

    # Act
    await docker_service.clean_existing_containers(
        ssh_client=ssh_client,
        default_extra={},
        pod_name="pod_target",
        clear_volume=True,
        active_container_names=["pod_active_sibling"],
    )

    # Assert — two calls: container rm + volume rm for all removed containers
    assert retry_ssh_mock.call_count == 2
    volume_command = retry_ssh_mock.call_args_list[1][0][1]
    assert "volume_target" in volume_command
    assert "volume_stale" in volume_command
    assert "volume_active_sibling" not in volume_command
    assert "volume prune" not in volume_command


@pytest.mark.asyncio
async def test_clean_containers_empty_active_list_falls_back_to_prune(docker_service, retry_ssh_mock):
    """When active_container_names is empty list, volume prune -af is used (old behavior)."""
    # Arrange
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(return_value=_make_ssh_run_result(
        "pod_target\npod_stale\n"
    ))

    # Act
    await docker_service.clean_existing_containers(
        ssh_client=ssh_client,
        default_extra={},
        pod_name="pod_target",
        clear_volume=True,
        active_container_names=[],
    )

    # Assert — falls back to prune, not targeted volume rm
    assert retry_ssh_mock.call_count == 2
    volume_command = retry_ssh_mock.call_args_list[1][0][1]
    assert "volume prune -af" in volume_command


@pytest.mark.asyncio
async def test_clean_containers_no_volume_cleanup_when_disabled(docker_service, retry_ssh_mock):
    """No volume cleanup when clear_volume=False (e.g. local volume reuse)."""
    # Arrange
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(return_value=_make_ssh_run_result("pod_abc123\n"))

    # Act
    await docker_service.clean_existing_containers(
        ssh_client=ssh_client,
        default_extra={},
        pod_name="pod_abc123",
        clear_volume=False,
        active_container_names=[],
    )

    # Assert — only one call for container rm, no volume command
    assert retry_ssh_mock.call_count == 1
    assert "docker rm" in retry_ssh_mock.call_args_list[0][0][1]
