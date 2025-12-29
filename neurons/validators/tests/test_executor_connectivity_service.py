from unittest.mock import AsyncMock

import pytest

from services.executor_connectivity_service import (
    ExecutorConnectivityService,
    PortSelector,
    NetcatScript,
    PortPair,
    AlpineContainer,
    DindVerifier,
)
from services.const import PREFERRED_POD_PORTS


@pytest.fixture
def executor_service(mock_redis_service, port_mapping_dao):
    """Create ExecutorConnectivityService for testing."""
    return ExecutorConnectivityService(mock_redis_service, port_mapping_dao)


# ========================================================================================
# Tests for get_available_port_maps method
# ========================================================================================


def test_get_available_port_maps_from_mappings(sample_executor_info):
    """Test port extraction from JSON port_mappings."""
    # Arrange
    batch_size = 2
    rented_ports = set()

    # Act
    result = PortSelector.select(sample_executor_info, batch_size, rented_ports)

    # Assert
    # Expect exactly 2 port pairs because batch_size=2 limits the result
    assert len(result) == 2
    # Expect all ports to be from sample_executor_info range (9000-10004) and match internal=external
    for port in result:
        assert 9000 <= port.internal <= 10004
        assert port.internal == port.external
        # Expect SSH port 22 to be excluded from available ports
        assert port.internal != 22


def test_get_available_port_maps_from_range():
    """Test port generation from port_range string."""
    # Arrange
    from datura.requests.miner_requests import ExecutorSSHInfo

    executor_info = ExecutorSSHInfo(
        uuid="550e8400-e29b-41d4-a716-446655440001",
        address="127.0.0.1",
        port=8080,
        ssh_username="root",
        ssh_port=22,
        port_mappings=None,
        port_range="9000-9005",
        python_path="/usr/bin/python3",
        root_dir="/tmp",
    )
    batch_size = 3
    rented_ports = set()

    # Act
    result = PortSelector.select(executor_info, batch_size, rented_ports)

    # Assert
    # Expect exactly batch_size ports because we requested 3 and have 6 available
    assert len(result) == batch_size
    # Expect all ports to be from the specified range 9000-9005
    valid_ports = {9000, 9001, 9002, 9003, 9004, 9005}
    for port in result:
        # Expect internal and external ports to be identical (no NAT mapping)
        assert port.internal == port.external
        # Expect selected ports to be from the specified range
        assert port.internal in valid_ports


def test_get_available_port_maps_default_range():
    """Test fallback to default port range when no mappings or range provided."""
    # Arrange
    from datura.requests.miner_requests import ExecutorSSHInfo

    executor_info = ExecutorSSHInfo(
        uuid="550e8400-e29b-41d4-a716-446655440002",
        address="127.0.0.1",
        port=8080,
        ssh_username="root",
        ssh_port=22,
        port_mappings=None,
        port_range=None,
        python_path="/usr/bin/python3",
        root_dir="/tmp",
    )
    batch_size = 5
    rented_ports = set()

    # Act
    result = PortSelector.select(executor_info, batch_size, rented_ports)

    # Assert
    # Expect exactly batch_size ports because we requested 5
    assert len(result) == batch_size
    for port in result:
        # Expect ports from default range 20000-65535 when no range specified
        assert 20000 <= port.internal <= 65535
        # Expect internal and external ports to be identical
        assert port.internal == port.external
        # Expect SSH port 22 to be excluded
        assert port.internal != 22


def test_get_available_port_maps_empty_range():
    """Test fallback to default range when port_range is empty string."""
    # Arrange
    from datura.requests.miner_requests import ExecutorSSHInfo

    executor_info = ExecutorSSHInfo(
        uuid="550e8400-e29b-41d4-a716-446655440003",
        address="127.0.0.1",
        port=8080,
        ssh_username="root",
        ssh_port=22,
        port_mappings=None,
        port_range="",
        python_path="/usr/bin/python3",
        root_dir="/tmp",
    )
    batch_size = 5
    rented_ports = set()

    # Act
    result = PortSelector.select(executor_info, batch_size, rented_ports)

    # Assert
    # Expect exactly batch_size ports
    assert len(result) == batch_size
    for port in result:
        # Expect default range 20000-65535 when port_range is empty string
        assert 20000 <= port.internal <= 65535


# ========================================================================================
# Tests for verify_ports method
# ========================================================================================


@pytest.mark.asyncio
async def test_verify_ports_invalid_json_mappings(executor_service, mock_ssh_client):
    """Test that verify_ports fails when port_mappings contains invalid JSON."""
    # Arrange
    from datura.requests.miner_requests import ExecutorSSHInfo

    executor_info = ExecutorSSHInfo(
        uuid="550e8400-e29b-41d4-a716-446655440004",
        address="127.0.0.1",
        port=8080,
        ssh_username="root",
        ssh_port=22,
        port_mappings="invalid json",
        port_range=None,
        python_path="/usr/bin/python3",
        root_dir="/tmp",
    )
    # Mock SSH cleanup to return empty container list
    mock_ssh_client.run.return_value.stdout = ""

    # Act
    result = await executor_service.verify_ports(
        mock_ssh_client, "job_123", "miner_key", executor_info, "private_key", "public_key"
    )

    # Assert
    # Expect failure because invalid JSON will cause json.loads() to fail
    assert result.success is False
    # Expect error message about JSON parsing failure
    assert "Expecting value" in result.log_text or "Verification failed" in result.log_text


def test_get_available_port_maps_preferred_ports_priority():
    """Test that preferred ports are prioritized when available in port_range."""
    # Arrange
    from datura.requests.miner_requests import ExecutorSSHInfo

    # Create port range that includes some preferred ports (20000-20009 are preferred)
    executor_info = ExecutorSSHInfo(
        uuid="550e8400-e29b-41d4-a716-446655440005",
        address="127.0.0.1",
        port=8080,
        ssh_username="root",
        ssh_port=22,
        port_mappings=None,
        port_range="20000-20090",  # Includes preferred ports 20000-20009
        python_path="/usr/bin/python3",
        root_dir="/tmp",
    )
    batch_size = 15
    rented_ports = set()

    # Act
    result = PortSelector.select(executor_info, batch_size, rented_ports)

    # Assert
    # Expect exactly batch_size ports
    assert len(result) == batch_size

    selected_ports = [port.internal for port in result]
    preferred_in_range = [port for port in PREFERRED_POD_PORTS if (20000 <= port <= 20090)]
    preferred_selected = [port for port in selected_ports if port in PREFERRED_POD_PORTS]

    # Expect at least some preferred ports to be selected
    assert len(preferred_selected) > 0

    # Expect all preferred ports within range to be included (they should be prioritized)
    for preferred_port in preferred_in_range:
        assert preferred_port in selected_ports


def test_get_available_port_maps_preferred_mappings_priority():
    """Test that preferred port mappings are prioritized from JSON mappings."""
    # Arrange
    from datura.requests.miner_requests import ExecutorSSHInfo
    import json

    # Create mappings with preferred (20000-20009) and non-preferred ports
    port_mappings = [
        [20000, 20000],  # Preferred port
        [20001, 20001],  # Preferred port
        [9000, 9000],  # Non-preferred
        [9001, 9001],  # Non-preferred
        [9002, 9002],  # Non-preferred
    ]

    executor_info = ExecutorSSHInfo(
        uuid="550e8400-e29b-41d4-a716-446655440006",
        address="127.0.0.1",
        port=8080,
        ssh_username="root",
        ssh_port=22,
        port_mappings=json.dumps(port_mappings),
        port_range=None,
        python_path="/usr/bin/python3",
        root_dir="/tmp",
    )
    batch_size = 3
    rented_ports = set()

    # Act
    result = PortSelector.select(executor_info, batch_size, rented_ports)

    # Assert
    # Expect exactly batch_size ports
    assert len(result) == batch_size

    selected_ports = [port.internal for port in result]

    # Expect preferred ports 20000 and 20001 to be included because they are prioritized
    assert 20000 in selected_ports
    assert 20001 in selected_ports


# ========================================================================================
# Tests for cleanup_docker_containers method
# ========================================================================================


@pytest.mark.asyncio
async def test_cleanup_docker_containers(executor_service, mock_ssh_client, sample_executor_info):
    """Test cleanup of Docker containers with 'container_' prefix."""
    # Arrange
    # Mock responses: 1) list containers command, 2) rm command, 3) prune command
    # Note: docker ps --filter "name=^/container_" only returns container_* names
    mock_ssh_client.run.side_effect = [
        AsyncMock(stdout="container_test1\ncontainer_test2", exit_status=0),
        AsyncMock(stdout="", exit_status=0),  # docker rm response
        AsyncMock(stdout="", exit_status=0),  # docker volume prune response
    ]

    # Act - using private method _cleanup_old_containers
    await executor_service._cleanup_old_containers(mock_ssh_client, sample_executor_info, {})

    # Assert
    # Expect 3 SSH commands: list, rm, prune
    assert mock_ssh_client.run.call_count == 3

    all_calls = [call.args[0] for call in mock_ssh_client.run.call_args_list]

    # Expect first call to list containers with name filter
    assert "docker ps" in all_calls[0]
    # Expect second call to remove found containers (both with container_ prefix)
    assert "docker rm" in all_calls[1] and "container_test1" in all_calls[1] and "container_test2" in all_calls[1]
    # Expect third call to prune volumes
    assert "docker volume prune" in all_calls[2]


# ========================================================================================
# Tests for verify_port_dind method
# ========================================================================================


@pytest.mark.asyncio
async def test_verify_port_dind_successful_connection(mock_ssh_client, sample_executor_info):
    """Test successful Docker-in-Docker verification with SSH connection."""
    # Arrange
    from unittest.mock import patch, MagicMock

    miner_hotkey = "test_miner"
    port = PortPair(9000, 9000)
    private_key = "-----BEGIN PRIVATE KEY-----\ntest_key\n-----END PRIVATE KEY-----"
    public_key = "ssh-rsa test_public_key"

    # Mock docker run command (successful)
    mock_ssh_client.run.side_effect = [
        AsyncMock(exit_status=0, stdout="container_id", stderr=""),  # docker run
        AsyncMock(exit_status=0, stdout="", stderr=""),  # docker rm cleanup
    ]

    # Mock asyncssh imports and connection
    mock_pkey = MagicMock()
    mock_container_ssh = AsyncMock()

    dind_verifier = DindVerifier(mock_ssh_client, sample_executor_info.address, {})

    with patch('asyncssh.import_private_key', return_value=mock_pkey) as mock_import_key, \
         patch('asyncssh.connect') as mock_connect, \
         patch('asyncio.sleep', new=AsyncMock()) as mock_sleep:

        # Setup asyncssh.connect as async context manager
        mock_connect.return_value.__aenter__.return_value = mock_container_ssh
        mock_connect.return_value.__aexit__.return_value = AsyncMock()

        # Act
        result = await dind_verifier.verify(port, miner_hotkey, private_key, public_key, sysbox=False)

        # Assert
        # Expect success because docker container created and SSH connected
        assert result.success is True
        # Expect success message with port number
        assert "dind: check ok" in result.log_text
        assert str(port.internal) in result.log_text

        # Expect docker run command was called with correct parameters
        docker_run_call = mock_ssh_client.run.call_args_list[0][0][0]
        assert "/usr/bin/docker run" in docker_run_call
        assert f"container_{miner_hotkey}_{port.external}" in docker_run_call
        assert f"-p {port.internal}:22" in docker_run_call

        # Expect SSH private key was imported
        mock_import_key.assert_called_once_with(private_key)

        # Expect SSH connection to container was established
        mock_connect.assert_called_once()
        connect_kwargs = mock_connect.call_args[1]
        assert connect_kwargs['host'] == sample_executor_info.address
        assert connect_kwargs['port'] == port.external
        assert connect_kwargs['username'] == 'root'

        # Expect container cleanup was called
        cleanup_call = mock_ssh_client.run.call_args_list[1][0][0]
        assert "docker rm" in cleanup_call
        assert f"container_{miner_hotkey}_{port.external}" in cleanup_call


@pytest.mark.asyncio
async def test_verify_ports_successful_flow(executor_service, mock_ssh_client, sample_executor_info):
    """Test complete successful verification flow with all components."""
    # Arrange
    from unittest.mock import patch
    from services.executor_connectivity_service import DockerConnectionCheckResult

    job_batch_id = "job_123"
    miner_hotkey = "test_miner"
    private_key = "test_private_key"
    public_key = "test_public_key"
    rented_ports = [8000, 8001]
    rented_pod_names = ["pod_1", "pod_2"]

    # Mock port selection and verification
    port_pairs = [PortPair(9000, 9000), PortPair(9001, 9001), PortPair(9002, 9002)]
    successful_bulk_ports = [PortPair(9001, 9001), PortPair(9002, 9002)]
    failed_bulk_ports = []

    with patch.object(executor_service, '_cleanup_old_containers', new=AsyncMock()) as mock_cleanup, \
         patch.object(PortSelector, 'select', return_value=port_pairs) as mock_get_ports, \
         patch('services.executor_connectivity_service.BatchVerifier.verify', new=AsyncMock(return_value=(successful_bulk_ports, failed_bulk_ports))) as mock_bulk, \
         patch('services.executor_connectivity_service.DindVerifier.verify', new=AsyncMock(return_value=DockerConnectionCheckResult(success=True, log_text="dind ok", sysbox_runtime=False))) as mock_dind, \
         patch.object(executor_service, '_save_results', new=AsyncMock()) as mock_save_db, \
         patch.object(executor_service.dao, 'get_busy_external_ports', new=AsyncMock(return_value=set())) as mock_get_busy_ports:

        # Act
        result = await executor_service.verify_ports(
            mock_ssh_client,
            job_batch_id,
            miner_hotkey,
            sample_executor_info,
            private_key,
            public_key,
            rented_ports=rented_ports,
            rented_pod_names=rented_pod_names,
        )

        # Assert
        # Expect success because all steps succeeded
        assert result.success is True
        # Expect log_text contains success summary with verification stats
        assert "verification complete" in result.log_text
        assert "available" in result.log_text

        # Expect cleanup was called first with ssh_client, executor_info, pod_names, and extra dict
        mock_cleanup.assert_called_once()
        cleanup_args = mock_cleanup.call_args
        assert cleanup_args[0][0] == mock_ssh_client
        assert cleanup_args[0][1] == sample_executor_info
        # Expect extra dict contains job metadata
        assert "job_batch_id" in cleanup_args[0][2]
        assert "miner_hotkey" in cleanup_args[0][2]

        # Expect PortSelector.select was called with correct batch size and rented ports
        from services.const import BATCH_PORT_VERIFICATION_SIZE
        mock_get_ports.assert_called_once_with(sample_executor_info, BATCH_PORT_VERIFICATION_SIZE, set(rented_ports))

        # Expect BatchVerifier.verify was called
        mock_bulk.assert_called_once()

        # Expect DindVerifier.verify was called with first successful port from bulk
        mock_dind.assert_called_once()
        dind_call_args = mock_dind.call_args[0]
        # First arg should be a PortPair (9001, 9001) since it was popped from successful_bulk_ports
        assert dind_call_args[0].internal == 9001
        assert dind_call_args[0].external == 9001
        assert dind_call_args[1] == miner_hotkey

        # Expect _save_results was called with successful and failed ports
        mock_save_db.assert_called_once()
        db_successful_ports = mock_save_db.call_args[0][2]
        db_failed_ports = mock_save_db.call_args[0][3]
        # Expect 2 successful ports (dind port re-added after verification)
        assert len(db_successful_ports) == 2
        # Expect 0 failed ports because all verifications succeeded
        assert len(db_failed_ports) == 0


# ========================================================================================
# Tests for verify_ports_bulk helper methods
# ========================================================================================


def test_build_netcat_script():
    """Test netcat script generation - pure function, no mocking needed!"""
    port_pairs = [PortPair(9000, 9000), PortPair(9001, 9001)]
    token = "abc123"

    script = NetcatScript.batch(port_pairs, token, 0)

    # Verify script contains token
    assert token in script
    # Verify script contains all ports
    assert "9000" in script
    assert "9001" in script
    # Verify script has batch structure
    assert "Batch 0" in script
    assert "nc -l -p" in script


@pytest.mark.asyncio
async def test_start_port_test_container_success(mock_ssh_client):
    """Test container starts successfully."""
    from unittest.mock import patch

    mock_ssh_client.run.side_effect = [
        AsyncMock(exit_status=0, stdout="container_abc12345\n"),  # docker run
        AsyncMock(exit_status=0, stdout="Up 2 seconds"),  # docker ps
    ]

    alpine = AlpineContainer(mock_ssh_client, {})

    with patch('asyncio.sleep', new=AsyncMock()):
        result = await alpine.start_and_verify("port_test_abc12345", "echo 'test'", "host", 60)

    assert result is True
    assert "docker.io/library/alpine:3.19" in mock_ssh_client.run.call_args_list[0][0][0]


@pytest.mark.asyncio
async def test_start_port_test_container_fails_to_start(mock_ssh_client):
    """Test container fails to start."""
    mock_ssh_client.run.return_value = AsyncMock(exit_status=1, stderr="Error: unable to start")

    alpine = AlpineContainer(mock_ssh_client, {})
    result = await alpine.start_and_verify("port_test_fail", "echo 'test'", "host", 60)

    assert result is False


@pytest.mark.asyncio
async def test_start_port_test_container_exits_immediately(mock_ssh_client):
    """Test container starts but exits immediately."""
    from unittest.mock import patch

    mock_ssh_client.run.side_effect = [
        AsyncMock(exit_status=0, stdout="container_dead\n"),  # docker run
        AsyncMock(exit_status=0, stdout=""),  # docker ps (empty = not running)
        AsyncMock(exit_status=0, stdout="Error: nc failed"),  # docker logs
    ]

    alpine = AlpineContainer(mock_ssh_client, {})

    with patch('asyncio.sleep', new=AsyncMock()):
        result = await alpine.start_and_verify("port_test_dead", "echo 'test'", "host", 60)

    assert result is False
    assert "docker logs" in mock_ssh_client.run.call_args_list[2][0][0]


@pytest.mark.asyncio
async def test_cleanup_port_test_container(mock_ssh_client):
    """Test container cleanup."""
    alpine = AlpineContainer(mock_ssh_client, {})
    await alpine.cleanup("port_test_xyz")

    assert mock_ssh_client.run.called
    assert "docker rm -f port_test_xyz" in mock_ssh_client.run.call_args[0][0]
