from unittest.mock import AsyncMock

import pytest

from services.executor_connectivity_service import ExecutorConnectivityService
from services.const import PREFERRED_POD_PORTS


@pytest.fixture
def executor_service(mock_redis_service, port_mapping_dao):
    """Create ExecutorConnectivityService for testing."""
    return ExecutorConnectivityService(mock_redis_service, port_mapping_dao)


# ========================================================================================
# Tests for get_available_port_maps method
# ========================================================================================


def test_get_available_port_maps_from_mappings(executor_service, sample_executor_info):
    """Test port extraction from JSON port_mappings."""
    # Arrange
    batch_size = 2

    # Act
    result = executor_service.get_available_port_maps(sample_executor_info, batch_size)

    # Assert
    # Expect exactly 2 port pairs because batch_size=2 limits the result
    assert len(result) == 2
    # Expect all ports to be from sample_executor_info range (9000-10004) and match internal=external
    for internal_port, external_port in result:
        assert 9000 <= internal_port <= 10004
        assert internal_port == external_port
        # Expect SSH port 22 to be excluded from available ports
        assert internal_port != 22


def test_get_available_port_maps_from_range(executor_service):
    """Test port generation from port_range string."""
    # Arrange
    from datura.requests.miner_requests import ExecutorSSHInfo

    executor_info = ExecutorSSHInfo(
        uuid="test",
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

    # Act
    result = executor_service.get_available_port_maps(executor_info, batch_size)

    # Assert
    # Expect exactly batch_size ports because we requested 3 and have 6 available
    assert len(result) == batch_size
    # Expect all ports to be from the specified range 9000-9005
    valid_ports = {9000, 9001, 9002, 9003, 9004, 9005}
    for internal_port, external_port in result:
        # Expect internal and external ports to be identical (no NAT mapping)
        assert internal_port == external_port
        # Expect selected ports to be from the specified range
        assert internal_port in valid_ports


def test_get_available_port_maps_default_range(executor_service):
    """Test fallback to default port range when no mappings or range provided."""
    # Arrange
    from datura.requests.miner_requests import ExecutorSSHInfo

    executor_info = ExecutorSSHInfo(
        uuid="test",
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

    # Act
    result = executor_service.get_available_port_maps(executor_info, batch_size)

    # Assert
    # Expect exactly batch_size ports because we requested 5
    assert len(result) == batch_size
    for internal_port, external_port in result:
        # Expect ports from default range 20000-65535 when no range specified
        assert 20000 <= internal_port <= 65535
        # Expect internal and external ports to be identical
        assert internal_port == external_port
        # Expect SSH port 22 to be excluded
        assert internal_port != 22


# ========================================================================================
# Tests for save_to_redis method
# ========================================================================================


@pytest.mark.asyncio
async def test_save_to_redis(executor_service, mock_redis_service, sample_executor_info):
    """Test saving successful ports to Redis with deduplication."""
    # Arrange
    miner_hotkey = "test_miner_key"
    successful_ports = [(9000, 9000), (9001, 9001)]
    mock_redis_service.lrange.return_value = ["9000,9000", "9001,9001"]

    # Act
    await executor_service.save_to_redis(sample_executor_info, miner_hotkey, successful_ports)

    # Assert
    expected_key = f"available_port_maps:{miner_hotkey}:{sample_executor_info.uuid}"
    # Expect lrem to be called for each port to remove duplicates before adding
    assert mock_redis_service.lrem.call_count == 2
    # Expect lpush to be called for each port to add them to the list
    assert mock_redis_service.lpush.call_count == 2
    # Expect specific port mappings to be added in correct format
    mock_redis_service.lpush.assert_any_call(expected_key, "9000,9000")
    mock_redis_service.lpush.assert_any_call(expected_key, "9001,9001")


def test_get_available_port_maps_empty_range(executor_service):
    """Test fallback to default range when port_range is empty string."""
    # Arrange
    from datura.requests.miner_requests import ExecutorSSHInfo

    executor_info = ExecutorSSHInfo(
        uuid="test",
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

    # Act
    result = executor_service.get_available_port_maps(executor_info, batch_size)

    # Assert
    # Expect exactly batch_size ports
    assert len(result) == batch_size
    for internal_port, external_port in result:
        # Expect default range 20000-65535 when port_range is empty string
        assert 20000 <= internal_port <= 65535


# ========================================================================================
# Tests for verify_ports method
# ========================================================================================


@pytest.mark.asyncio
async def test_verify_ports_invalid_json_mappings(executor_service, mock_ssh_client):
    """Test that verify_ports fails when port_mappings contains invalid JSON."""
    # Arrange
    from datura.requests.miner_requests import ExecutorSSHInfo

    executor_info = ExecutorSSHInfo(
        uuid="test",
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


def test_get_available_port_maps_preferred_ports_priority(executor_service):
    """Test that preferred ports are prioritized when available in port_range."""
    # Arrange
    from datura.requests.miner_requests import ExecutorSSHInfo

    # Create port range that includes some preferred ports (20000-20009 are preferred)
    executor_info = ExecutorSSHInfo(
        uuid="test",
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

    # Act
    result = executor_service.get_available_port_maps(executor_info, batch_size)

    # Assert
    # Expect exactly batch_size ports
    assert len(result) == batch_size

    selected_ports = [port_pair[0] for port_pair in result]
    preferred_in_range = [port for port in PREFERRED_POD_PORTS if (20000 <= port <= 20090)]
    preferred_selected = [port for port in selected_ports if port in PREFERRED_POD_PORTS]

    # Expect at least some preferred ports to be selected
    assert len(preferred_selected) > 0

    # Expect all preferred ports within range to be included (they should be prioritized)
    for preferred_port in preferred_in_range:
        assert preferred_port in selected_ports


def test_get_available_port_maps_preferred_mappings_priority(executor_service):
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
        uuid="test",
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

    # Act
    result = executor_service.get_available_port_maps(executor_info, batch_size)

    # Assert
    # Expect exactly batch_size ports
    assert len(result) == batch_size

    selected_ports = [port_pair[0] for port_pair in result]

    # Expect preferred ports 20000 and 20001 to be included because they are prioritized
    assert 20000 in selected_ports
    assert 20001 in selected_ports


# ========================================================================================
# Tests for cleanup_docker_containers method
# ========================================================================================


@pytest.mark.asyncio
async def test_cleanup_docker_containers(executor_service, mock_ssh_client):
    """Test cleanup of Docker containers with 'container_' prefix."""
    # Arrange
    # Mock responses: 1) list containers command, 2) rm command, 3) prune command
    mock_ssh_client.run.side_effect = [
        AsyncMock(stdout="container_test1\ncontainer_test2\n", exit_status=0),
        AsyncMock(stdout="", exit_status=0),  # docker rm response
        AsyncMock(stdout="", exit_status=0),  # docker volume prune response
    ]

    # Act
    await executor_service.cleanup_docker_containers(mock_ssh_client)

    # Assert
    # Expect 3 SSH commands: list, rm, prune
    assert mock_ssh_client.run.call_count == 3

    all_calls = [call.args[0] for call in mock_ssh_client.run.call_args_list]

    # Expect first call to list containers with name filter
    assert "docker ps" in all_calls[0] and "container_" in all_calls[0]
    # Expect second call to remove found containers
    assert "docker rm" in all_calls[1] and "container_test1" in all_calls[1]
    # Expect third call to prune volumes
    assert "docker volume prune" in all_calls[2]


# ========================================================================================
# Tests for verify_port_dind method
# ========================================================================================


@pytest.mark.asyncio
async def test_verify_port_dind_successful_connection(executor_service, mock_ssh_client, sample_executor_info):
    """Test successful Docker-in-Docker verification with SSH connection."""
    # Arrange
    from unittest.mock import patch, MagicMock

    miner_hotkey = "test_miner"
    internal_port = 9000
    external_port = 9000
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

    with patch('asyncssh.import_private_key', return_value=mock_pkey) as mock_import_key, \
         patch('asyncssh.connect') as mock_connect:

        # Setup asyncssh.connect as async context manager
        mock_connect.return_value.__aenter__.return_value = mock_container_ssh
        mock_connect.return_value.__aexit__.return_value = AsyncMock()

        # Act
        result = await executor_service.verify_port_dind(
            mock_ssh_client,
            miner_hotkey,
            sample_executor_info,
            private_key,
            public_key,
            internal_port,
            external_port,
            sysbox_runtime=False,
        )

        # Assert
        # Expect success because docker container created and SSH connected
        assert result.success is True
        # Expect success message with port number
        assert "dind: check ok" in result.log_text
        assert str(internal_port) in result.log_text

        # Expect docker run command was called with correct parameters
        docker_run_call = mock_ssh_client.run.call_args_list[0][0][0]
        assert "/usr/bin/docker run" in docker_run_call
        assert f"container_{miner_hotkey}_{external_port}" in docker_run_call
        assert f"-p {internal_port}:22" in docker_run_call

        # Expect SSH private key was imported
        mock_import_key.assert_called_once_with(private_key)

        # Expect SSH connection to container was established
        mock_connect.assert_called_once()
        connect_kwargs = mock_connect.call_args[1]
        assert connect_kwargs['host'] == sample_executor_info.address
        assert connect_kwargs['port'] == external_port
        assert connect_kwargs['username'] == 'root'

        # Expect container cleanup was called
        cleanup_call = mock_ssh_client.run.call_args_list[1][0][0]
        assert "docker rm" in cleanup_call
        assert f"container_{miner_hotkey}_{external_port}" in cleanup_call


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

    # Mock all methods in the verification flow
    port_maps = [(9000, 9000), (9001, 9001), (9002, 9002)]
    successful_bulk_ports = [(9001, 9001), (9002, 9002)]
    failed_bulk_ports = []
    dind_port = (9000, 9000)

    with patch.object(executor_service, 'cleanup_docker_containers', new=AsyncMock()) as mock_cleanup, \
         patch.object(executor_service, 'get_available_port_maps', return_value=port_maps) as mock_get_ports, \
         patch.object(executor_service, 'verify_ports_bulk', new=AsyncMock(return_value=(successful_bulk_ports, failed_bulk_ports))) as mock_bulk, \
         patch.object(executor_service, 'verify_port_dind', new=AsyncMock(return_value=DockerConnectionCheckResult(success=True, log_text="dind ok", sysbox_runtime=False))) as mock_dind, \
         patch.object(executor_service, 'save_to_redis', new=AsyncMock()) as mock_save_redis, \
         patch.object(executor_service, 'save_to_db', new=AsyncMock()) as mock_save_db:

        # Act
        result = await executor_service.verify_ports(
            mock_ssh_client,
            job_batch_id,
            miner_hotkey,
            sample_executor_info,
            private_key,
            public_key,
        )

        # Assert
        # Expect success because all steps succeeded
        assert result.success is True
        # Expect log_text contains success summary with verification stats
        assert "verification complete" in result.log_text
        assert "available" in result.log_text

        # Expect cleanup was called first with ssh_client and extra dict
        mock_cleanup.assert_called_once()
        cleanup_args = mock_cleanup.call_args
        assert cleanup_args[0][0] == mock_ssh_client
        # Expect extra dict contains job metadata
        assert "job_batch_id" in cleanup_args[0][1]
        assert "miner_hotkey" in cleanup_args[0][1]

        # Expect get_available_port_maps was called with correct batch size
        from services.const import BATCH_PORT_VERIFICATION_SIZE
        mock_get_ports.assert_called_once_with(sample_executor_info, BATCH_PORT_VERIFICATION_SIZE)

        # Expect verify_ports_bulk was called
        mock_bulk.assert_called_once()

        # Expect verify_port_dind was called with first successful port from bulk
        mock_dind.assert_called_once()
        dind_call_args = mock_dind.call_args[0]
        assert dind_call_args[0] == mock_ssh_client
        assert dind_call_args[1] == miner_hotkey
        assert dind_call_args[2] == sample_executor_info
        # Expect dind was called with first port from successful_bulk_ports (9001, 9001)
        assert dind_call_args[5] == 9001  # internal_port
        assert dind_call_args[6] == 9001  # external_port

        # Expect save_to_redis was called with successful ports
        mock_save_redis.assert_called_once()
        redis_successful_ports = mock_save_redis.call_args[0][2]
        # Expect 2 ports: dind port (9001) is popped from bulk, verified, then added back
        assert len(redis_successful_ports) == 2

        # Expect save_to_db was called with successful and failed ports
        mock_save_db.assert_called_once()
        db_successful_ports = mock_save_db.call_args[0][2]
        db_failed_ports = mock_save_db.call_args[0][3]
        # Expect 2 successful ports (dind port re-added after verification)
        assert len(db_successful_ports) == 2
        # Expect 0 failed ports because all verifications succeeded
        assert len(db_failed_ports) == 0
