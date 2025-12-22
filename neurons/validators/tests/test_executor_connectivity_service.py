from unittest.mock import AsyncMock

import pytest

from services.executor_connectivity_service import ExecutorConnectivityService


@pytest.fixture
def executor_service(mock_redis_service, port_mapping_dao, mock_backend_port_client):
    """Create ExecutorConnectivityService for testing."""
    return ExecutorConnectivityService(mock_redis_service, port_mapping_dao, mock_backend_port_client)


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


# ========================================================================================
# Tests for cleanup_docker_containers method
# ========================================================================================


@pytest.mark.asyncio
async def test_cleanup_docker_containers(executor_service, mock_ssh_client, sample_executor_info):
    """Test cleanup of Docker containers with 'container_' prefix."""
    # Arrange
    # Mock redis_service.get_rented_machine to return None
    executor_service.redis_service.get_rented_machine = AsyncMock(return_value=None)

    # Mock responses: 1) list containers command, 2) rm command, 3) prune command
    # Note: docker ps --filter "name=^/container_" only returns container_* names
    mock_ssh_client.run.side_effect = [
        AsyncMock(stdout="container_test1\ncontainer_test2", exit_status=0),
        AsyncMock(stdout="", exit_status=0),  # docker rm response
        AsyncMock(stdout="", exit_status=0),  # docker volume prune response
    ]

    # Act
    await executor_service.cleanup_docker_containers(mock_ssh_client, sample_executor_info)

    # Assert
    # Expect 3 SSH commands: list, rm, prune
    assert mock_ssh_client.run.call_count == 3

    all_calls = [call.args[0] for call in mock_ssh_client.run.call_args_list]

    # Expect first call to list containers with name filter
    assert "docker ps" in all_calls[0]
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

    with patch.object(executor_service, 'cleanup_docker_containers', new=AsyncMock()) as mock_cleanup, \
         patch('services.executor_connectivity_service.get_all_ports', return_value=port_maps) as mock_get_ports, \
         patch.object(executor_service, 'verify_ports_bulk', new=AsyncMock(return_value=(successful_bulk_ports, failed_bulk_ports))) as mock_bulk, \
         patch.object(executor_service, 'verify_port_dind', new=AsyncMock(return_value=DockerConnectionCheckResult(success=True, log_text="dind ok", sysbox_runtime=False))) as mock_dind, \
         patch.object(executor_service, 'save_to_db', new=AsyncMock()) as mock_save_db, \
         patch.object(executor_service.port_mapping_dao, 'get_busy_external_ports', new=AsyncMock(return_value=set())) as mock_get_busy_ports:

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

        # Expect cleanup was called first with ssh_client, executor_info and extra dict
        mock_cleanup.assert_called_once()
        cleanup_args = mock_cleanup.call_args
        assert cleanup_args[0][0] == mock_ssh_client
        assert cleanup_args[0][1] == sample_executor_info
        # Expect extra dict contains job metadata
        assert "job_batch_id" in cleanup_args[0][2]
        assert "miner_hotkey" in cleanup_args[0][2]

        # Expect get_all_ports was called
        mock_get_ports.assert_called_once()

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

        # Expect save_to_db was called with successful and failed ports
        mock_save_db.assert_called_once()
        db_successful_ports = mock_save_db.call_args[0][2]
        db_failed_ports = mock_save_db.call_args[0][3]
        # Expect 2 successful ports (dind port re-added after verification)
        assert len(db_successful_ports) == 2
        # Expect 0 failed ports because all verifications succeeded
        assert len(db_failed_ports) == 0
