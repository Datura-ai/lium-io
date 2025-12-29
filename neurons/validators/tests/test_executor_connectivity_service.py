from unittest.mock import AsyncMock

import pytest

from services.executor_connectivity_service import ExecutorConnectivityService


@pytest.fixture
def executor_service(mock_redis_service, port_mapping_dao):
    """Create ExecutorConnectivityService for testing."""
    return ExecutorConnectivityService(mock_redis_service, port_mapping_dao)


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

    # Act
    await executor_service.cleanup_docker_containers(mock_ssh_client, sample_executor_info, [])

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
         patch('asyncssh.connect') as mock_connect, \
         patch('asyncio.sleep', new=AsyncMock()) as mock_sleep:

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
    rented_ports = [8000, 8001]
    rented_pod_names = ["pod_1", "pod_2"]

    # Mock all methods in the verification flow
    successful_bulk_ports = [(9001, 9001), (9002, 9002)]
    failed_bulk_ports = []

    with patch.object(executor_service, 'cleanup_docker_containers', new=AsyncMock()) as mock_cleanup, \
         patch.object(executor_service, 'verify_ports_bulk', new=AsyncMock(return_value=(successful_bulk_ports, failed_bulk_ports))) as mock_bulk, \
         patch.object(executor_service, 'verify_port_dind', new=AsyncMock(return_value=DockerConnectionCheckResult(success=True, log_text="dind ok", sysbox_runtime=False))) as mock_dind, \
         patch.object(executor_service, 'save_to_db', new=AsyncMock()) as mock_save_db:

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
        # Third argument is pod_names list
        assert cleanup_args[0][2] == rented_pod_names
        # Fourth argument (extra dict) contains job metadata
        extra_dict = cleanup_args[0][3]
        assert "job_batch_id" in extra_dict
        assert "miner_hotkey" in extra_dict

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


# ========================================================================================
# Tests for verify_ports_bulk helper methods
# ========================================================================================


def test_build_netcat_script(executor_service):
    """Test netcat script generation - pure function, no mocking needed!"""
    port_maps = [(9000, 9000), (9001, 9001)]
    token = "abc123"

    script = executor_service._build_netcat_script(port_maps, token, 0)

    # Verify script contains token
    assert token in script
    # Verify script contains all ports
    assert "9000" in script
    assert "9001" in script
    # Verify script has batch structure
    assert "Batch 0" in script
    assert "nc -l -p" in script


@pytest.mark.asyncio
async def test_start_port_test_container_success(executor_service, mock_ssh_client):
    """Test container starts successfully."""
    from unittest.mock import patch

    mock_ssh_client.run.side_effect = [
        AsyncMock(exit_status=0, stdout="container_abc12345\n"),  # docker run
        AsyncMock(exit_status=0, stdout="Up 2 seconds"),  # docker ps
    ]

    with patch('asyncio.sleep', new=AsyncMock()):
        result = await executor_service._start_port_test_container(
            mock_ssh_client, "port_test_abc12345", "echo 'test'", {}
        )

    assert result is True
    assert "docker.io/library/alpine:3.19" in mock_ssh_client.run.call_args_list[0][0][0]


@pytest.mark.asyncio
async def test_start_port_test_container_fails_to_start(executor_service, mock_ssh_client):
    """Test container fails to start."""
    mock_ssh_client.run.return_value = AsyncMock(exit_status=1, stderr="Error: unable to start")

    result = await executor_service._start_port_test_container(
        mock_ssh_client, "port_test_fail", "echo 'test'", {}
    )

    assert result is False


@pytest.mark.asyncio
async def test_start_port_test_container_exits_immediately(executor_service, mock_ssh_client):
    """Test container starts but exits immediately."""
    from unittest.mock import patch

    mock_ssh_client.run.side_effect = [
        AsyncMock(exit_status=0, stdout="container_dead\n"),  # docker run
        AsyncMock(exit_status=0, stdout=""),  # docker ps (empty = not running)
        AsyncMock(exit_status=0, stdout="Error: nc failed"),  # docker logs
        AsyncMock(exit_status=0, stdout="2"),  # docker inspect exit code
    ]

    with patch('asyncio.sleep', new=AsyncMock()):
        result = await executor_service._start_port_test_container(
            mock_ssh_client, "port_test_dead", "echo 'test'", {}
        )

    assert result is False
    assert "docker logs" in mock_ssh_client.run.call_args_list[2][0][0]


@pytest.mark.asyncio
async def test_test_ports_in_batches_all_success(executor_service):
    """Test port testing with all ports successful."""
    from unittest.mock import patch

    port_maps = [(9000, 9000), (9001, 9001)]
    token = "test123"

    # Mock the HTTP session
    async def mock_test_port(session, host, int_port, ext_port, tok, extra):
        return True  # All succeed

    with patch.object(executor_service, '_test_single_port_with_session', side_effect=mock_test_port), \
         patch('asyncio.sleep', new=AsyncMock()):

        successful, failed = await executor_service._test_ports_in_batches(
            port_maps, "192.168.1.1", token, {}
        )

    assert successful == port_maps
    assert failed == []


@pytest.mark.asyncio
async def test_test_ports_in_batches_mixed_results(executor_service):
    """Test port testing with mixed success/failure."""
    from unittest.mock import patch

    port_maps = [(9000, 9000), (9001, 9001), (9002, 9002)]
    token = "test456"

    # Mock: first port succeeds, others fail
    call_count = 0
    async def mock_test_port(session, host, int_port, ext_port, tok, extra):
        nonlocal call_count
        call_count += 1
        return call_count == 1  # Only first call succeeds

    with patch.object(executor_service, '_test_single_port_with_session', side_effect=mock_test_port), \
         patch('asyncio.sleep', new=AsyncMock()):

        successful, failed = await executor_service._test_ports_in_batches(
            port_maps, "192.168.1.1", token, {}
        )

    assert successful == [(9000, 9000)]
    assert failed == [(9001, 9001), (9002, 9002)]


@pytest.mark.asyncio
async def test_cleanup_port_test_container(executor_service, mock_ssh_client):
    """Test container cleanup."""
    await executor_service._cleanup_port_test_container(mock_ssh_client, "port_test_xyz")

    assert mock_ssh_client.run.called
    assert "docker rm -f port_test_xyz" in mock_ssh_client.run.call_args[0][0]


# ========================================================================================
# Integration tests for verify_ports_bulk
# ========================================================================================


@pytest.mark.asyncio
async def test_verify_ports_bulk_integration_success(executor_service, sample_executor_info):
    """Test full verify_ports_bulk flow by mocking only the helper methods."""
    from unittest.mock import patch, AsyncMock

    port_maps = [(9000, 9000), (9001, 9001)]

    # Mock helper methods (not internals!)
    with patch.object(executor_service, '_build_netcat_script', return_value="mock_script"), \
         patch.object(executor_service, '_start_port_test_container', return_value=True), \
         patch.object(executor_service, '_test_ports_in_batches', return_value=(port_maps, [])), \
         patch.object(executor_service, '_cleanup_port_test_container', new=AsyncMock()), \
         patch('uuid.uuid4') as mock_uuid:

        mock_uuid.return_value.hex = "testtoken123"

        successful, failed = await executor_service.verify_ports_bulk(
            AsyncMock(), port_maps, sample_executor_info, {}
        )

    assert successful == port_maps
    assert failed == []


@pytest.mark.asyncio
async def test_verify_ports_bulk_integration_container_fails(executor_service, sample_executor_info):
    """Test verify_ports_bulk when container fails to start."""
    from unittest.mock import patch, AsyncMock

    port_maps = [(9000, 9000)]

    with patch.object(executor_service, '_build_netcat_script', return_value="mock_script"), \
         patch.object(executor_service, '_start_port_test_container', return_value=False), \
         patch('uuid.uuid4') as mock_uuid:

        mock_uuid.return_value.hex = "failtoken"

        successful, failed = await executor_service.verify_ports_bulk(
            AsyncMock(), port_maps, sample_executor_info, {}
        )

    assert successful == []
    assert failed == port_maps


@pytest.mark.asyncio
async def test_verify_ports_bulk_empty_ports(executor_service, sample_executor_info):
    """Test verify_ports_bulk with empty port list."""
    successful, failed = await executor_service.verify_ports_bulk(
        AsyncMock(), [], sample_executor_info, {}
    )

    assert successful == []
    assert failed == []
