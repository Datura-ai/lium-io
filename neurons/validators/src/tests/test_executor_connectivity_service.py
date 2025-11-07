import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from datura.requests.miner_requests import ExecutorSSHInfo
from services.executor_connectivity_service import (
    ExecutorConnectivityService,
    DockerConnectionCheckResult,
    MAX_REDIS_KEEP,
)
from services.const import PREFERRED_POD_PORTS, BATCH_PORT_VERIFICATION_SIZE


@pytest.fixture
def executor_service(mock_redis_service, port_mapping_dao):
    """Create ExecutorConnectivityService for testing."""
    return ExecutorConnectivityService(mock_redis_service, port_mapping_dao)


def test_get_available_port_maps_from_mappings(executor_service, sample_executor_info):
    """Test port extraction from JSON port_mappings."""
    batch_size = 2
    result = executor_service.get_available_port_maps(sample_executor_info, batch_size)

    assert len(result) == 2
    for internal_port, external_port in result:
        assert 9000 <= internal_port <= 10004
        assert internal_port == external_port
        assert internal_port != 22


def test_get_available_port_maps_from_range(executor_service, executor_with_port_range):
    """Test port generation from port_range string."""
    batch_size = 3
    result = executor_service.get_available_port_maps(executor_with_port_range, batch_size)

    assert len(result) == batch_size
    valid_ports = {9000, 9001, 9002, 9003, 9004, 9005}
    for internal_port, external_port in result:
        assert internal_port == external_port
        assert internal_port in valid_ports


def test_get_available_port_maps_default_range(executor_service, executor_default):
    """Test fallback to default port range when no mappings or range provided."""
    batch_size = 5
    result = executor_service.get_available_port_maps(executor_default, batch_size)

    assert len(result) == batch_size
    for internal_port, external_port in result:
        assert 20000 <= internal_port <= 65535
        assert internal_port == external_port
        assert internal_port != 22


def test_get_available_port_maps_empty_range(executor_service):
    """Test fallback to default range when port_range is empty string."""
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
    result = executor_service.get_available_port_maps(executor_info, batch_size)

    assert len(result) == batch_size
    for internal_port, external_port in result:
        assert 20000 <= internal_port <= 65535


def test_get_available_port_maps_preferred_ports_priority(executor_service):
    """Test that preferred ports are prioritized when available in port_range."""
    executor_info = ExecutorSSHInfo(
        uuid="test",
        address="127.0.0.1",
        port=8080,
        ssh_username="root",
        ssh_port=22,
        port_mappings=None,
        port_range="20000-30090",
        python_path="/usr/bin/python3",
        root_dir="/tmp",
    )
    batch_size = 15
    result = executor_service.get_available_port_maps(executor_info, batch_size)

    assert len(result) == batch_size
    selected_ports = [port_pair[0] for port_pair in result]
    preferred_in_range = [port for port in PREFERRED_POD_PORTS if (20000 <= port <= 20000+batch_size*4)]

    for port in PREFERRED_POD_PORTS:
        assert port in selected_ports

    for preferred_port in preferred_in_range:
        assert preferred_port in selected_ports


def test_get_available_port_maps_preferred_mappings_priority(executor_service):
    """Test that preferred port mappings are prioritized from JSON mappings."""
    port_mappings = [
        [20000, 20000],
        [20001, 20001],
        [9000, 9000],
        [9001, 9001],
        [9002, 9002],
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
    result = executor_service.get_available_port_maps(executor_info, batch_size)

    assert len(result) == batch_size
    selected_ports = [port_pair[0] for port_pair in result]
    assert 20000 in selected_ports
    assert 20001 in selected_ports


@pytest.mark.asyncio
async def test_save_to_redis(executor_service, mock_redis_service, sample_executor_info):
    """Test saving successful ports to Redis with deduplication."""
    miner_hotkey = "test_miner_key"
    successful_ports = [(9000, 9000), (9001, 9001)]
    mock_redis_service.lrange.return_value = ["9000,9000", "9001,9001"]

    await executor_service.save_to_redis(sample_executor_info, miner_hotkey, successful_ports)

    expected_key = f"available_port_maps:{miner_hotkey}:{sample_executor_info.uuid}"
    assert mock_redis_service.lrem.call_count == 2
    assert mock_redis_service.lpush.call_count == 2
    mock_redis_service.lpush.assert_any_call(expected_key, "9000,9000")
    mock_redis_service.lpush.assert_any_call(expected_key, "9001,9001")


@pytest.mark.asyncio
async def test_verify_ports_invalid_json_mappings(executor_service, mock_ssh_client):
    """Test that verify_ports fails when port_mappings contains invalid JSON."""
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
    mock_ssh_client.run.return_value.stdout = ""

    result = await executor_service.verify_ports(
        mock_ssh_client, "job_123", "miner_key", executor_info, "private_key", "public_key"
    )

    assert result.success is False
    assert "Expecting value" in result.log_text or "Verification failed" in result.log_text


@pytest.mark.asyncio
async def test_cleanup_docker_containers(executor_service, mock_ssh_client):
    """Test cleanup of Docker containers with 'container_' prefix."""
    mock_ssh_client.run.side_effect = [
        AsyncMock(stdout="container_test1\ncontainer_test2\nexecutor-executor-1", exit_status=0),
        AsyncMock(stdout="", exit_status=0),
        AsyncMock(stdout="", exit_status=0),
    ]

    await executor_service.cleanup_docker_containers(mock_ssh_client)

    assert mock_ssh_client.run.call_count == 3
    all_calls = [call.args[0] for call in mock_ssh_client.run.call_args_list]

    assert "docker ps" in all_calls[0]
    assert "docker rm" in all_calls[1] and "container_test1" in all_calls[1] and "executor-executor-1" not in all_calls[1]
    assert "docker volume prune" in all_calls[2]


@pytest.mark.asyncio
async def test_verify_port_dind_successful_connection(executor_service, mock_ssh_client, sample_executor_info):
    """Test successful Docker-in-Docker verification with SSH connection."""
    miner_hotkey = "test_miner"
    internal_port = 9000
    external_port = 9000
    private_key = "-----BEGIN PRIVATE KEY-----\ntest_key\n-----END PRIVATE KEY-----"
    public_key = "ssh-rsa test_public_key"

    mock_ssh_client.run.side_effect = [
        AsyncMock(exit_status=0, stdout="container_id", stderr=""),
        AsyncMock(exit_status=0, stdout="", stderr=""),
    ]

    mock_pkey = MagicMock()
    mock_container_ssh = AsyncMock()

    with patch('asyncssh.import_private_key', return_value=mock_pkey) as mock_import_key, \
         patch('asyncssh.connect') as mock_connect:

        mock_connect.return_value.__aenter__.return_value = mock_container_ssh
        mock_connect.return_value.__aexit__.return_value = AsyncMock()

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

        assert result.success is True
        assert "dind: check ok" in result.log_text
        assert str(internal_port) in result.log_text

        docker_run_call = mock_ssh_client.run.call_args_list[0][0][0]
        assert "/usr/bin/docker run" in docker_run_call
        assert f"container_{miner_hotkey}_{external_port}" in docker_run_call
        assert f"-p {internal_port}:22" in docker_run_call

        mock_import_key.assert_called_once_with(private_key)
        mock_connect.assert_called_once()
        connect_kwargs = mock_connect.call_args[1]
        assert connect_kwargs['host'] == sample_executor_info.address
        assert connect_kwargs['port'] == external_port
        assert connect_kwargs['username'] == 'root'

        cleanup_call = mock_ssh_client.run.call_args_list[1][0][0]
        assert "docker rm" in cleanup_call
        assert f"container_{miner_hotkey}_{external_port}" in cleanup_call


@pytest.mark.asyncio
async def test_verify_ports_successful_flow(executor_service, mock_ssh_client, sample_executor_info):
    """Test complete successful verification flow with all components."""
    job_batch_id = "job_123"
    miner_hotkey = "test_miner"
    private_key = "test_private_key"
    public_key = "test_public_key"

    port_maps = [(9000, 9000), (9001, 9001), (9002, 9002)]
    successful_bulk_ports = [(9001, 9001), (9002, 9002)]
    failed_bulk_ports = []

    with patch.object(executor_service, 'cleanup_docker_containers', new=AsyncMock()) as mock_cleanup, \
         patch.object(executor_service, 'get_available_port_maps', return_value=port_maps) as mock_get_ports, \
         patch.object(executor_service, 'verify_ports_bulk', new=AsyncMock(return_value=(successful_bulk_ports, failed_bulk_ports))) as mock_bulk, \
         patch.object(executor_service, 'verify_port_dind', new=AsyncMock(return_value=DockerConnectionCheckResult(success=True, log_text="dind ok", sysbox_runtime=False))) as mock_dind, \
         patch.object(executor_service, 'save_to_redis', new=AsyncMock()) as mock_save_redis, \
         patch.object(executor_service, 'save_to_db', new=AsyncMock()) as mock_save_db:

        result = await executor_service.verify_ports(
            mock_ssh_client,
            job_batch_id,
            miner_hotkey,
            sample_executor_info,
            private_key,
            public_key,
        )

        assert result.success is True
        assert "verification complete" in result.log_text
        assert "available" in result.log_text

        mock_cleanup.assert_called_once()
        cleanup_args = mock_cleanup.call_args
        assert cleanup_args[0][0] == mock_ssh_client
        assert "job_batch_id" in cleanup_args[0][1]
        assert "miner_hotkey" in cleanup_args[0][1]

        mock_get_ports.assert_called_once_with(sample_executor_info, BATCH_PORT_VERIFICATION_SIZE)
        mock_bulk.assert_called_once()

        mock_dind.assert_called_once()
        dind_call_args = mock_dind.call_args[0]
        assert dind_call_args[0] == mock_ssh_client
        assert dind_call_args[1] == miner_hotkey
        assert dind_call_args[2] == sample_executor_info
        assert dind_call_args[5] == 9001
        assert dind_call_args[6] == 9001

        mock_save_redis.assert_called_once()
        redis_successful_ports = mock_save_redis.call_args[0][2]
        assert len(redis_successful_ports) == 2

        mock_save_db.assert_called_once()
        db_successful_ports = mock_save_db.call_args[0][2]
        db_failed_ports = mock_save_db.call_args[0][3]
        assert len(db_successful_ports) == 2
        assert len(db_failed_ports) == 0


@pytest.mark.asyncio
async def test_verify_ports_uses_top10_ports_when_bulk_fails(executor_service, mock_ssh_client, executor_with_large_range):
    """Test that when bulk verification fails, random port is selected from top 10 ports by external_port."""
    job_batch_id = "job_123"
    miner_hotkey = "test_miner"
    private_key = "test_private_key"
    public_key = "test_public_key"

    port_maps = executor_service.get_available_port_maps(executor_with_large_range, BATCH_PORT_VERIFICATION_SIZE)
    expected_shorter_port_maps = sorted(port_maps, key=lambda m: m[1], reverse=True)[:MAX_REDIS_KEEP]

    successful_bulk_ports = []
    failed_bulk_ports = port_maps

    with patch.object(executor_service, 'cleanup_docker_containers', new=AsyncMock()), \
         patch.object(executor_service, 'verify_ports_bulk', new=AsyncMock(return_value=(successful_bulk_ports, failed_bulk_ports))), \
         patch.object(executor_service, 'verify_port_dind', new=AsyncMock(return_value=DockerConnectionCheckResult(success=True, log_text="dind ok", sysbox_runtime=False))), \
         patch.object(executor_service, 'save_to_redis', new=AsyncMock()), \
         patch.object(executor_service, 'save_to_db', new=AsyncMock()), \
         patch('services.executor_connectivity_service.random.choice') as mock_random_choice:

        mock_random_choice.return_value = expected_shorter_port_maps[0]

        result = await executor_service.verify_ports(
            mock_ssh_client,
            job_batch_id,
            miner_hotkey,
            executor_with_large_range,
            private_key,
            public_key,
        )

        mock_random_choice.assert_called_once()
        actual_arg = mock_random_choice.call_args[0][0]

        assert len(actual_arg) == MAX_REDIS_KEEP
        assert actual_arg == expected_shorter_port_maps

        external_ports = [port_pair[1] for port_pair in actual_arg]
        assert external_ports == sorted(external_ports, reverse=True)
