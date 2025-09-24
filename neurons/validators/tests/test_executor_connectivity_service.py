from unittest.mock import AsyncMock

import pytest

from services.executor_connectivity_service import (
    ExecutorConnectivityService,
)


@pytest.fixture
def executor_service(mock_redis_service):
    """Create ExecutorConnectivityService for testing."""
    return ExecutorConnectivityService(mock_redis_service)


def test_get_available_port_maps_from_mappings(executor_service, sample_executor_info):
    """Test port extraction from JSON port_mappings."""
    batch_size = 2

    result = executor_service.get_available_port_maps(sample_executor_info, batch_size)

    # Implementation currently returns 1000 ports (hardcoded in line 349)
    assert len(result) == 2
    # All ports should be in the range 9000-10004, excluding SSH port 22
    for internal_port, external_port in result:
        assert 9000 <= internal_port <= 10004
        assert internal_port == external_port
        assert internal_port != 22


def test_get_available_port_maps_from_range(executor_service):
    """Test port generation from port_range string."""
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

    result = executor_service.get_available_port_maps(executor_info, batch_size)

    assert len(result) == batch_size
    valid_ports = {9000, 9001, 9002, 9003, 9004, 9005}
    for internal_port, external_port in result:
        assert internal_port == external_port
        assert internal_port in valid_ports


def test_get_available_port_maps_default_range(executor_service):
    """Test fallback to default port range when no mappings or range provided."""
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

    result = executor_service.get_available_port_maps(executor_info, batch_size)

    assert len(result) == batch_size
    for internal_port, external_port in result:
        assert 40000 <= internal_port <= 65535
        assert internal_port == external_port
        assert internal_port != 22


@pytest.mark.asyncio
async def test_save_to_redis(executor_service, mock_redis_service, sample_executor_info):
    """Test saving successful ports to Redis."""
    miner_hotkey = "test_miner_key"
    successful_ports = [(9000, 9000), (9001, 9001)]
    mock_redis_service.lrange.return_value = ["9000,9000", "9001,9001"]

    await executor_service.save_to_redis(sample_executor_info, miner_hotkey, successful_ports)

    expected_key = f"available_port_maps:{miner_hotkey}:{sample_executor_info.uuid}"
    assert mock_redis_service.lrem.call_count == 2
    assert mock_redis_service.lpush.call_count == 2
    mock_redis_service.lpush.assert_any_call(expected_key, "9000,9000")
    mock_redis_service.lpush.assert_any_call(expected_key, "9001,9001")


def test_get_available_port_maps_empty_range(executor_service):
    """Test handling of empty port range."""
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

    result = executor_service.get_available_port_maps(executor_info, batch_size)

    assert len(result) == batch_size
    for internal_port, external_port in result:
        assert 40000 <= internal_port <= 65535


@pytest.mark.asyncio
async def test_batch_verify_ports_invalid_json_mappings(executor_service, mock_ssh_client):
    """Test handling of invalid JSON in port_mappings."""
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

    result = await executor_service.batch_verify_ports(
        mock_ssh_client, "job_123", "miner_key", executor_info, "private_key", "public_key"
    )

    assert result.success is False
    assert "No port available" in result.log_text or "Expecting value" in result.log_text


@pytest.mark.asyncio
async def test_cleanup_docker_containers(executor_service, mock_ssh_client):
    """Test cleanup of old Docker containers."""
    mock_ssh_client.run.side_effect = [
        AsyncMock(stdout="container_test1\ncontainer_test2\n"),
        AsyncMock(stdout="container_batch_verifier_9000\n"),
        AsyncMock(exit_status=0),
        AsyncMock(exit_status=0),
    ]

    await executor_service.cleanup_docker_containers(mock_ssh_client)

    assert mock_ssh_client.run.call_count == 3
    cleanup_calls = [call.args[0] for call in mock_ssh_client.run.call_args_list[-2:]]
    assert any("docker rm" in cmd for cmd in cleanup_calls)
    assert any("docker volume prune" in cmd for cmd in cleanup_calls)
