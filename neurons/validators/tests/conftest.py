from unittest.mock import AsyncMock, MagicMock

import pytest
from datura.requests.miner_requests import ExecutorSSHInfo


@pytest.fixture
def mock_ssh_client():
    """Mock SSH client for testing Docker operations."""
    client = AsyncMock()
    client.run = AsyncMock(return_value=MagicMock(exit_status=0, stdout="", stderr=""))
    return client


@pytest.fixture
def mock_redis_service():
    """Mock Redis service for testing port storage operations."""
    service = AsyncMock()
    service.lpush = AsyncMock()
    service.lrem = AsyncMock()
    service.lrange = AsyncMock(return_value=[])
    service.rpop = AsyncMock()
    return service


@pytest.fixture
def sample_executor_info():
    """Sample ExecutorSSHInfo for testing."""
    # Create enough ports to avoid the hardcoded 1000 limit in line 349
    port_mappings = [[9000 + i, 9000 + i] for i in range(1005)]
    return ExecutorSSHInfo(
        uuid="test-executor-123",
        address="192.168.1.100",
        port=8080,
        ssh_username="root",
        ssh_port=22,
        port_mappings=str(port_mappings),
        port_range="40000-50000",
        python_path="/usr/bin/python3",
        root_dir="/tmp",
    )


@pytest.fixture
def mock_aiohttp_session():
    """Mock aiohttp session for testing HTTP requests."""
    with patch("aiohttp.ClientSession") as mock_session_class:
        session = AsyncMock()
        mock_session_class.return_value.__aenter__.return_value = session

        # Configure default response
        response = AsyncMock()
        response.status = 200
        response.json = AsyncMock(return_value={"status": "ok"})
        session.get.return_value.__aenter__.return_value = response
        session.post.return_value.__aenter__.return_value = response

        yield session
