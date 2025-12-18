"""Mock fixtures for validators tests."""

from unittest.mock import AsyncMock, MagicMock, patch

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
    port_mappings = [[9000 + i, 9000 + i] for i in range(1005)]
    return ExecutorSSHInfo(
        uuid="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
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

        response = AsyncMock()
        response.status = 200
        response.json = AsyncMock(return_value={"status": "ok"})
        session.get.return_value.__aenter__.return_value = response
        session.post.return_value.__aenter__.return_value = response

        yield session


@pytest.fixture
def mock_async_session_maker(test_db_session):
    """Mock the global AsyncSessionMaker to use test database."""

    class MockContextManager:
        def __init__(self, session):
            self._session = session

        async def __aenter__(self):
            return self._session

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            if exc_type:
                await self._session.rollback()
            return False  # Propagate exceptions

    with patch('daos.base.AsyncSessionMaker') as mock_maker:
        mock_maker.return_value = MockContextManager(test_db_session)
        yield mock_maker


@pytest.fixture
def port_mapping_dao(mock_async_session_maker):
    """Create PortMappingDao for testing with test database."""
    from daos.port_mapping_dao import PortMappingDao
    return PortMappingDao()
