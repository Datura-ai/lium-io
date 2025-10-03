from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from datura.requests.miner_requests import ExecutorSSHInfo
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession
import logging
import sys


@pytest.fixture(scope="session", autouse=True)
def setup_sql_logging():
    """Enable SQL query logging for all tests."""
    # Configure root logging to show INFO level
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)

    # Enable SQLAlchemy engine logging - just set level, let basicConfig handle output
    sql_logger = logging.getLogger('sqlalchemy.engine')
    sql_logger.setLevel(logging.INFO)

    # Also enable pool logging for connection monitoring
    pool_logger = logging.getLogger('sqlalchemy.pool')
    pool_logger.setLevel(logging.DEBUG)

    print("✅ SQL logging enabled for all tests")


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


@pytest_asyncio.fixture(scope="session")
async def test_engine():
    """Create an async engine for testing with SQLite in-memory database."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=True,  # Enable SQL query logging
        future=True,
    )

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    yield engine

    await engine.dispose()


@pytest_asyncio.fixture
async def test_db_session(test_engine):
    """Create a test database session for each test."""
    from sqlalchemy.orm import sessionmaker

    async_session_maker = sessionmaker(
        bind=test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with async_session_maker() as session:
        yield session
        await session.rollback()


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

    with patch('daos.base.AsyncSessionMaker') as mock_maker:
        mock_maker.return_value = MockContextManager(test_db_session)
        yield mock_maker


@pytest.fixture
def port_mapping_dao(mock_async_session_maker):
    """Create PortMappingDao for testing with test database."""
    from daos.port_mapping_dao import PortMappingDao
    return PortMappingDao()
