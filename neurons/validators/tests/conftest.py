import logging
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from datura.requests.miner_requests import ExecutorSSHInfo
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

# Settings is instantiated during test collection through app imports. These
# defaults keep unit tests self-contained without requiring a local validator env.
os.environ.setdefault("BITTENSOR_WALLET_NAME", "test-wallet")
os.environ.setdefault("BITTENSOR_WALLET_HOTKEY_NAME", "test-hotkey")
os.environ.setdefault("SQLALCHEMY_DATABASE_URI", "sqlite:///:memory:")
os.environ.setdefault("ASYNC_SQLALCHEMY_DATABASE_URI", "sqlite+aiosqlite:///:memory:")

# Prevent network calls during module-level SharedConfigClient instantiation.
# core/config.py creates `shared_client = SharedConfigClient(...)` at import time, which
# otherwise performs a blocking HTTP fetch against the live API. Forcing _fetch to return
# None keeps the offline fallback (DEFAULT_SHARED_CONFIG) so unit tests stay hermetic.
# This patch must run before test collection.
_shared_config_patcher = patch(
    "lium_core.shared_config.client.SharedConfigClient._fetch",
    return_value=None,
)
_shared_config_patcher.start()

from helpers import make_context

# Prevent wallet KeyFileError during module-level PriceProvider() instantiation in rental_price.py.
# rental_price.py creates PriceProvider() as a class variable at import time; after the
# price_provider.py update, PriceProvider.__init__ calls SubtensorClient.get_instance() which
# reads the wallet keypair file. This patch must run before test collection.
_subtensor_patcher = patch(
    "clients.subtensor_client.SubtensorClient.get_instance",
    return_value=MagicMock(),
)
_subtensor_patcher.start()


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--update-snapshot",
        action="store_true",
        default=False,
        help="Update expected_output in snapshot JSON instead of asserting",
    )


@pytest.fixture
def update_snapshot(request: pytest.FixtureRequest) -> bool:
    return request.config.getoption("--update-snapshot")


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
def context_factory():
    def _factory(**overrides):
        return make_context(**overrides)

    return _factory


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
        echo=True,
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

    with patch("daos.base.AsyncSessionMaker") as mock_maker:
        mock_maker.return_value = MockContextManager(test_db_session)
        yield mock_maker


@pytest.fixture
def port_mapping_dao(mock_async_session_maker):
    """Create PortMappingDao for testing with test database."""
    from daos.port_mapping_dao import PortMappingDao

    return PortMappingDao()

