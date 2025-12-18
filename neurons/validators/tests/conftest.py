"""Pytest configuration and fixtures for validators tests.

This module provides shared fixtures for all test modules.
Fixtures are organized into separate modules for better maintainability:
- fixtures/db_fixtures.py: Database-related fixtures
- fixtures/mock_fixtures.py: Mock objects and services
- helpers/context.py: Context builders for pipeline tests
- factories/port_mapping.py: Factory functions for test data
"""

import logging
import sys
from pathlib import Path

# Add validators directory to path FIRST (before lium-io root which also has tests/)
VALIDATORS_ROOT = Path(__file__).resolve().parents[1]
# Remove lium-io root if present, add validators paths at the beginning
lium_root = str(VALIDATORS_ROOT.parents[1])
if lium_root in sys.path:
    sys.path.remove(lium_root)
sys.path.insert(0, str(VALIDATORS_ROOT))
sys.path.insert(1, str(VALIDATORS_ROOT / "src"))
if lium_root not in sys.path:
    sys.path.append(lium_root)

import pytest

# Import fixtures from organized modules
from tests.fixtures.db_fixtures import test_db_session, test_engine  # noqa: E402
from tests.fixtures.mock_fixtures import (  # noqa: E402
    mock_aiohttp_session,
    mock_async_session_maker,
    mock_redis_service,
    mock_ssh_client,
    port_mapping_dao,
    sample_executor_info,
)
from tests.helpers.context import make_context  # noqa: E402


# Legacy fixture for pipeline tests
@pytest.fixture
def context_factory():
    """Factory fixture for creating pipeline Context objects.

    Usage:
        def test_something(context_factory):
            ctx = context_factory(miner_hotkey="test-key")
            # ... use ctx
    """

    def _factory(**overrides):
        return make_context(**overrides)

    return _factory


@pytest.fixture(scope="session", autouse=True)
def setup_sql_logging():
    """Enable SQL query logging for all tests."""
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)

    sql_logger = logging.getLogger('sqlalchemy.engine')
    sql_logger.setLevel(logging.INFO)

    pool_logger = logging.getLogger('sqlalchemy.pool')
    pool_logger.setLevel(logging.DEBUG)

    print("SQL logging enabled for all tests")


# Re-export all fixtures for pytest discovery
__all__ = [
    "context_factory",
    "mock_aiohttp_session",
    "mock_async_session_maker",
    "mock_redis_service",
    "mock_ssh_client",
    "port_mapping_dao",
    "sample_executor_info",
    "setup_sql_logging",
    "test_db_session",
    "test_engine",
]
