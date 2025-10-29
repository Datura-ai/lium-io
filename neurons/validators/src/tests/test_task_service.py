from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from services.task_service import TaskService


@pytest.fixture
def mock_port_mapping_dao():
    """Mock PortMappingDao for testing."""
    dao = AsyncMock()
    dao.get_successful_ports_count = AsyncMock()
    return dao


@pytest.fixture
def task_service(mock_redis_service, mock_port_mapping_dao):
    """Create TaskService with mocked dependencies."""
    service = TaskService(
        ssh_service=Mock(),
        redis_service=mock_redis_service,
        validation_service=Mock(),
        verifyx_validation_service=Mock(),
        collateral_contract_service=Mock(),
        executor_connectivity_service=Mock(),
        attestation_service=Mock(),
        port_mapping_dao=mock_port_mapping_dao,
    )
    return service


@pytest.mark.asyncio
async def test_get_available_port_count_returns_from_db_when_enough_ports(
    task_service, mock_port_mapping_dao, mock_redis_service
):
    """Test that function returns count from DB when ports >= 3, without Redis fallback."""
    # Arrange
    miner_hotkey = "test_miner_hotkey"
    executor_id = str(uuid4())
    db_port_count = 5

    mock_port_mapping_dao.get_successful_ports_count.return_value = db_port_count

    # Act
    result = await task_service.get_available_port_count(miner_hotkey, executor_id)

    # Assert
    assert result == db_port_count
    mock_port_mapping_dao.get_successful_ports_count.assert_called_once_with(executor_id)
    mock_redis_service.lrange.assert_not_called()


@pytest.mark.asyncio
async def test_get_available_port_count_falls_back_to_redis_when_few_ports(
    task_service, mock_port_mapping_dao, mock_redis_service
):
    """Test that function falls back to Redis when DB has < 3 ports."""
    # Arrange
    miner_hotkey = "test_miner_hotkey"
    executor_id = str(uuid4())
    db_port_count = 2  # Less than MIN_PORTS (3)
    redis_ports = [b"9000,9000", b"9001,9001", b"9002,9002", b"9003,9003"]

    mock_port_mapping_dao.get_successful_ports_count.return_value = db_port_count
    mock_redis_service.lrange.return_value = redis_ports

    # Act
    result = await task_service.get_available_port_count(miner_hotkey, executor_id)

    # Assert
    assert result == len(redis_ports)
    mock_port_mapping_dao.get_successful_ports_count.assert_called_once_with(executor_id)
    expected_redis_key = f"available_port_maps:{miner_hotkey}:{executor_id}"
    mock_redis_service.lrange.assert_called_once_with(expected_redis_key)


@pytest.mark.asyncio
async def test_get_available_port_count_falls_back_to_redis_on_db_error(
    task_service, mock_port_mapping_dao, mock_redis_service
):
    """Test that function falls back to Redis when DB raises an exception."""
    # Arrange
    miner_hotkey = "test_miner_hotkey"
    executor_id = str(uuid4())
    redis_ports = [b"9000,9000", b"9001,9001", b"9002,9002"]

    mock_port_mapping_dao.get_successful_ports_count.side_effect = Exception(
        "Database connection error"
    )
    mock_redis_service.lrange.return_value = redis_ports

    # Act
    result = await task_service.get_available_port_count(miner_hotkey, executor_id)

    # Assert
    assert result == len(redis_ports)
    mock_port_mapping_dao.get_successful_ports_count.assert_called_once_with(executor_id)
    expected_redis_key = f"available_port_maps:{miner_hotkey}:{executor_id}"
    mock_redis_service.lrange.assert_called_once_with(expected_redis_key)
