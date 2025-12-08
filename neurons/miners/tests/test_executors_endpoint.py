"""Unit tests for /executors endpoint with simple signature validation."""

from unittest.mock import MagicMock
from uuid import uuid4

import bittensor
import pytest
from fastapi import status
from fastapi.testclient import TestClient


@pytest.fixture
def mock_validator_service():
    """Mock ValidatorService for testing."""
    service = MagicMock()
    service.is_valid_validator.return_value = True
    return service


@pytest.fixture
def mock_executor_service():
    """Mock ExecutorService for testing."""
    service = MagicMock()
    service.get_executors_for_validator.return_value = []
    return service


@pytest.fixture
def mock_keypair():
    """Mock bittensor Keypair for signature verification."""
    keypair = MagicMock()
    keypair.verify.return_value = True
    return keypair


@pytest.fixture
def app():
    """Get FastAPI app instance."""
    from miner import app
    return app


@pytest.fixture
def client(app, mock_validator_service, mock_executor_service, mock_keypair, monkeypatch):
    """Create test client with mocked dependencies."""
    from services.validator_service import ValidatorService
    from services.executor_service import ExecutorService

    # Override dependencies
    app.dependency_overrides[ValidatorService] = lambda: mock_validator_service
    app.dependency_overrides[ExecutorService] = lambda: mock_executor_service

    # Mock bittensor.Keypair
    monkeypatch.setattr(bittensor, "Keypair", lambda ss58_address: mock_keypair)

    yield TestClient(app)

    # Cleanup
    app.dependency_overrides.clear()


def test_get_executors_returns_empty_list_when_no_executors(client):
    """
    Test that endpoint returns 200 with empty list when validator has no executors.

    Arrange: Mock validator service to return True and executor service to return empty list
    Act: POST to /executors with valid signature
    Assert: Response is 200 with empty executors list
    """
    # Arrange
    validator_hotkey = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
    signature = "0x1234567890abcdef"

    # Act
    response = client.post(
        "/executors",
        json={
            "signature": signature,
            "validator_hotkey": validator_hotkey
        }
    )

    # Assert
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["validator_hotkey"] == validator_hotkey
    assert data["executors"] == []


def test_get_executors_returns_executor_list(client, mock_executor_service):
    """
    Test that endpoint returns 200 with executor list when validators has executors.

    Arrange: Mock services to return list of executors
    Act: POST to /executors with valid signature
    Assert: Response is 200 with list of executors
    """
    # Arrange
    validator_hotkey = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
    signature = "0x1234567890abcdef"

    # Mock executor
    executor_uuid = uuid4()
    mock_executor = MagicMock()
    mock_executor.uuid = executor_uuid
    mock_executor.address = "192.168.1.100"
    mock_executor.port = 8080

    # Update mock to return executor list
    mock_executor_service.get_executors_for_validator.return_value = [mock_executor]

    # Act
    response = client.post(
        "/executors",
        json={
            "signature": signature,
            "validator_hotkey": validator_hotkey
        }
    )

    # Assert
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["validator_hotkey"] == validator_hotkey
    assert len(data["executors"]) == 1
    assert data["executors"][0]["uuid"] == str(executor_uuid)
    assert data["executors"][0]["address"] == "192.168.1.100"
    assert data["executors"][0]["port"] == 8080


def test_get_executors_invalid_signature(client, mock_keypair):
    """
    Test that endpoint returns 401 when signature is invalid.

    Arrange: Mock keypair.verify to return False
    Act: POST to /executors with invalid signature
    Assert: Response is 401 Unauthorized
    """
    # Arrange
    validator_hotkey = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
    signature = "0xinvalid_signature"

    # Mock signature verification to return False
    mock_keypair.verify.return_value = False

    # Act
    response = client.post(
        "/executors",
        json={
            "signature": signature,
            "validator_hotkey": validator_hotkey
        }
    )

    # Assert
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert "signature" in response.json()["detail"].lower()


def test_get_executors_unregistered_validator(client, mock_validator_service):
    """
    Test that endpoint returns 403 when validator is not registered.

    Arrange: Mock validator service to return False for is_valid_validator
    Act: POST to /executors with unregistered validator
    Assert: Response is 403 Forbidden
    """
    # Arrange
    validator_hotkey = "5UnregisteredValidatorHotkey"
    signature = "0x1234567890abcdef"

    # Mock validator service to return False
    mock_validator_service.is_valid_validator.return_value = False

    # Act
    response = client.post(
        "/executors",
        json={
            "signature": signature,
            "validator_hotkey": validator_hotkey
        }
    )

    # Assert
    assert response.status_code == status.HTTP_403_FORBIDDEN
    assert "not registered" in response.json()["detail"].lower()


def test_get_executors_signature_without_0x_prefix(client):
    """
    Test that endpoint accepts signature without 0x prefix (normalization).

    Arrange: Mock validator service and executor service with valid data
    Act: POST to /executors with signature WITHOUT 0x prefix
    Assert: Response is 200 - signature is normalized automatically
    """
    # Arrange
    validator_hotkey = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
    signature = "1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"  # NO 0x prefix

    # Act
    response = client.post(
        "/executors",
        json={
            "signature": signature,
            "validator_hotkey": validator_hotkey
        }
    )

    # Assert
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["validator_hotkey"] == validator_hotkey
    assert data["executors"] == []
