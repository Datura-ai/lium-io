from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
import os


@pytest.fixture(scope="session", autouse=True)
def mock_env_variables():
    """Set environment variables for tests"""
    os.environ["MINER_HOTKEY_SS58_ADDRESS"] = "test_miner_hotkey"
    os.environ["DB_URI"] = "sqlite:///test.db"
    os.environ["ENV"] = "test"
    yield
    # Cleanup
    os.environ.pop("MINER_HOTKEY_SS58_ADDRESS", None)
    os.environ.pop("DB_URI", None)
    os.environ.pop("ENV", None)


@pytest.fixture
def mock_keypair():
    """Mock bittensor Keypair for signature verification"""
    with patch('bittensor.Keypair') as mock:
        keypair_instance = MagicMock()
        mock.return_value = keypair_instance
        yield keypair_instance


@pytest.fixture
def test_client(mock_keypair):
    """Create FastAPI test client with mocked dependencies"""
    # Import here to avoid circular dependencies
    from executor import app

    return TestClient(app)


@pytest.fixture
def valid_signature():
    """Valid signature for testing"""
    return "0xabcd1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcd"


@pytest.fixture
def invalid_signature():
    """Invalid signature for testing"""
    return "0xinvalidsignature"
