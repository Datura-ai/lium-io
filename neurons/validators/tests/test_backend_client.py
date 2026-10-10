"""Tests for BackendClient."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from neurons.validators.src.clients.backend_client import BackendClient
from pydantic import BaseModel



class SampleResponse(BaseModel):
    data: str
    count: int


@pytest.fixture
def reset_session():
    BackendClient._session = None
    yield
    BackendClient._session = None


@pytest.fixture
def mock_keypair():
    keypair = MagicMock()
    keypair.ss58_address = "5FakeValidatorHotkey"
    keypair.sign = MagicMock(return_value=b"\x00" * 64)
    return keypair


@pytest.fixture
def client(mock_keypair):
    return BackendClient(base_url="https://api.example.com", keypair=mock_keypair)


def create_mock_response(status: int, json_data: dict | None = None):
    mock_resp = AsyncMock()
    mock_resp.status = status
    mock_resp.json = AsyncMock(return_value=json_data)
    mock_resp.text = AsyncMock(return_value="error")
    mock_resp.content_type = "application/json"
    return mock_resp


def create_mock_session(mock_response):
    mock_session = AsyncMock()
    mock_session.closed = False
    mock_session.close = AsyncMock()

    async_cm = AsyncMock()
    async_cm.__aenter__ = AsyncMock(return_value=mock_response)
    async_cm.__aexit__ = AsyncMock(return_value=None)

    mock_session.request = MagicMock(return_value=async_cm)

    return mock_session


@pytest.mark.asyncio
async def test_get_with_signature(reset_session, client):
    response_data = {"data": "test", "count": 42}
    mock_response = create_mock_response(200, response_data)
    mock_session = create_mock_session(mock_response)

    with patch.object(BackendClient, "get_session", return_value=mock_session):
        await client.get("/data", SampleResponse, add_signature=True)

        call_args = mock_session.request.call_args
        headers = call_args.kwargs.get("headers", {})
        assert "hotkey" in headers
        assert "timestamp" in headers
        assert "signature" in headers


