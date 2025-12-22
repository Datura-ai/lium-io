"""Tests for BackendClient."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
from pydantic import BaseModel

from neurons.validators.src.clients.backend_client import BackendClient


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


def create_mock_session(mock_response, method: str = "get"):
    mock_session = AsyncMock()
    mock_session.closed = False
    mock_session.close = AsyncMock()

    async_cm = AsyncMock()
    async_cm.__aenter__ = AsyncMock(return_value=mock_response)
    async_cm.__aexit__ = AsyncMock(return_value=None)

    if method == "get":
        mock_session.get = MagicMock(return_value=async_cm)
    else:
        mock_session.post = MagicMock(return_value=async_cm)

    return mock_session


@pytest.mark.asyncio
async def test_get_success(reset_session, client):
    response_data = {"data": "test", "count": 42}
    mock_response = create_mock_response(200, response_data)
    mock_session = create_mock_session(mock_response, "get")

    with patch.object(BackendClient, "get_session", return_value=mock_session):
        result = await client.get("/data", SampleResponse)

        assert result is not None
        assert result.data == "test"
        assert result.count == 42


@pytest.mark.asyncio
async def test_get_non_200_status(reset_session, client):
    mock_response = create_mock_response(500)
    mock_session = create_mock_session(mock_response, "get")

    with patch.object(BackendClient, "get_session", return_value=mock_session):
        result = await client.get("/data", SampleResponse)
        assert result is None


@pytest.mark.asyncio
async def test_get_validation_error(reset_session, client):
    invalid_data = {"wrong_field": "value"}
    mock_response = create_mock_response(200, invalid_data)
    mock_session = create_mock_session(mock_response, "get")

    with patch.object(BackendClient, "get_session", return_value=mock_session):
        result = await client.get("/data", SampleResponse)
        assert result is None


@pytest.mark.asyncio
async def test_get_timeout(reset_session, client):
    mock_session = AsyncMock()
    async_cm = AsyncMock()
    async_cm.__aenter__ = AsyncMock(side_effect=asyncio.TimeoutError())
    async_cm.__aexit__ = AsyncMock()
    mock_session.get = MagicMock(return_value=async_cm)

    with patch.object(BackendClient, "get_session", return_value=mock_session):
        result = await client.get("/data", SampleResponse, timeout=1)
        assert result is None


@pytest.mark.asyncio
async def test_get_with_signature(reset_session, client):
    response_data = {"data": "test", "count": 42}
    mock_response = create_mock_response(200, response_data)
    mock_session = create_mock_session(mock_response, "get")

    with patch.object(BackendClient, "get_session", return_value=mock_session):
        await client.get("/data", SampleResponse, add_signature=True)

        call_args = mock_session.get.call_args
        headers = call_args.kwargs.get("headers", {})
        assert "hotkey" in headers
        assert "timestamp" in headers
        assert "signature" in headers


@pytest.mark.asyncio
async def test_post_success(reset_session, client):
    request_data = {"action": "create"}
    response_data = {"data": "created", "count": 1}
    mock_response = create_mock_response(200, response_data)
    mock_session = create_mock_session(mock_response, "post")

    with patch.object(BackendClient, "get_session", return_value=mock_session):
        result = await client.post("/submit", SampleResponse, json_data=request_data)

        assert result is not None
        assert result.data == "created"


@pytest.mark.asyncio
async def test_url_construction(reset_session, client):
    response_data = {"data": "test", "count": 42}
    mock_response = create_mock_response(200, response_data)
    mock_session = create_mock_session(mock_response, "get")

    with patch.object(BackendClient, "get_session", return_value=mock_session):
        await client.get("/some/path", SampleResponse)

        call_args = mock_session.get.call_args
        assert call_args[0][0] == "https://api.example.com/some/path"
