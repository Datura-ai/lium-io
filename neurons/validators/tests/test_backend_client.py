"""Tests for BackendClient."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
from pydantic import BaseModel

from neurons.validators.src.clients.backend_client import BackendClient
from neurons.validators.src.protocol.vc_protocol.compute_requests import RentedExecutorsResponse


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


@pytest.mark.asyncio
async def test_session_has_no_total_timeout_cap(reset_session):
    """Regression for DAH-1939: session-level total timeout must not cap per-request timeouts."""
    session = await BackendClient.get_session()
    try:
        assert session.timeout.total is None, (
            "Session-level total timeout must be None so per-request timeouts apply as authored"
        )
    finally:
        await BackendClient.close_session()


@pytest.mark.asyncio
async def test_per_request_timeout_outlives_old_session_cap(reset_session, mock_keypair):
    """Regression for DAH-1939: a request slower than the old 30s cap must complete.

    Spins up a local aiohttp server that delays response by ~1.5s, then issues a POST
    with explicit timeout=5. Before the fix, the 30s session cap would have allowed
    this; the analogous production case (35s response, 300s per-request timeout) was
    being killed at 30s. We use compressed durations here to keep the test fast, and
    additionally assert the session timeout is None (see the test above).
    """
    from aiohttp import web

    async def slow_handler(_request: web.Request) -> web.Response:
        await asyncio.sleep(1.5)
        return web.json_response({"data": "ok", "count": 1})

    app = web.Application()
    app.router.add_post("/slow", slow_handler)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]

    try:
        client = BackendClient(base_url=f"http://127.0.0.1:{port}", keypair=mock_keypair)
        result = await client.post("/slow", SampleResponse, timeout=5)
        assert result is not None
        assert result.data == "ok"
    finally:
        await BackendClient.close_session()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_per_request_timeout_still_enforced(reset_session, mock_keypair):
    """Per-request timeout is honoured (returns None on timeout)."""
    from aiohttp import web

    async def slow_handler(_request: web.Request) -> web.Response:
        await asyncio.sleep(2.0)
        return web.json_response({"data": "ok", "count": 1})

    app = web.Application()
    app.router.add_post("/slow", slow_handler)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]

    try:
        client = BackendClient(base_url=f"http://127.0.0.1:{port}", keypair=mock_keypair)
        result = await client.post("/slow", SampleResponse, timeout=1)
        assert result is None
    finally:
        await BackendClient.close_session()
        await runner.cleanup()


def test_rented_executors_response_defaults_filler_mapping():
    result = RentedExecutorsResponse.model_validate({"executors": {}})

    assert result.executors == {}
    assert result.filler_containers_by_executor == {}


def test_rented_executors_response_keeps_customer_rentals_separate_from_filler():
    result = RentedExecutorsResponse.model_validate(
        {
            "executors": {
                "executor-123": {
                    "miner_hotkey": "miner-hotkey",
                    "executor_ip_address": "127.0.0.1",
                    "executor_ip_port": "22",
                    "pods": [
                        {
                            "pod_id": "pod-1",
                            "container_name": "pod_pod-1",
                            "rented_ports": [8080],
                        }
                    ],
                }
            },
            "filler_containers_by_executor": {"executor-123": "filler_active"},
        }
    )

    assert result.executors["executor-123"].pods[0].container_name == "pod_pod-1"
    assert result.filler_containers_by_executor == {"executor-123": "filler_active"}
    assert result.get_filler_container("executor-123") == "filler_active"
    assert result.get_filler_container("executor-456") is None


def test_rented_executors_response_filters_non_filler_container_names():
    result = RentedExecutorsResponse.model_validate(
        {
            "executors": {},
            "filler_containers_by_executor": {
                "executor-123": "filler_active",
                "executor-456": "pod_customer-rental",
                "executor-789": "container_arbitrary",
            },
        }
    )

    assert result.filler_containers_by_executor == {"executor-123": "filler_active"}


@pytest.mark.asyncio
async def test_get_all_rented_executors_parses_filler_mapping(reset_session, client):
    response_data = {
        "executors": {},
        "filler_containers_by_executor": {
            "executor-123": "filler_5703f4c9-c2f4-4fae-a652-3dee4753030a",
            "executor-456": "pod_customer-rental",
        },
    }
    mock_response = create_mock_response(200, response_data)
    mock_session = create_mock_session(mock_response, "get")

    with patch.object(BackendClient, "get_session", return_value=mock_session):
        result = await client.get_all_rented_executors()

    assert result is not None
    assert result.executors == {}
    assert result.filler_containers_by_executor == {
        "executor-123": "filler_5703f4c9-c2f4-4fae-a652-3dee4753030a"
    }
