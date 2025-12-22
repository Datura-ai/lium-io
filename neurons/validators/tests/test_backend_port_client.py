"""Tests for BackendPortClient."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from neurons.validators.src.clients.backend_port_client import BackendPortClient, RentedPortsResponse


@pytest.fixture
def mock_backend_client():
    return MagicMock()


@pytest.fixture
def client(mock_backend_client):
    return BackendPortClient(backend_client=mock_backend_client)


@pytest.mark.asyncio
async def test_get_rented_ports_success(client, mock_backend_client):
    executor_id = uuid4()
    expected_ports = [8080, 8081, 8082]
    mock_response = RentedPortsResponse(rented_external_ports=expected_ports)
    mock_backend_client.get = AsyncMock(return_value=mock_response)

    result = await client.get_rented_ports(executor_id)

    assert result == set(expected_ports)
    mock_backend_client.get.assert_called_once()


@pytest.mark.asyncio
async def test_get_rented_ports_api_error(client, mock_backend_client):
    executor_id = uuid4()
    mock_backend_client.get = AsyncMock(return_value=None)

    result = await client.get_rented_ports(executor_id)

    assert result == set()


@pytest.mark.asyncio
async def test_get_rented_ports_empty_response(client, mock_backend_client):
    executor_id = uuid4()
    mock_response = RentedPortsResponse(rented_external_ports=[])
    mock_backend_client.get = AsyncMock(return_value=mock_response)

    result = await client.get_rented_ports(executor_id)

    assert result == set()


@pytest.mark.asyncio
async def test_get_rented_ports_correct_path(client, mock_backend_client):
    executor_id = uuid4()
    mock_response = RentedPortsResponse(rented_external_ports=[8080])
    mock_backend_client.get = AsyncMock(return_value=mock_response)

    await client.get_rented_ports(executor_id)

    call_args = mock_backend_client.get.call_args
    expected_path = f"/internal/executors/{executor_id}/rented-ports"
    assert call_args[0][0] == expected_path
    assert call_args[0][1] == RentedPortsResponse
