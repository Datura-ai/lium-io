import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from services.executor_connectivity.batch_port_verifier.batch_port_verifier import BatchPortVerifier


@pytest.fixture
def batch_port_verifier():
    """Create BatchPortVerifier instance for testing."""
    return BatchPortVerifier(max_concurrent=10)


@pytest.mark.asyncio
async def test_get_response_success(batch_port_verifier):
    """Test successful nonce exchange with port."""
    nonce = "test123"
    expected_response = f"OK {nonce}"

    mock_reader = AsyncMock()
    mock_writer = AsyncMock()
    mock_reader.readline.return_value = (expected_response + "\n").encode()

    with patch("asyncio.open_connection", return_value=(mock_reader, mock_writer)):
        result = await batch_port_verifier.get_response("127.0.0.1", 8080, nonce)

        assert result == expected_response
        mock_writer.write.assert_called_once_with((nonce + "\n").encode())


@pytest.mark.asyncio
async def test_get_response_timeout(batch_port_verifier):
    """Test connection timeout handling."""
    nonce = "test123"

    with patch("asyncio.open_connection", side_effect=asyncio.TimeoutError):
        with pytest.raises(asyncio.TimeoutError):
            await batch_port_verifier.get_response("127.0.0.1", 8080, nonce)


@pytest.mark.asyncio
async def test_check_ports_success(batch_port_verifier):
    """Test successful port checking with mock TCP connections."""
    ports = [(9000, 9000), (9001, 9001)]
    external_ip = "192.168.1.100"
    nonce = "abc123"

    with (
        patch("secrets.token_hex", return_value=nonce),
        patch("asyncio.start_server") as mock_start_server,
        patch.object(batch_port_verifier, "get_response") as mock_get_response,
    ):
        mock_server = AsyncMock()
        mock_start_server.return_value = mock_server
        mock_get_response.return_value = f"OK {nonce}"

        results = await batch_port_verifier.check_ports(ports, external_ip)

        assert results == {9000: True, 9001: True}
        assert mock_start_server.call_count == 2


@pytest.mark.asyncio
async def test_check_ports_mixed_results(batch_port_verifier):
    """Test port checking with some ports failing."""
    ports = [(9000, 9000), (9001, 9001)]
    external_ip = "192.168.1.100"
    nonce = "abc123"

    with (
        patch("secrets.token_hex", return_value=nonce),
        patch("asyncio.start_server") as mock_start_server,
        patch.object(batch_port_verifier, "get_response") as mock_get_response,
    ):
        mock_server = AsyncMock()
        mock_start_server.return_value = mock_server
        mock_get_response.side_effect = [f"OK {nonce}", "WRONG_RESPONSE"]

        results = await batch_port_verifier.check_ports(ports, external_ip)

        assert results == {9000: True, 9001: False}


@pytest.mark.asyncio
async def test_get_response_unexpected_response(batch_port_verifier):
    """Test handling of unexpected response from port."""
    nonce = "test123"
    unexpected_response = "WRONG RESPONSE"

    mock_reader = AsyncMock()
    mock_writer = AsyncMock()
    mock_reader.readline.return_value = (unexpected_response + "\n").encode()

    with patch("asyncio.open_connection", return_value=(mock_reader, mock_writer)):
        result = await batch_port_verifier.get_response("127.0.0.1", 8080, nonce)

        assert result == unexpected_response
        mock_writer.write.assert_called_once_with((nonce + "\n").encode())


@pytest.mark.asyncio
async def test_check_ports_empty_list(batch_port_verifier):
    """Test handling of empty ports list."""
    ports = []
    external_ip = "127.0.0.1"

    results = await batch_port_verifier.check_ports(ports, external_ip)

    assert results == {}


@pytest.mark.asyncio
async def test_check_ports_server_bind_failure(batch_port_verifier):
    """Test handling of port binding failures."""
    ports = [(22, 22)]
    external_ip = "127.0.0.1"

    with (
        patch("secrets.token_hex", return_value="test"),
        patch("asyncio.start_server", side_effect=OSError("Port already in use")),
    ):
        results = await batch_port_verifier.check_ports(ports, external_ip)

        assert results == {22: False}
