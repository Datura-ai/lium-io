"""
Tests for /ping endpoint with signature verification.
Uses AAA (Arrange-Act-Assert) pattern, one test per function.
"""


def test_ping_endpoint_success(test_client, mock_keypair, valid_signature):
    """Test ping endpoint with valid signature returns 200 and pong"""
    # Arrange
    mock_keypair.verify.return_value = True
    payload = {"signature": valid_signature}

    # Act
    response = test_client.post("/ping", json=payload)

    # Assert
    assert response.status_code == 200
    assert response.json() == {"status": "pong"}
    mock_keypair.verify.assert_called_once_with("ping_request", valid_signature)


def test_ping_endpoint_invalid_signature(test_client, mock_keypair, invalid_signature):
    """Test ping endpoint with invalid signature returns 401"""
    # Arrange
    mock_keypair.verify.return_value = False
    payload = {"signature": invalid_signature}

    # Act
    response = test_client.post("/ping", json=payload)

    # Assert
    assert response.status_code == 401
    assert "Invalid signature" in response.json()["detail"]


def test_ping_endpoint_missing_signature(test_client):
    """Test ping endpoint with missing signature returns 422"""
    # Arrange
    payload = {}

    # Act
    response = test_client.post("/ping", json=payload)

    # Assert
    assert response.status_code == 422
    assert "signature" in str(response.json()).lower()
