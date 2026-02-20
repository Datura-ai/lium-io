"""
Tests for watchtower.py - Docker image monitoring and update service.
"""

import time
import pytest
from unittest.mock import Mock, patch
import docker
import requests

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from watchtower import (
    get_current_image_digest,
    fetch_verified_digest,
    verify_watchtower_signature,
    find_container_by_name,
    pull_and_restart_containers,
    check_and_update,
    EXECUTOR_RUNNER_CONTAINER_NAME,
)
from models import WatchtowerDigestResponse


# ── get_current_image_digest ──────────────────────────────────────────────────

def test_get_current_image_digest_returns_digest_from_container():
    """Should resolve the image digest via the running container."""
    # Arrange
    mock_client = Mock()
    mock_container = Mock()
    mock_container.attrs = {"Image": "sha256:imageid123"}
    mock_client.containers.get.return_value = mock_container
    mock_image = Mock()
    mock_image.attrs = {"RepoDigests": ["daturaai/compute-subnet-executor-runner@sha256:abc123def456"]}
    mock_client.images.get.return_value = mock_image

    # Act
    digest = get_current_image_digest(mock_client, "executor-runner")

    # Assert — digest is extracted from the part after '@' in RepoDigests
    assert digest == "sha256:abc123def456"
    mock_client.containers.get.assert_called_once_with("executor-runner")
    mock_client.images.get.assert_called_once_with("sha256:imageid123")


def test_get_current_image_digest_returns_none_when_container_not_found():
    """Should return None when the named container does not exist."""
    # Arrange
    mock_client = Mock()
    mock_client.containers.get.side_effect = docker.errors.NotFound("not found")

    # Act
    digest = get_current_image_digest(mock_client, "executor-runner")

    # Assert — NotFound is handled gracefully without raising
    assert digest is None


def test_get_current_image_digest_returns_none_when_container_has_no_image_id():
    """Should return None without querying images when Image attr is absent."""
    # Arrange
    mock_client = Mock()
    mock_container = Mock()
    mock_container.attrs = {"Image": None}
    mock_client.containers.get.return_value = mock_container

    # Act
    digest = get_current_image_digest(mock_client, "executor-runner")

    # Assert — missing image ID short-circuits before calling images.get
    assert digest is None
    mock_client.images.get.assert_not_called()


def test_get_current_image_digest_returns_none_when_repo_digests_empty():
    """Should return None when the image has no RepoDigests."""
    # Arrange
    mock_client = Mock()
    mock_container = Mock()
    mock_container.attrs = {"Image": "sha256:imageid"}
    mock_client.containers.get.return_value = mock_container
    mock_image = Mock()
    mock_image.attrs = {"RepoDigests": []}
    mock_client.images.get.return_value = mock_image

    # Act
    digest = get_current_image_digest(mock_client, "executor-runner")

    # Assert — empty RepoDigests means the digest is unavailable
    assert digest is None


def test_get_current_image_digest_returns_none_on_unexpected_exception():
    """Should return None and not propagate unexpected errors."""
    # Arrange
    mock_client = Mock()
    mock_client.containers.get.side_effect = Exception("Unexpected error")

    # Act
    digest = get_current_image_digest(mock_client, "executor-runner")

    # Assert — unexpected exceptions are swallowed and None returned
    assert digest is None


# ── verify_watchtower_signature ───────────────────────────────────────────────

@patch('watchtower.bittensor.Keypair')
def test_verify_watchtower_signature_accepts_valid_signature(mock_keypair_class):
    """Should complete without raising for a valid, recent-timestamped signature."""
    # Arrange
    mock_keypair = Mock()
    mock_keypair.verify.return_value = True
    mock_keypair_class.return_value = mock_keypair
    payload = WatchtowerDigestResponse(
        digest="sha256:abc123",
        timestamp=int(time.time()),
        signature="0xvalid_signature",
    )

    # Act / Assert — no exception raised for a valid signature
    verify_watchtower_signature(payload)
    mock_keypair.verify.assert_called_once()


@patch('watchtower.bittensor.Keypair')
def test_verify_watchtower_signature_raises_on_invalid_signature(mock_keypair_class):
    """Should raise when keypair.verify returns False."""
    # Arrange
    mock_keypair = Mock()
    mock_keypair.verify.return_value = False
    mock_keypair_class.return_value = mock_keypair
    payload = WatchtowerDigestResponse(
        digest="sha256:abc123",
        timestamp=int(time.time()),
        signature="0xinvalid_signature",
    )

    # Act / Assert — False from verify triggers an exception with a clear message
    with pytest.raises(Exception, match="Invalid signature"):
        verify_watchtower_signature(payload)


@patch('watchtower.bittensor.Keypair')
def test_verify_watchtower_signature_uses_digest_colon_timestamp_format(mock_keypair_class):
    """signing_data passed to keypair.verify should be '{digest}:{timestamp}'."""
    # Arrange
    mock_keypair = Mock()
    mock_keypair.verify.return_value = True
    mock_keypair_class.return_value = mock_keypair
    now = int(time.time())
    payload = WatchtowerDigestResponse(
        digest="sha256:test",
        timestamp=now,
        signature="0xsig",
    )

    # Act
    verify_watchtower_signature(payload)

    # Assert — exact signing data format matches what the validator signs
    call_args = mock_keypair.verify.call_args
    signing_data = call_args[0][0]
    assert signing_data == f"sha256:test:{now}"


@patch('watchtower.bittensor.Keypair')
def test_verify_watchtower_signature_uses_watchtower_validator_hotkey(mock_keypair_class):
    """Keypair should be constructed with the WATCHTOWER_VALIDATOR_HOTKEY constant."""
    # Arrange
    mock_keypair = Mock()
    mock_keypair.verify.return_value = True
    mock_keypair_class.return_value = mock_keypair
    test_hotkey = "5TestHotkeyAddress"
    payload = WatchtowerDigestResponse(
        digest="sha256:abc",
        timestamp=int(time.time()),
        signature="0xsig",
    )

    # Act
    with patch('watchtower.WATCHTOWER_VALIDATOR_HOTKEY', test_hotkey):
        verify_watchtower_signature(payload)

    # Assert — hotkey constant is passed to the Keypair constructor
    mock_keypair_class.assert_called_once_with(ss58_address=test_hotkey)


@patch('watchtower.bittensor.Keypair')
def test_verify_watchtower_signature_raises_on_stale_timestamp(mock_keypair_class):
    """Should raise when payload timestamp is more than 10 minutes in the past."""
    # Arrange
    mock_keypair = Mock()
    mock_keypair.verify.return_value = True
    mock_keypair_class.return_value = mock_keypair
    payload = WatchtowerDigestResponse(
        digest="sha256:abc123",
        timestamp=1000,  # far in the past
        signature="0xsig",
    )

    # Act / Assert — stale timestamp is rejected even with an otherwise valid signature
    with pytest.raises(Exception, match="timestamp out of allowed range"):
        verify_watchtower_signature(payload)


# ── fetch_verified_digest ─────────────────────────────────────────────────────

@patch('watchtower.verify_watchtower_signature')
@patch('watchtower.requests.get')
def test_fetch_verified_digest_returns_digest_on_success(mock_get, mock_verify):
    """Should return the digest when the endpoint responds and signature verifies."""
    # Arrange
    mock_response = Mock()
    mock_response.json.return_value = {
        "digest": "sha256:newdigest",
        "timestamp": 1234567890,
        "signature": "0xsignature",
    }
    mock_get.return_value = mock_response

    # Act
    with patch('watchtower.WATCHTOWER_ENDPOINT_URL', "http://test-endpoint.com/digest"):
        digest = fetch_verified_digest()

    # Assert — verified digest is returned and the correct URL was used
    assert digest == "sha256:newdigest"
    mock_get.assert_called_once_with("http://test-endpoint.com/digest", timeout=30)
    mock_verify.assert_called_once()


@patch('watchtower.requests.get')
def test_fetch_verified_digest_returns_none_on_request_exception(mock_get):
    """Should return None when the HTTP request itself fails."""
    # Arrange
    mock_get.side_effect = requests.RequestException("Connection error")

    # Act
    with patch('watchtower.WATCHTOWER_ENDPOINT_URL', "http://test-endpoint.com/digest"):
        digest = fetch_verified_digest()

    # Assert — network errors are handled and None is returned
    assert digest is None


@patch('watchtower.verify_watchtower_signature')
@patch('watchtower.requests.get')
def test_fetch_verified_digest_returns_none_when_signature_invalid(mock_get, mock_verify):
    """Should return None when signature verification raises."""
    # Arrange
    mock_response = Mock()
    mock_response.json.return_value = {
        "digest": "sha256:tampered",
        "timestamp": 1234567890,
        "signature": "0xbadsignature",
    }
    mock_get.return_value = mock_response
    mock_verify.side_effect = Exception("Invalid signature")

    # Act
    with patch('watchtower.WATCHTOWER_ENDPOINT_URL', "http://test-endpoint.com/digest"):
        digest = fetch_verified_digest()

    # Assert — verification failure causes None to be returned
    assert digest is None


@patch('watchtower.requests.get')
def test_fetch_verified_digest_returns_none_on_http_error(mock_get):
    """Should return None when the endpoint returns an HTTP error status."""
    # Arrange
    mock_response = Mock()
    mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
    mock_get.return_value = mock_response

    # Act
    with patch('watchtower.WATCHTOWER_ENDPOINT_URL', "http://test-endpoint.com/digest"):
        digest = fetch_verified_digest()

    # Assert — HTTP errors propagate as None
    assert digest is None


@patch('watchtower.requests.get')
def test_fetch_verified_digest_returns_none_on_invalid_json(mock_get):
    """Should return None when the response body is not valid JSON."""
    # Arrange
    mock_response = Mock()
    mock_response.json.side_effect = ValueError("Invalid JSON")
    mock_get.return_value = mock_response

    # Act
    with patch('watchtower.WATCHTOWER_ENDPOINT_URL', "http://test-endpoint.com/digest"):
        digest = fetch_verified_digest()

    # Assert — malformed response is handled gracefully
    assert digest is None


# ── find_container_by_name ────────────────────────────────────────────────────

def test_find_container_by_name_returns_container_when_found():
    """Should return the container object when it exists by name."""
    # Arrange
    mock_client = Mock()
    mock_container = Mock()
    mock_container.short_id = "abc123"
    mock_container.status = "running"
    mock_client.containers.get.return_value = mock_container

    # Act
    result = find_container_by_name(mock_client, "executor-runner")

    # Assert — correct container object returned and looked up by name
    assert result is mock_container
    mock_client.containers.get.assert_called_once_with("executor-runner")


def test_find_container_by_name_returns_none_when_not_found():
    """Should return None when the container does not exist."""
    # Arrange
    mock_client = Mock()
    mock_client.containers.get.side_effect = docker.errors.NotFound("not found")

    # Act
    result = find_container_by_name(mock_client, "executor-runner")

    # Assert — NotFound is handled and None returned
    assert result is None


def test_find_container_by_name_returns_none_on_exception():
    """Should return None on any unexpected Docker error."""
    # Arrange
    mock_client = Mock()
    mock_client.containers.get.side_effect = Exception("Docker error")

    # Act
    result = find_container_by_name(mock_client, "executor-runner")

    # Assert — unexpected errors are swallowed and None returned
    assert result is None


# ── pull_and_restart_containers ───────────────────────────────────────────────

@patch('watchtower.find_container_by_name')
def test_pull_and_restart_containers_stops_old_and_creates_new(mock_find):
    """Should pull the new image, stop/remove the old container, then create a fresh one."""
    # Arrange
    mock_client = Mock()
    mock_container = Mock()
    mock_container.short_id = "old123"
    mock_find.return_value = mock_container

    # Act
    result = pull_and_restart_containers(mock_client, "test-image", "sha256:newdigest")

    # Assert — full lifecycle: pull → stop → remove → create
    assert result is True
    mock_client.images.pull.assert_called_once_with("test-image@sha256:newdigest")
    mock_container.stop.assert_called_once_with(timeout=10)
    mock_container.remove.assert_called_once()
    mock_client.containers.run.assert_called_once()


@patch('watchtower.find_container_by_name')
def test_pull_and_restart_containers_creates_container_when_none_exists(mock_find):
    """Should still create a new container even when no prior container is found."""
    # Arrange
    mock_client = Mock()
    mock_find.return_value = None

    # Act
    result = pull_and_restart_containers(mock_client, "test-image", "sha256:newdigest")

    # Assert — no stop/remove attempted, but new container is still created
    assert result is True
    mock_client.images.pull.assert_called_once_with("test-image@sha256:newdigest")
    mock_client.containers.run.assert_called_once()


@patch('watchtower.find_container_by_name')
def test_pull_and_restart_containers_creates_container_with_digest_image(mock_find):
    """New container should be launched from the exact digest-pinned image reference."""
    # Arrange
    mock_client = Mock()
    mock_find.return_value = None

    # Act
    pull_and_restart_containers(mock_client, "daturaai/executor", "sha256:abc123")

    # Assert — containers.run receives the fully-qualified image@digest reference
    call_kwargs = mock_client.containers.run.call_args
    image_arg = call_kwargs[0][0]
    assert image_arg == "daturaai/executor@sha256:abc123"


@patch('watchtower.find_container_by_name')
def test_pull_and_restart_containers_returns_false_on_docker_api_error(mock_find):
    """Should return False when Docker raises an APIError during pull."""
    # Arrange
    mock_client = Mock()
    mock_find.return_value = Mock()
    mock_client.images.pull.side_effect = docker.errors.APIError("Pull failed")

    # Act
    result = pull_and_restart_containers(mock_client, "test-image", "sha256:digest")

    # Assert — Docker API errors are caught and failure is reported
    assert result is False


@patch('watchtower.find_container_by_name')
def test_pull_and_restart_containers_returns_false_on_unexpected_error(mock_find):
    """Should return False on any unexpected exception."""
    # Arrange
    mock_client = Mock()
    mock_find.side_effect = Exception("Unexpected error")

    # Act
    result = pull_and_restart_containers(mock_client, "test-image", "sha256:digest")

    # Assert — unexpected errors are caught and failure is reported
    assert result is False


# ── check_and_update ──────────────────────────────────────────────────────────

@patch('watchtower.pull_and_restart_containers')
@patch('watchtower.fetch_verified_digest')
@patch('watchtower.get_current_image_digest')
@patch('watchtower.docker.from_env')
@patch('watchtower.settings')
def test_check_and_update_pulls_when_digests_differ(
    mock_settings, mock_docker, mock_get_digest, mock_fetch, mock_pull
):
    """Should trigger pull_and_restart when current and remote digests differ."""
    # Arrange
    mock_settings.WATCHTOWER_IMAGE = "test-image"
    mock_client = Mock()
    mock_docker.return_value = mock_client
    mock_get_digest.return_value = "sha256:old"
    mock_fetch.return_value = "sha256:new"
    mock_pull.return_value = True

    # Act
    check_and_update()

    # Assert — pull triggered with the client, image name, and the new remote digest
    mock_pull.assert_called_once_with(mock_client, "test-image", "sha256:new")


@patch('watchtower.pull_and_restart_containers')
@patch('watchtower.fetch_verified_digest')
@patch('watchtower.get_current_image_digest')
@patch('watchtower.docker.from_env')
@patch('watchtower.settings')
def test_check_and_update_skips_pull_when_digests_match(
    mock_settings, mock_docker, mock_get_digest, mock_fetch, mock_pull
):
    """Should not pull when the current digest already matches the remote one."""
    # Arrange
    mock_settings.WATCHTOWER_IMAGE = "test-image"
    mock_docker.return_value = Mock()
    mock_get_digest.return_value = "sha256:same"
    mock_fetch.return_value = "sha256:same"

    # Act
    check_and_update()

    # Assert — identical digests mean the image is already up to date
    mock_pull.assert_not_called()


@patch('watchtower.fetch_verified_digest')
@patch('watchtower.get_current_image_digest')
@patch('watchtower.docker.from_env')
@patch('watchtower.settings')
def test_check_and_update_skips_when_remote_fetch_fails(
    mock_settings, mock_docker, mock_get_digest, mock_fetch
):
    """Should return early without error when the remote digest cannot be fetched."""
    # Arrange
    mock_settings.WATCHTOWER_IMAGE = "test-image"
    mock_docker.return_value = Mock()
    mock_get_digest.return_value = "sha256:current"
    mock_fetch.return_value = None

    # Act / Assert — None from fetch does not cause an exception
    check_and_update()


@patch('watchtower.docker.from_env')
def test_check_and_update_handles_docker_connection_error(mock_docker):
    """Should handle Docker client initialisation errors without raising."""
    # Arrange
    mock_docker.side_effect = Exception("Docker connection error")

    # Act / Assert — top-level exception handler prevents propagation
    check_and_update()
