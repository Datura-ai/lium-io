"""
Tests for watchtower.py - Docker image monitoring and update service.
"""

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
    find_containers_by_image,
    pull_and_restart_containers,
    check_and_update
)
from models import WatchtowerDigestResponse


class TestGetCurrentImageDigest:
    """Test getting current Docker image digest"""

    def test_successful_digest_retrieval(self):
        """Should extract digest from RepoDigests"""
        mock_client = Mock()
        mock_image = Mock()
        mock_image.attrs = {
            'RepoDigests': ['daturaai/compute-subnet-executor-runner@sha256:abc123def456']
        }
        mock_client.images.get.return_value = mock_image

        digest = get_current_image_digest(mock_client, "daturaai/compute-subnet-executor-runner")

        assert digest == "sha256:abc123def456"
        mock_client.images.get.assert_called_once_with("daturaai/compute-subnet-executor-runner")

    def test_image_not_found(self):
        """Should return None when image doesn't exist locally"""
        mock_client = Mock()
        mock_client.images.get.side_effect = docker.errors.ImageNotFound("not found")

        digest = get_current_image_digest(mock_client, "nonexistent-image")

        assert digest is None

    def test_no_repo_digests(self):
        """Should return None when RepoDigests is empty"""
        mock_client = Mock()
        mock_image = Mock()
        mock_image.attrs = {'RepoDigests': []}
        mock_client.images.get.return_value = mock_image

        digest = get_current_image_digest(mock_client, "test-image")

        assert digest is None

    def test_exception_handling(self):
        """Should return None and log error on exception"""
        mock_client = Mock()
        mock_client.images.get.side_effect = Exception("Unexpected error")

        digest = get_current_image_digest(mock_client, "test-image")

        assert digest is None


class TestVerifyWatchtowerSignature:
    """Test signature verification"""

    @patch('watchtower.bittensor.Keypair')
    @patch('watchtower.settings')
    def test_valid_signature(self, mock_settings, mock_keypair_class):
        """Should verify valid signature without raising exception"""
        mock_settings.WATCHTOWER_VALIDATOR_HOTKEY = "5E1nK3myeWNWrmffVaH76f2mCFCbe9VcHGwgkfdcD7k3E8D1"
        mock_keypair = Mock()
        mock_keypair.verify.return_value = True
        mock_keypair_class.return_value = mock_keypair

        payload = WatchtowerDigestResponse(
            digest="sha256:abc123",
            timestamp=1234567890,
            signature="0xvalid_signature"
        )

        verify_watchtower_signature(payload)
        mock_keypair.verify.assert_called_once()

    @patch('watchtower.bittensor.Keypair')
    @patch('watchtower.settings')
    def test_invalid_signature(self, mock_settings, mock_keypair_class):
        """Should raise exception for invalid signature"""
        mock_settings.WATCHTOWER_VALIDATOR_HOTKEY = "5E1nK3myeWNWrmffVaH76f2mCFCbe9VcHGwgkfdcD7k3E8D1"
        mock_keypair = Mock()
        mock_keypair.verify.return_value = False
        mock_keypair_class.return_value = mock_keypair

        payload = WatchtowerDigestResponse(
            digest="sha256:abc123",
            timestamp=1234567890,
            signature="0xinvalid_signature"
        )

        with pytest.raises(Exception, match="Invalid signature"):
            verify_watchtower_signature(payload)

    @patch('watchtower.bittensor.Keypair')
    @patch('watchtower.settings')
    def test_signature_without_0x_prefix(self, mock_settings, mock_keypair_class):
        """Should add 0x prefix to signature if missing"""
        mock_settings.WATCHTOWER_VALIDATOR_HOTKEY = "5E1nK3myeWNWrmffVaH76f2mCFCbe9VcHGwgkfdcD7k3E8D1"
        mock_keypair = Mock()
        mock_keypair.verify.return_value = True
        mock_keypair_class.return_value = mock_keypair

        payload = WatchtowerDigestResponse(
            digest="sha256:abc123",
            timestamp=1234567890,
            signature="valid_signature_without_prefix"
        )

        verify_watchtower_signature(payload)

        call_args = mock_keypair.verify.call_args
        assert call_args[0][1].startswith('0x')

    @patch('watchtower.bittensor.Keypair')
    @patch('watchtower.settings')
    def test_message_format_with_sorted_keys(self, mock_settings, mock_keypair_class):
        """Should create message with sorted JSON keys"""
        mock_settings.WATCHTOWER_VALIDATOR_HOTKEY = "5E1nK3myeWNWrmffVaH76f2mCFCbe9VcHGwgkfdcD7k3E8D1"
        mock_keypair = Mock()
        mock_keypair.verify.return_value = True
        mock_keypair_class.return_value = mock_keypair

        payload = WatchtowerDigestResponse(
            digest="sha256:test",
            timestamp=9999999,
            signature="0xsig"
        )

        verify_watchtower_signature(payload)

        call_args = mock_keypair.verify.call_args
        message = call_args[0][0]
        assert message == '{"digest": "sha256:test", "timestamp": 9999999}'


class TestFetchVerifiedDigest:
    """Test fetching and verifying remote digest"""

    @patch('watchtower.verify_watchtower_signature')
    @patch('watchtower.requests.get')
    @patch('watchtower.settings')
    def test_successful_fetch(self, mock_settings, mock_get, mock_verify):
        """Should fetch and verify digest successfully"""
        mock_settings.WATCHTOWER_ENDPOINT_URL = "http://test-endpoint.com/digest"
        mock_response = Mock()
        mock_response.json.return_value = {
            "digest": "sha256:newdigest",
            "timestamp": 1234567890,
            "signature": "0xsignature"
        }
        mock_get.return_value = mock_response

        digest = fetch_verified_digest()

        assert digest == "sha256:newdigest"
        mock_get.assert_called_once_with("http://test-endpoint.com/digest", timeout=30)
        mock_verify.assert_called_once()

    @patch('watchtower.requests.get')
    @patch('watchtower.settings')
    def test_request_failure(self, mock_settings, mock_get):
        """Should return None on request failure"""
        mock_settings.WATCHTOWER_ENDPOINT_URL = "http://test-endpoint.com/digest"
        mock_get.side_effect = requests.RequestException("Connection error")

        digest = fetch_verified_digest()

        assert digest is None

    @patch('watchtower.verify_watchtower_signature')
    @patch('watchtower.requests.get')
    @patch('watchtower.settings')
    def test_verification_failure(self, mock_settings, mock_get, mock_verify):
        """Should return None when signature verification fails"""
        mock_settings.WATCHTOWER_ENDPOINT_URL = "http://test-endpoint.com/digest"
        mock_response = Mock()
        mock_response.json.return_value = {
            "digest": "sha256:tampered",
            "timestamp": 1234567890,
            "signature": "0xbadsignature"
        }
        mock_get.return_value = mock_response
        mock_verify.side_effect = Exception("Invalid signature")

        digest = fetch_verified_digest()

        assert digest is None

    @patch('watchtower.requests.get')
    @patch('watchtower.settings')
    def test_http_error_status(self, mock_settings, mock_get):
        """Should return None on HTTP error status"""
        mock_settings.WATCHTOWER_ENDPOINT_URL = "http://test-endpoint.com/digest"
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
        mock_get.return_value = mock_response

        digest = fetch_verified_digest()

        assert digest is None

    @patch('watchtower.verify_watchtower_signature')
    @patch('watchtower.requests.get')
    @patch('watchtower.settings')
    def test_invalid_json_response(self, mock_settings, mock_get, mock_verify):
        """Should return None on invalid JSON response"""
        mock_settings.WATCHTOWER_ENDPOINT_URL = "http://test-endpoint.com/digest"
        mock_response = Mock()
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_get.return_value = mock_response

        digest = fetch_verified_digest()

        assert digest is None


class TestFindContainersByImage:
    """Test finding containers by image name"""

    def test_find_multiple_containers(self):
        """Should return all containers using the image"""
        mock_client = Mock()
        mock_container1 = Mock()
        mock_container1.name = "container_1"
        mock_container2 = Mock()
        mock_container2.name = "container_2"
        mock_client.containers.list.return_value = [mock_container1, mock_container2]

        containers = find_containers_by_image(mock_client, "test-image")

        assert len(containers) == 2
        mock_client.containers.list.assert_called_once_with(filters={"ancestor": "test-image"})

    def test_no_containers_found(self):
        """Should return empty list when no containers found"""
        mock_client = Mock()
        mock_client.containers.list.return_value = []

        containers = find_containers_by_image(mock_client, "test-image")

        assert len(containers) == 0

    def test_exception_handling(self):
        """Should return empty list on exception"""
        mock_client = Mock()
        mock_client.containers.list.side_effect = Exception("Docker error")

        containers = find_containers_by_image(mock_client, "test-image")

        assert len(containers) == 0


class TestPullAndRestartContainers:
    """Test pulling image and restarting containers"""

    def test_successful_pull_and_restart(self):
        """Should pull image and restart all containers"""
        mock_client = Mock()
        mock_container1 = Mock()
        mock_container1.name = "container_1"
        mock_container1.short_id = "abc123"
        mock_container2 = Mock()
        mock_container2.name = "container_2"
        mock_container2.short_id = "def456"

        mock_client.containers.list.return_value = [mock_container1, mock_container2]

        result = pull_and_restart_containers(mock_client, "test-image")

        assert result is True
        mock_client.images.pull.assert_called_once_with("test-image")
        mock_container1.restart.assert_called_once_with(timeout=10)
        mock_container2.restart.assert_called_once_with(timeout=10)

    def test_no_containers_to_restart(self):
        """Should return False when no containers found"""
        mock_client = Mock()
        mock_client.containers.list.return_value = []

        result = pull_and_restart_containers(mock_client, "test-image")

        assert result is False
        mock_client.images.pull.assert_not_called()

    def test_partial_restart_failure(self):
        """Should continue restarting other containers if one fails"""
        mock_client = Mock()
        mock_container1 = Mock()
        mock_container1.name = "container_1"
        mock_container1.short_id = "abc123"
        mock_container1.restart.side_effect = Exception("Restart failed")
        mock_container2 = Mock()
        mock_container2.name = "container_2"
        mock_container2.short_id = "def456"

        mock_client.containers.list.return_value = [mock_container1, mock_container2]

        result = pull_and_restart_containers(mock_client, "test-image")

        assert result is True
        mock_container2.restart.assert_called_once()

    def test_docker_api_error_on_pull(self):
        """Should return False on Docker API error during pull"""
        mock_client = Mock()
        mock_container = Mock()
        mock_container.name = "container_1"
        mock_client.containers.list.return_value = [mock_container]
        mock_client.images.pull.side_effect = docker.errors.APIError("Pull failed")

        result = pull_and_restart_containers(mock_client, "test-image")

        assert result is False
        mock_container.restart.assert_not_called()

    def test_unexpected_error(self):
        """Should return False on unexpected error"""
        mock_client = Mock()
        mock_client.containers.list.side_effect = Exception("Unexpected error")

        result = pull_and_restart_containers(mock_client, "test-image")

        assert result is False


class TestCheckAndUpdate:
    """Test the main check and update logic"""

    @patch('watchtower.pull_and_restart_containers')
    @patch('watchtower.fetch_verified_digest')
    @patch('watchtower.get_current_image_digest')
    @patch('watchtower.docker.from_env')
    @patch('watchtower.settings')
    def test_update_when_digests_differ(self, mock_settings, mock_docker, mock_get_digest, mock_fetch, mock_pull):
        """Should pull and restart when digests differ"""
        mock_settings.WATCHTOWER_IMAGE = "test-image"
        mock_client = Mock()
        mock_docker.return_value = mock_client
        mock_get_digest.return_value = "sha256:old"
        mock_fetch.return_value = "sha256:new"
        mock_pull.return_value = True

        check_and_update()

        mock_pull.assert_called_once_with(mock_client, "test-image")

    @patch('watchtower.pull_and_restart_containers')
    @patch('watchtower.fetch_verified_digest')
    @patch('watchtower.get_current_image_digest')
    @patch('watchtower.docker.from_env')
    @patch('watchtower.settings')
    def test_no_update_when_digests_match(self, mock_settings, mock_docker, mock_get_digest, mock_fetch, mock_pull):
        """Should not update when digests are the same"""
        mock_settings.WATCHTOWER_IMAGE = "test-image"
        mock_docker.return_value = Mock()
        mock_get_digest.return_value = "sha256:same"
        mock_fetch.return_value = "sha256:same"

        check_and_update()

        mock_pull.assert_not_called()

    @patch('watchtower.fetch_verified_digest')
    @patch('watchtower.get_current_image_digest')
    @patch('watchtower.docker.from_env')
    @patch('watchtower.settings')
    def test_skip_update_when_remote_fetch_fails(self, mock_settings, mock_docker, mock_get_digest, mock_fetch):
        """Should skip update when remote digest fetch fails"""
        mock_settings.WATCHTOWER_IMAGE = "test-image"
        mock_docker.return_value = Mock()
        mock_get_digest.return_value = "sha256:current"
        mock_fetch.return_value = None

        check_and_update()

    @patch('watchtower.get_current_image_digest')
    @patch('watchtower.docker.from_env')
    @patch('watchtower.settings')
    def test_skip_update_when_current_digest_unavailable(self, mock_settings, mock_docker, mock_get_digest):
        """Should skip update when current digest cannot be retrieved"""
        mock_settings.WATCHTOWER_IMAGE = "test-image"
        mock_docker.return_value = Mock()
        mock_get_digest.return_value = None

        check_and_update()

    @patch('watchtower.docker.from_env')
    def test_exception_handling(self, mock_docker):
        """Should handle exceptions gracefully"""
        mock_docker.side_effect = Exception("Docker connection error")

        check_and_update()
