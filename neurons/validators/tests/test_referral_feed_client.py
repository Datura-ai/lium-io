"""Tests for the fail-closed referral-weights feed client (DAH-2481)."""
from unittest.mock import MagicMock, patch

import pytest
import requests
from clients.referral_feed_client import ReferralFeedClient


def _mock_response(json_body=None, json_side_effect=None, status_ok=True):
    response = MagicMock()
    if status_ok:
        response.raise_for_status.return_value = None
    else:
        response.raise_for_status.side_effect = requests.HTTPError("boom")
    if json_side_effect is not None:
        response.json.side_effect = json_side_effect
    else:
        response.json.return_value = json_body
    return response


@pytest.fixture
def client() -> ReferralFeedClient:
    return ReferralFeedClient()


def test_get_weights_happy_path(client: ReferralFeedClient):
    body = {"epoch_index": 10, "weights": {"HK_A": "1.5", "HK_B": "0.25"}}
    with patch("clients.referral_feed_client.requests.get", return_value=_mock_response(body)):
        result = client.get_weights()

    assert result == {"HK_A": 1.5, "HK_B": 0.25}


def test_get_weights_drops_non_positive_ema(client: ReferralFeedClient):
    body = {"epoch_index": 10, "weights": {"HK_A": "1.0", "HK_C": "0"}}
    with patch("clients.referral_feed_client.requests.get", return_value=_mock_response(body)):
        result = client.get_weights()

    assert result == {"HK_A": 1.0}
    assert "HK_C" not in result


def test_get_weights_http_error_fails_closed(client: ReferralFeedClient):
    with patch(
        "clients.referral_feed_client.requests.get",
        return_value=_mock_response(status_ok=False),
    ):
        result = client.get_weights()

    assert result == {}


def test_get_weights_malformed_json_fails_closed(client: ReferralFeedClient):
    with patch(
        "clients.referral_feed_client.requests.get",
        return_value=_mock_response(json_side_effect=ValueError("bad json")),
    ):
        result = client.get_weights()

    assert result == {}


def test_get_weights_missing_weights_key_fails_closed(client: ReferralFeedClient):
    body = {"epoch_index": 10}
    with patch("clients.referral_feed_client.requests.get", return_value=_mock_response(body)):
        result = client.get_weights()

    assert result == {}


def test_get_weights_empty_weights_fails_closed(client: ReferralFeedClient):
    body = {"epoch_index": 10, "weights": {}}
    with patch("clients.referral_feed_client.requests.get", return_value=_mock_response(body)):
        result = client.get_weights()

    assert result == {}


def test_get_weights_stale_epoch_fails_closed(client: ReferralFeedClient):
    body = {"epoch_index": 5, "weights": {"HK_A": "1.5"}}
    with patch("clients.referral_feed_client.requests.get", return_value=_mock_response(body)):
        result = client.get_weights(current_epoch=100)

    assert result == {}
