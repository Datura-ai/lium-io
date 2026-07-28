"""Tests for the fail-closed referral-weights feed client (DAH-2481)."""
from unittest.mock import MagicMock, patch

import aiohttp
import pytest
from clients.referral_feed_client import ReferralFeedClient


def _async_return(value):
    async def _coro(*args, **kwargs):
        return value

    return _coro


def _mock_session(json_body=None, json_side_effect=None, status_ok=True):
    """Build a fake ``aiohttp.ClientSession`` class whose GET yields a canned response.

    Both ``ClientSession(...)`` and ``session.get(...)`` are used as async context
    managers, so each needs ``__aenter__``/``__aexit__`` rather than a plain return value.
    """
    response = MagicMock()
    if status_ok:
        response.raise_for_status.return_value = None
    else:
        response.raise_for_status.side_effect = aiohttp.ClientResponseError(
            request_info=MagicMock(), history=(), status=500
        )
    if json_side_effect is not None:
        response.json = MagicMock(side_effect=json_side_effect)
    else:
        response.json = _async_return(json_body)

    get_ctx = MagicMock()
    get_ctx.__aenter__ = _async_return(response)
    get_ctx.__aexit__ = _async_return(False)

    session = MagicMock()
    session.get.return_value = get_ctx

    session_ctx = MagicMock()
    session_ctx.__aenter__ = _async_return(session)
    session_ctx.__aexit__ = _async_return(False)
    return MagicMock(return_value=session_ctx)


@pytest.fixture
def client() -> ReferralFeedClient:
    return ReferralFeedClient()


@pytest.mark.asyncio
async def test_get_weights_happy_path(client: ReferralFeedClient):
    body = {"epoch_index": 10, "weights": {"HK_A": "1.5", "HK_B": "0.25"}}
    with patch("clients.referral_feed_client.aiohttp.ClientSession", _mock_session(body)):
        result = await client.get_weights()

    assert result == {"HK_A": 1.5, "HK_B": 0.25}


@pytest.mark.asyncio
async def test_get_weights_drops_non_positive_ema(client: ReferralFeedClient):
    body = {"epoch_index": 10, "weights": {"HK_A": "1.0", "HK_C": "0"}}
    with patch("clients.referral_feed_client.aiohttp.ClientSession", _mock_session(body)):
        result = await client.get_weights()

    assert result == {"HK_A": 1.0}
    assert "HK_C" not in result


@pytest.mark.asyncio
async def test_get_weights_http_error_fails_closed(client: ReferralFeedClient):
    with patch(
        "clients.referral_feed_client.aiohttp.ClientSession",
        _mock_session(status_ok=False),
    ):
        result = await client.get_weights()

    assert result == {}


@pytest.mark.asyncio
async def test_get_weights_malformed_json_fails_closed(client: ReferralFeedClient):
    with patch(
        "clients.referral_feed_client.aiohttp.ClientSession",
        _mock_session(json_side_effect=ValueError("bad json")),
    ):
        result = await client.get_weights()

    assert result == {}


@pytest.mark.asyncio
async def test_get_weights_missing_weights_key_fails_closed(client: ReferralFeedClient):
    body = {"epoch_index": 10}
    with patch("clients.referral_feed_client.aiohttp.ClientSession", _mock_session(body)):
        result = await client.get_weights()

    assert result == {}


@pytest.mark.asyncio
async def test_get_weights_empty_weights_fails_closed(client: ReferralFeedClient):
    body = {"epoch_index": 10, "weights": {}}
    with patch("clients.referral_feed_client.aiohttp.ClientSession", _mock_session(body)):
        result = await client.get_weights()

    assert result == {}


@pytest.mark.asyncio
async def test_get_weights_stale_epoch_fails_closed(client: ReferralFeedClient):
    body = {"epoch_index": 5, "weights": {"HK_A": "1.5"}}
    with patch("clients.referral_feed_client.aiohttp.ClientSession", _mock_session(body)):
        result = await client.get_weights(current_epoch=100)

    assert result == {}
