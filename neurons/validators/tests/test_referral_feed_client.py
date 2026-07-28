"""Tests for the fail-closed referral-weights feed client (DAH-2251)."""
from unittest.mock import MagicMock, patch

import aiohttp
import pytest
from clients.referral_feed_client import ReferralFeedClient

from core.config import settings


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


class TestFeedUrlDerivation:
    """`REFERRAL_FEED_URL` unset must follow the deployment, not resolve to prod.

    A hardcoded prod default meant a staging validator read prod referral weights and
    would have paid staging emission against real customers' referrals.
    """

    def test_derives_from_compute_rest_api_url_when_unset(self, monkeypatch):
        monkeypatch.setattr(settings, "REFERRAL_FEED_URL", None)
        monkeypatch.setattr(settings, "COMPUTE_REST_API_URL", "https://staging.lium.io/api")

        assert settings.get_referral_feed_url() == "https://staging.lium.io/api/v1/referral-weights"

    def test_explicit_override_wins(self, monkeypatch):
        monkeypatch.setattr(settings, "REFERRAL_FEED_URL", "https://elsewhere.example/feed")
        monkeypatch.setattr(settings, "COMPUTE_REST_API_URL", "https://staging.lium.io/api")

        assert settings.get_referral_feed_url() == "https://elsewhere.example/feed"

    def test_trailing_slash_does_not_double_up(self, monkeypatch):
        monkeypatch.setattr(settings, "REFERRAL_FEED_URL", None)
        monkeypatch.setattr(settings, "COMPUTE_REST_API_URL", "https://lium.io/api/")

        assert settings.get_referral_feed_url() == "https://lium.io/api/v1/referral-weights"

    def test_returns_empty_when_neither_is_set(self, monkeypatch):
        """No base URL -> empty, which the client's fail-closed path turns into {}."""
        monkeypatch.setattr(settings, "REFERRAL_FEED_URL", None)
        monkeypatch.setattr(settings, "COMPUTE_REST_API_URL", None)

        assert settings.get_referral_feed_url() == ""

    @pytest.mark.asyncio
    async def test_client_fetches_the_derived_url(self, client, monkeypatch):
        """End-to-end: the client GETs the derived URL, not the raw setting."""
        monkeypatch.setattr(settings, "REFERRAL_FEED_URL", None)
        monkeypatch.setattr(settings, "COMPUTE_REST_API_URL", "https://staging.lium.io/api")
        session_cls = _mock_session({"epoch_index": 10, "weights": {"HK_A": "1.5"}})

        with patch("clients.referral_feed_client.aiohttp.ClientSession", session_cls):
            await client.get_weights()

        requested = session_cls.return_value.__aenter__
        # __aenter__ is our coroutine factory; grab the session it yields to read the call.
        session = await requested()
        session.get.assert_called_once_with("https://staging.lium.io/api/v1/referral-weights")
