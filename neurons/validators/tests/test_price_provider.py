"""
Unit tests for PriceProvider - Cache & Retry Logic.

Tests cover:
- Time-based caching with 15-minute TTL
- Retry logic with exponential backoff
- Multi-provider failover for TAO price
- Subtensor integration for alpha rate
- Fallback strategies (expired cache → defaults)
- Helper methods (refresh_cache, clear_cache, set_mock_prices, get_cache_status)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp

from core.config import settings
from core.config import Settings as SettingsCls
from incentive.price_provider import (
    PriceProvider,
    DEFAULT_TAO_PRICE,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def price_provider():
    """Fresh PriceProvider instance for each test."""
    with patch("incentive.price_provider.SubtensorClient") as mock_sc_cls:
        mock_client = MagicMock()
        mock_sc_cls.get_instance.return_value = mock_client
        yield PriceProvider()


@pytest.fixture
def mock_time(monkeypatch):
    """Control time.time() for cache TTL testing without freezegun."""
    current_time = [1000000.0]  # Mutable list for closure

    def _get_time():
        return current_time[0]

    def _set_time(new_time):
        current_time[0] = new_time

    monkeypatch.setattr("time.time", _get_time)
    # Return setter function for tests to advance time
    return _set_time


@pytest.fixture
def mock_aiohttp_session():
    """Mock aiohttp ClientSession for API provider testing."""
    with patch("aiohttp.ClientSession") as mock_session_class:
        # Create session mock
        session = AsyncMock()

        # Setup response mock
        response = AsyncMock()
        response.raise_for_status = MagicMock()
        response.json = AsyncMock()

        # Setup context manager for session.get() - get() is NOT async, it returns an async CM
        get_cm = AsyncMock()
        get_cm.__aenter__ = AsyncMock(return_value=response)
        get_cm.__aexit__ = AsyncMock(return_value=None)
        session.get = MagicMock(return_value=get_cm)  # MagicMock not AsyncMock!

        # Setup context manager for ClientSession itself
        session_cm = AsyncMock()
        session_cm.__aenter__ = AsyncMock(return_value=session)
        session_cm.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = session_cm

        yield session, response


@pytest.fixture
def mock_subtensor():
    """Mock AsyncSubtensor for alpha rate testing."""
    with patch("incentive.price_provider.AsyncSubtensor") as mock_cls:
        subtensor = AsyncMock()
        mock_cls.return_value = subtensor

        # Mock price object with .tao attribute
        price_obj = MagicMock()
        price_obj.tao = 0.001
        subtensor.get_subnet_price = AsyncMock(return_value=price_obj)

        async def shim_to_thread(func, *args, **kwargs):
            price = await subtensor.get_subnet_price(netuid=settings.BITTENSOR_NETUID)
            return price.tao

        with patch("asyncio.to_thread", side_effect=shim_to_thread):
            yield subtensor


@pytest.fixture
def mock_settings(monkeypatch):
    """Mock settings for BITTENSOR_NETUID and get_bittensor_config."""
    mock_config = MagicMock()
    monkeypatch.setattr(settings, "BITTENSOR_NETUID", 1)
    monkeypatch.setattr(SettingsCls, "get_bittensor_config", lambda self: mock_config)
    return mock_config


# ============================================================================
# CACHE LOGIC TESTS
# ============================================================================


# ============================================================================
# RETRY LOGIC TESTS
# ============================================================================


    

# ============================================================================
# TAO PRICE PROVIDER FAILOVER TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_all_providers_fail_fallback_to_expired_cache(price_provider, mock_time, mock_aiohttp_session, caplog):
    """All providers fail but expired cache exists and is returned."""
    # Arrange: Set expired cache
    price_provider._cached_tao_price = 230.0
    price_provider._tao_price_timestamp = 1000000.0
    mock_time(1001000.0)  # 1000s later (expired)

    _, response = mock_aiohttp_session
    response.json = AsyncMock(side_effect=aiohttp.ClientError("All down"))

    # Act
    with patch("asyncio.sleep", new_callable=AsyncMock):
        result = await price_provider.get_tao_price()

    # Assert
    assert result == 230.0
    assert "Falling back to expired TAO price cache" in caplog.text


@pytest.mark.asyncio
async def test_all_providers_fail_no_cache_returns_default(price_provider, mock_aiohttp_session, caplog):
    """All providers fail, no cache exists, returns default."""
    # Arrange: No cache
    _, response = mock_aiohttp_session
    response.json = AsyncMock(side_effect=aiohttp.ClientError("All down"))

    # Act
    with patch("asyncio.sleep", new_callable=AsyncMock):
        result = await price_provider.get_tao_price()

    # Assert
    assert result == DEFAULT_TAO_PRICE
    assert "falling back to default value" in caplog.text


# ============================================================================
# ALPHA RATE PROVIDER TESTS
# ============================================================================


# ============================================================================
# HELPER METHOD TESTS
# ============================================================================


# ============================================================================
# INDIVIDUAL PROVIDER TESTS
# ============================================================================


# ============================================================================
# DATA INTEGRITY TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_positive_values_only(price_provider, mock_aiohttp_session, mock_subtensor, mock_settings):
    """All returned values are positive."""
    # Arrange
    _ = mock_settings  # Ensure settings are mocked
    _, response = mock_aiohttp_session
    response.json.return_value = {"market_data": {"current_price": {"usd": 250.0}}}

    price_obj = MagicMock()
    price_obj.tao = 0.002
    mock_subtensor.get_subnet_price.return_value = price_obj

    # Act
    with patch("asyncio.sleep", new_callable=AsyncMock):
        tao_price = await price_provider.get_tao_price()
        alpha_rate = await price_provider.get_alpha_rate()

    # Assert
    assert tao_price > 0
    assert alpha_rate > 0
