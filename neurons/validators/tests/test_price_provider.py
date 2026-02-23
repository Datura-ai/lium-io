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
    CoinGeckoProvider,
    CoinbaseProvider,
    CryptoCompareProvider,
    DEFAULT_TAO_PRICE,
    DEFAULT_ALPHA_RATE,
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


def test_cache_hit_tao_price_valid_cache(price_provider, mock_time):
    """TAO price cache hit when cache is valid within TTL."""
    # Arrange: Set valid cache at t=1000000
    price_provider._cached_tao_price = 250.0
    price_provider._tao_price_timestamp = 1000000.0

    # Still within TTL at t=1000500 (500s < 900s)
    mock_time(1000500.0)

    # Act
    result = price_provider._is_cache_valid(price_provider._tao_price_timestamp)

    # Assert
    assert result is True
    assert price_provider._cached_tao_price == 250.0


@pytest.mark.asyncio
async def test_cache_hit_prevents_api_call(price_provider, mock_time, mock_aiohttp_session):
    """Valid TAO price cache prevents API call."""
    # Arrange: Set valid cache
    price_provider._cached_tao_price = 250.0
    price_provider._tao_price_timestamp = 1000000.0
    mock_time(1000500.0)  # Within TTL

    _, response = mock_aiohttp_session

    # Act
    result = await price_provider.get_tao_price()

    # Assert
    assert result == 250.0
    response.json.assert_not_called()  # No API call made


def test_cache_hit_alpha_rate_valid_cache(price_provider, mock_time):
    """Alpha rate cache hit when cache is valid within TTL."""
    # Arrange: Set valid cache
    price_provider._cached_alpha_rate = 0.002
    price_provider._alpha_rate_timestamp = 1000000.0

    # Still within TTL at t=1000500
    mock_time(1000500.0)

    # Act
    result = price_provider._is_cache_valid(price_provider._alpha_rate_timestamp)

    # Assert
    assert result is True
    assert price_provider._cached_alpha_rate == 0.002


@pytest.mark.asyncio
async def test_cache_hit_prevents_subtensor_call(price_provider, mock_time, mock_subtensor, mock_settings):
    """Valid alpha rate cache prevents subtensor call."""
    # Arrange: Set valid cache
    price_provider._cached_alpha_rate = 0.002
    price_provider._alpha_rate_timestamp = 1000000.0
    mock_time(1000500.0)  # Within TTL

    # Act
    result = await price_provider.get_alpha_rate()

    # Assert
    assert result == 0.002
    mock_subtensor.get_subnet_price.assert_not_called()


@pytest.mark.asyncio
async def test_cache_miss_no_cache_exists(price_provider, mock_aiohttp_session):
    """Empty cache triggers fresh fetch."""
    # Arrange: No cache exists
    session, response = mock_aiohttp_session
    response.json.return_value = {"market_data": {"current_price": {"usd": 260.0}}}

    # Act
    with patch("asyncio.sleep", new_callable=AsyncMock):
        result = await price_provider.get_tao_price()

    # Assert
    assert result == 260.0
    session.get.assert_called()


def test_cache_miss_expired_beyond_ttl(price_provider, mock_time):
    """Cache expired (timestamp + 901s) is invalid."""
    # Arrange: Set cache at t=1000000
    price_provider._cached_alpha_rate = 0.002
    price_provider._alpha_rate_timestamp = 1000000.0

    # Advance time by 901s (beyond 900s TTL)
    mock_time(1000901.0)

    # Act
    result = price_provider._is_cache_valid(price_provider._alpha_rate_timestamp)

    # Assert: Cache should be invalid
    assert result is False


def test_cache_miss_exactly_at_ttl_boundary(price_provider, mock_time):
    """Edge case: timestamp + 900s exactly should be invalid."""
    # Arrange: Set cache at t=1000000
    price_provider._cached_alpha_rate = 0.002
    price_provider._alpha_rate_timestamp = 1000000.0

    # Exactly at TTL boundary (900s)
    mock_time(1000900.0)

    # Act
    result = price_provider._is_cache_valid(price_provider._alpha_rate_timestamp)

    # Assert: Should be invalid (< not <=)
    assert result is False


@pytest.mark.asyncio
async def test_cache_update_after_successful_tao_fetch(price_provider, mock_time, mock_aiohttp_session):
    """TAO price updates cache on successful fetch."""
    # Arrange
    session, response = mock_aiohttp_session
    response.json.return_value = {"market_data": {"current_price": {"usd": 245.0}}}
    mock_time(1000000.0)

    # Act
    with patch("asyncio.sleep", new_callable=AsyncMock):
        result = await price_provider.get_tao_price()

    # Assert: Cache updated
    assert result == 245.0
    assert price_provider._cached_tao_price == 245.0
    assert price_provider._tao_price_timestamp == 1000000.0


@pytest.mark.asyncio
async def test_cache_update_after_successful_alpha_fetch(price_provider, mock_time, mock_subtensor, mock_settings):
    """Alpha rate updates cache on successful fetch."""
    # Arrange
    price_obj = MagicMock()
    price_obj.tao = 0.0025
    mock_subtensor.get_subnet_price = AsyncMock(return_value=price_obj)
    mock_time(1000000.0)

    # Act
    with patch("asyncio.sleep", new_callable=AsyncMock):
        result = await price_provider.get_alpha_rate()

    # Assert: Cache updated
    assert result == 0.0025
    assert price_provider._cached_alpha_rate == 0.0025
    assert price_provider._alpha_rate_timestamp == 1000000.0


def test_clear_cache_resets_all_values(price_provider):
    """clear_cache() resets all cached state."""
    # Arrange: Set some cached values
    price_provider._cached_tao_price = 250.0
    price_provider._tao_price_timestamp = 1000000.0
    price_provider._cached_alpha_rate = 0.002
    price_provider._alpha_rate_timestamp = 1000000.0

    # Act
    price_provider.clear_cache()

    # Assert: All values reset to None
    assert price_provider._cached_tao_price is None
    assert price_provider._tao_price_timestamp is None
    assert price_provider._cached_alpha_rate is None
    assert price_provider._alpha_rate_timestamp is None


def test_is_cache_valid_with_none_timestamp(price_provider):
    """_is_cache_valid(None) returns False."""
    # Act
    result = price_provider._is_cache_valid(None)

    # Assert
    assert result is False


def test_is_cache_valid_with_valid_timestamp(price_provider, mock_time):
    """_is_cache_valid(recent_timestamp) returns True."""
    # Arrange: Set time and create recent timestamp
    mock_time(1000000.0)
    recent_timestamp = 999500.0  # 500s ago

    # Act
    result = price_provider._is_cache_valid(recent_timestamp)

    # Assert
    assert result is True


# ============================================================================
# RETRY LOGIC TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_tao_price_success_first_attempt(price_provider, mock_aiohttp_session):
    """TAO price succeeds on first attempt without retries."""
    # Arrange
    session, response = mock_aiohttp_session
    response.json.return_value = {"market_data": {"current_price": {"usd": 250.0}}}

    # Act
    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        result = await price_provider.get_tao_price()

    # Assert
    assert result == 250.0
    mock_sleep.assert_not_called()  # No retries


@pytest.mark.asyncio
async def test_tao_price_success_after_3_retries(price_provider, mock_aiohttp_session):
    """TAO price succeeds after 3 retry attempts."""
    # Arrange
    session, response = mock_aiohttp_session
    attempt_count = [0]

    async def failing_then_success():
        attempt_count[0] += 1
        if attempt_count[0] < 4:
            raise aiohttp.ClientError("Temporary failure")
        return {"market_data": {"current_price": {"usd": 250.0}}}

    response.json.side_effect = failing_then_success

    # Act
    with patch("asyncio.sleep", new_callable=AsyncMock):
        result = await price_provider.get_tao_price()

    # Assert
    assert result == 250.0
    assert attempt_count[0] == 4  # 1 initial + 3 retries


@pytest.mark.asyncio
async def test_tao_price_all_retries_exhausted(price_provider, mock_aiohttp_session):
    """All 6 retry attempts fail and exception is raised."""
    # Arrange
    session, response = mock_aiohttp_session
    response.json = AsyncMock(side_effect=aiohttp.ClientError("Persistent failure"))

    # Act & Assert
    with patch("asyncio.sleep", new_callable=AsyncMock):
        result = await price_provider.get_tao_price()

    # Assert
    assert result == DEFAULT_TAO_PRICE


@pytest.mark.asyncio
async def test_alpha_rate_success_after_retries(price_provider, mock_subtensor, mock_settings):
    """Alpha rate succeeds after retry attempts."""
    # Arrange
    attempt_count = [0]

    async def failing_then_success(netuid):
        attempt_count[0] += 1
        if attempt_count[0] < 3:
            raise Exception("Temporary subtensor failure")
        price_obj = MagicMock()
        price_obj.tao = 0.0015
        return price_obj

    mock_subtensor.get_subnet_price.side_effect = failing_then_success

    # Act
    with patch("asyncio.sleep", new_callable=AsyncMock):
        result = await price_provider.get_alpha_rate()

    # Assert
    assert result == 0.0015
    assert attempt_count[0] == 3


@pytest.mark.asyncio
async def test_retry_on_different_exception_types(price_provider, mock_aiohttp_session):
    """Non-ClientError exceptions also trigger retry."""
    # Arrange
    _, response = mock_aiohttp_session
    attempt_count = [0]
    price_provider.providers = [CoinGeckoProvider()]

    async def failing_with_generic_exception():
        attempt_count[0] += 1
        if attempt_count[0] < 2:
            raise aiohttp.ClientError("Generic failure")
        return {"market_data": {"current_price": {"usd": 255.0}}}

    response.json.side_effect = failing_with_generic_exception

    # Act
    with patch("asyncio.sleep", new_callable=AsyncMock):
        result = await price_provider.get_tao_price()

    # Assert
    assert result == 255.0
    assert attempt_count[0] == 2


@pytest.mark.asyncio
async def test_retry_before_sleep_logging(price_provider, mock_aiohttp_session, caplog):
    """Warning logs before each retry attempt."""
    # Arrange
    _, response = mock_aiohttp_session
    attempt_count = [0]
    price_provider.providers = [CoinGeckoProvider()]

    async def failing_then_success():
        attempt_count[0] += 1
        if attempt_count[0] < 3:
            raise aiohttp.ClientError("Retry test")
        return {"market_data": {"current_price": {"usd": 240.0}}}

    response.json.side_effect = failing_then_success

    # Act
    with patch("asyncio.sleep", new_callable=AsyncMock):
        await price_provider.get_tao_price()

    # Assert: Check for retry warnings in logs
    assert "Retrying" in caplog.text or "retry" in caplog.text.lower()


@pytest.mark.asyncio
async def test_retry_reraise_on_final_failure(price_provider, mock_aiohttp_session):
    """Exception re-raised after final retry attempt."""
    # Arrange
    _, response = mock_aiohttp_session
    response.json = AsyncMock(side_effect=aiohttp.ClientError("Final failure"))

    # Act & Assert: Verify reraise=True behavior
    with patch("asyncio.sleep", new_callable=AsyncMock):
        result = await price_provider.get_tao_price()

    # Assert
    assert result == DEFAULT_TAO_PRICE
    

# ============================================================================
# TAO PRICE PROVIDER FAILOVER TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_coingecko_success_first_provider(price_provider, mock_aiohttp_session):
    """CoinGecko succeeds on first provider."""
    # Arrange
    _, response = mock_aiohttp_session
    response.json.return_value = {"market_data": {"current_price": {"usd": 250.5}}}

    # Act
    with patch("asyncio.sleep", new_callable=AsyncMock):
        result = await price_provider.get_tao_price()

    # Assert
    assert result == 250.5


@pytest.mark.asyncio
async def test_coingecko_fails_coinbase_success(price_provider, mock_aiohttp_session):
    """CoinGecko fails, Coinbase succeeds."""
    # Arrange
    _, response = mock_aiohttp_session
    call_count = [0]

    async def simulate_failover():
        call_count[0] += 1
        if call_count[0] == 1:
            # CoinGecko fails
            raise aiohttp.ClientError("CoinGecko down")
        elif call_count[0] == 2:
            # Coinbase succeeds
            return {"data": {"amount": "245.50"}}

    response.json.side_effect = simulate_failover

    # Act
    with patch("asyncio.sleep", new_callable=AsyncMock):
        result = await price_provider.get_tao_price()

    # Assert
    assert result == 245.5
    assert call_count[0] == 2


@pytest.mark.asyncio
async def test_first_two_fail_cryptocompare_success(price_provider, mock_aiohttp_session):
    """CoinGecko + Coinbase fail, CryptoCompare succeeds."""
    # Arrange
    _, response = mock_aiohttp_session
    call_count = [0]

    async def simulate_full_failover():
        call_count[0] += 1
        if call_count[0] in [1, 2]:
            raise aiohttp.ClientError("Provider down")
        else:
            # CryptoCompare succeeds
            return {"USD": 248.75}

    response.json.side_effect = simulate_full_failover

    # Act
    with patch("asyncio.sleep", new_callable=AsyncMock):
        result = await price_provider.get_tao_price()

    # Assert
    assert result == 248.75
    assert call_count[0] == 3


@pytest.mark.asyncio
async def test_all_providers_fail_raises_exception(price_provider, mock_aiohttp_session):
    """All 3 providers fail and exception is raised."""
    # Arrange
    _, response = mock_aiohttp_session
    response.json = AsyncMock(side_effect=aiohttp.ClientError("All providers down"))

    # Act & Assert
    with patch("asyncio.sleep", new_callable=AsyncMock):
        result = await price_provider.get_tao_price()

    # Assert
    assert result == DEFAULT_TAO_PRICE


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


@pytest.mark.asyncio
async def test_parse_coingecko_response_format(mock_aiohttp_session):
    """Correctly parse CoinGecko JSON structure."""
    # Arrange
    _, response = mock_aiohttp_session
    response.json.return_value = {
        "market_data": {
            "current_price": {
                "usd": 252.30,
                "eur": 230.50
            }
        }
    }

    provider = CoinGeckoProvider()

    # Act
    result = await provider.get_rate()

    # Assert
    assert result == 252.30


@pytest.mark.asyncio
async def test_parse_coinbase_response_format(mock_aiohttp_session):
    """Correctly parse Coinbase JSON (string amount)."""
    # Arrange
    _, response = mock_aiohttp_session
    response.json.return_value = {
        "data": {
            "amount": "247.89"
        }
    }

    provider = CoinbaseProvider()

    # Act
    result = await provider.get_rate()

    # Assert
    assert result == 247.89


@pytest.mark.asyncio
async def test_parse_cryptocompare_response_format(mock_aiohttp_session):
    """Correctly parse CryptoCompare JSON."""
    # Arrange
    _, response = mock_aiohttp_session
    response.json.return_value = {
        "USD": 249.12,
        "EUR": 228.50
    }

    provider = CryptoCompareProvider()

    # Act
    result = await provider.get_rate()

    # Assert
    assert result == 249.12


@pytest.mark.asyncio
async def test_http_timeout_triggers_next_provider(price_provider, mock_aiohttp_session):
    """Provider timeout triggers failover to next provider."""
    # Arrange
    _, response = mock_aiohttp_session
    call_count = [0]

    async def timeout_then_success():
        call_count[0] += 1
        if call_count[0] == 1:
            raise aiohttp.ClientError("Timeout")
        return {"data": {"amount": "243.50"}}

    response.json.side_effect = timeout_then_success

    # Act
    with patch("asyncio.sleep", new_callable=AsyncMock):
        result = await price_provider.get_tao_price()

    # Assert
    assert result == 243.5
    assert call_count[0] == 2


# ============================================================================
# ALPHA RATE PROVIDER TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_alpha_rate_successful_fetch_updates_cache(price_provider, mock_time, mock_subtensor, mock_settings):
    """Successful alpha rate fetch updates cache."""
    # Arrange
    _ = mock_settings  # Ensure settings are mocked
    price_obj = MagicMock()
    price_obj.tao = 0.0015
    mock_subtensor.get_subnet_price.return_value = price_obj
    mock_time(1000000.0)

    # Act
    with patch("asyncio.sleep", new_callable=AsyncMock):
        result = await price_provider.get_alpha_rate()

    # Assert
    assert result == 0.0015
    assert price_provider._cached_alpha_rate == 0.0015
    assert price_provider._alpha_rate_timestamp == 1000000.0


@pytest.mark.asyncio
async def test_subtensor_exception_fallback_to_expired_cache(price_provider, mock_time, mock_subtensor, mock_settings, caplog):
    """Subtensor fails, expired cache exists and is returned."""
    # Arrange: Set expired cache
    _ = mock_settings  # Ensure settings are mocked
    price_provider._cached_alpha_rate = 0.0018
    price_provider._alpha_rate_timestamp = 1000000.0
    mock_time(1001000.0)  # Expired

    mock_subtensor.get_subnet_price.side_effect = Exception("Subtensor down")

    # Act
    with patch("asyncio.sleep", new_callable=AsyncMock):
        result = await price_provider.get_alpha_rate()

    # Assert
    assert result == 0.0018
    assert "Falling back to expired alpha rate cache" in caplog.text


@pytest.mark.asyncio
async def test_subtensor_exception_no_cache_returns_default(price_provider, mock_subtensor, mock_settings, caplog):
    """Subtensor fails, no cache exists, returns default."""
    # Arrange: No cache
    _ = mock_settings  # Ensure settings are mocked
    mock_subtensor.get_subnet_price.side_effect = Exception("Subtensor down")

    # Act
    with patch("asyncio.sleep", new_callable=AsyncMock):
        result = await price_provider.get_alpha_rate()

    # Assert
    assert result == DEFAULT_ALPHA_RATE
    assert "falling back to default value" in caplog.text


@pytest.mark.asyncio
async def test_subtensor_called_with_correct_netuid(price_provider, mock_subtensor, mock_settings):
    """Subtensor called with correct netuid parameter."""
    # Arrange
    _ = mock_settings  # Ensure settings are mocked
    price_obj = MagicMock()
    price_obj.tao = 0.0012
    mock_subtensor.get_subnet_price.return_value = price_obj

    # Act
    with patch("asyncio.sleep", new_callable=AsyncMock):
        await price_provider.get_alpha_rate()

    # Assert
    mock_subtensor.get_subnet_price.assert_called_once_with(netuid=1)


@pytest.mark.asyncio
async def test_parse_price_object_tao_attribute(price_provider, mock_subtensor, mock_settings):
    """Correctly extracts .tao attribute from price object."""
    # Arrange
    _ = mock_settings  # Ensure settings are mocked
    price_obj = MagicMock()
    price_obj.tao = 0.0022
    mock_subtensor.get_subnet_price.return_value = price_obj

    # Act
    with patch("asyncio.sleep", new_callable=AsyncMock):
        result = await price_provider.get_alpha_rate()

    # Assert
    assert result == 0.0022


# ============================================================================
# HELPER METHOD TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_refresh_cache_clears_and_fetches(price_provider, mock_aiohttp_session, mock_subtensor, mock_settings):
    """refresh_cache() clears cache and fetches fresh data."""
    # Arrange: Set old cache
    _ = mock_settings  # Ensure settings are mocked
    price_provider._cached_tao_price = 200.0
    price_provider._cached_alpha_rate = 0.001

    _, response = mock_aiohttp_session
    response.json.return_value = {"market_data": {"current_price": {"usd": 260.0}}}

    price_obj = MagicMock()
    price_obj.tao = 0.0025
    mock_subtensor.get_subnet_price.return_value = price_obj

    # Act
    with patch("asyncio.sleep", new_callable=AsyncMock):
        await price_provider.refresh_cache()

    # Assert: Cache updated with new values
    assert price_provider._cached_tao_price == 260.0
    assert price_provider._cached_alpha_rate == 0.0025


def test_set_mock_prices_updates_both(price_provider, mock_time):
    """set_mock_prices() updates both caches with current timestamps."""
    # Arrange
    mock_time(1000000.0)

    # Act
    price_provider.set_mock_prices(300.0, 0.005)

    # Assert
    assert price_provider._cached_tao_price == 300.0
    assert price_provider._cached_alpha_rate == 0.005
    assert price_provider._tao_price_timestamp == 1000000.0
    assert price_provider._alpha_rate_timestamp == 1000000.0


def test_get_cache_status_returns_complete_state(price_provider, mock_time):
    """get_cache_status() returns complete cache state."""
    # Arrange
    mock_time(1000000.0)
    price_provider._cached_tao_price = 250.0
    price_provider._tao_price_timestamp = 999500.0  # 500s ago
    price_provider._cached_alpha_rate = 0.002
    price_provider._alpha_rate_timestamp = 999700.0  # 300s ago

    # Act
    status = price_provider.get_cache_status()

    # Assert
    assert status["tao_price"]["value"] == 250.0
    assert status["tao_price"]["timestamp"] == 999500.0
    assert status["tao_price"]["age_seconds"] == 500.0
    assert status["tao_price"]["is_valid"] is True

    assert status["alpha_rate"]["value"] == 0.002
    assert status["alpha_rate"]["timestamp"] == 999700.0
    assert status["alpha_rate"]["age_seconds"] == 300.0
    assert status["alpha_rate"]["is_valid"] is True

    assert status["cache_ttl_seconds"] == 900


def test_get_cache_status_handles_none_values(price_provider):
    """get_cache_status() handles empty cache gracefully."""
    # Arrange: No cache set

    # Act
    status = price_provider.get_cache_status()

    # Assert
    assert status["tao_price"]["value"] is None
    assert status["tao_price"]["timestamp"] is None
    assert status["tao_price"]["age_seconds"] is None
    assert status["tao_price"]["is_valid"] is False

    assert status["alpha_rate"]["value"] is None
    assert status["alpha_rate"]["timestamp"] is None
    assert status["alpha_rate"]["age_seconds"] is None
    assert status["alpha_rate"]["is_valid"] is False


# ============================================================================
# INDIVIDUAL PROVIDER TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_coingecko_provider_get_rate_success(mock_aiohttp_session):
    """CoinGeckoProvider.get_rate() returns float value."""
    # Arrange
    _, response = mock_aiohttp_session
    response.json.return_value = {"market_data": {"current_price": {"usd": 255.0}}}

    provider = CoinGeckoProvider()

    # Act
    result = await provider.get_rate()

    # Assert
    assert isinstance(result, float)
    assert result == 255.0


@pytest.mark.asyncio
async def test_coingecko_provider_http_error(mock_aiohttp_session):
    """CoinGeckoProvider raises exception on HTTP error."""
    # Arrange
    _, response = mock_aiohttp_session
    response.raise_for_status.side_effect = aiohttp.ClientResponseError(
        request_info=MagicMock(),
        history=(),
        status=500,
        message="Internal Server Error"
    )

    provider = CoinGeckoProvider()

    # Act & Assert
    with pytest.raises(aiohttp.ClientResponseError):
        await provider.get_rate()


@pytest.mark.asyncio
async def test_coinbase_provider_get_rate_success(mock_aiohttp_session):
    """CoinbaseProvider.get_rate() parses string to float."""
    # Arrange
    _, response = mock_aiohttp_session
    response.json.return_value = {"data": {"amount": "248.99"}}

    provider = CoinbaseProvider()

    # Act
    result = await provider.get_rate()

    # Assert
    assert isinstance(result, float)
    assert result == 248.99


@pytest.mark.asyncio
async def test_cryptocompare_provider_get_rate_success(mock_aiohttp_session):
    """CryptoCompareProvider.get_rate() returns USD value."""
    # Arrange
    _, response = mock_aiohttp_session
    response.json.return_value = {"USD": 251.45}

    provider = CryptoCompareProvider()

    # Act
    result = await provider.get_rate()

    # Assert
    assert isinstance(result, float)
    assert result == 251.45


# ============================================================================
# DATA INTEGRITY TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_tao_price_always_returns_float(price_provider, mock_aiohttp_session):
    """TAO price always returns float type."""
    # Arrange
    _, response = mock_aiohttp_session
    response.json.return_value = {"market_data": {"current_price": {"usd": 245.0}}}

    # Act
    with patch("asyncio.sleep", new_callable=AsyncMock):
        result = await price_provider.get_tao_price()

    # Assert
    assert isinstance(result, float)


@pytest.mark.asyncio
async def test_alpha_rate_always_returns_float(price_provider, mock_subtensor, mock_settings):
    """Alpha rate always returns float type."""
    # Arrange
    _ = mock_settings  # Ensure settings are mocked
    price_obj = MagicMock()
    price_obj.tao = 0.0015
    mock_subtensor.get_subnet_price.return_value = price_obj

    # Act
    with patch("asyncio.sleep", new_callable=AsyncMock):
        result = await price_provider.get_alpha_rate()

    # Assert
    assert isinstance(result, float)


@pytest.mark.asyncio
async def test_tao_price_never_returns_none(price_provider, mock_aiohttp_session):
    """TAO price never returns None, even on failures."""
    # Arrange: Force failure
    _, response = mock_aiohttp_session
    response.json = AsyncMock(side_effect=aiohttp.ClientError("All down"))

    # Act
    with patch("asyncio.sleep", new_callable=AsyncMock):
        result = await price_provider.get_tao_price()

    # Assert
    assert result is not None
    assert isinstance(result, float)


@pytest.mark.asyncio
async def test_alpha_rate_never_returns_none(price_provider, mock_subtensor, mock_settings):
    """Alpha rate never returns None, even on failures."""
    # Arrange: Force failure
    _ = mock_settings  # Ensure settings are mocked
    mock_subtensor.get_subnet_price.side_effect = Exception("Subtensor down")

    # Act
    with patch("asyncio.sleep", new_callable=AsyncMock):
        result = await price_provider.get_alpha_rate()

    # Assert
    assert result is not None
    assert isinstance(result, float)


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
