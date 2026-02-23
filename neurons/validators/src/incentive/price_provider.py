"""
PriceProvider service for fetching TAO price and alpha rate with caching.

This module provides a service class that fetches real-time TAO price from CoinGecko API
and alpha rate (TAO emission rate per block) from the subtensor blockchain. It implements
a time-based caching mechanism (15-minute TTL) to minimize API calls and reduce load on
external services.
"""

import time
import logging
from typing import Optional
import asyncio
import aiohttp

from bittensor import AsyncSubtensor
from core.config import settings
from core.utils import get_logger, _m
from clients.subtensor_client import SubtensorClient
from tenacity import before_sleep_log, retry, retry_if_exception_type, stop_after_attempt, wait_exponential

# Default fallback values when fetch/query fails and no cache exists
# These values are based on the worked example in SPEC.md and provide
# reasonable defaults to ensure the system can continue operating
DEFAULT_TAO_PRICE = 200.0  # USD
DEFAULT_ALPHA_RATE = 0.001  # TAO emission rate per block

logger = get_logger(__name__)


class CoinbaseProvider:
    name = "Coinbase"

    async def get_rate(self) -> float:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://api.coinbase.com/v2/prices/TAO-USD/spot",
                timeout=10,
            ) as response:
                response.raise_for_status()
                rate_str = (await response.json())["data"]["amount"]
                return float(rate_str)


class CoinGeckoProvider:
    name = "CoinGecko"

    async def get_rate(self) -> float:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://api.coingecko.com/api/v3/coins/bittensor",
                timeout=10,
            ) as response:
                response.raise_for_status()
                data = await response.json()
                rate_float = data["market_data"]["current_price"]["usd"]
                return float(rate_float)


class CryptoCompareProvider:
    name = "CryptoCompare"

    async def get_rate(self) -> float:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://min-api.cryptocompare.com/data/price?fsym=TAO&tsyms=USD",
                timeout=10,
            ) as response:
                response.raise_for_status()
                rate_float = (await response.json())["USD"]
                return float(rate_float)


class PriceProvider:
    """
    Provides TAO price and alpha rate data with caching and fault tolerance.

    The provider fetches TAO price from CoinGecko REST API and queries alpha rate
    from subtensor blockchain. Both values are cached for 15 minutes to minimize
    external API calls. On fetch failures, the provider falls back to cached values
    (even if expired) or returns default values if no cache exists.
    """

    providers = [CoinGeckoProvider(), CoinbaseProvider(), CryptoCompareProvider()]

    def __init__(self):
        """Initialize the price provider with empty cache."""
        # TAO price cache
        self._cached_tao_price: Optional[float] = None
        self._tao_price_timestamp: Optional[float] = None

        # Alpha rate cache
        self._cached_alpha_rate: Optional[float] = None
        self._alpha_rate_timestamp: Optional[float] = None

        # Cache TTL: 15 minutes in seconds
        self._cache_ttl: int = 900

        self.subtensor_client = SubtensorClient.get_instance()

    def _is_cache_valid(self, timestamp: Optional[float]) -> bool:
        """
        Check if a cached value is still valid based on its timestamp.

        Args:
            timestamp: The timestamp when the value was cached

        Returns:
            True if the cache is valid (within TTL), False otherwise
        """
        if timestamp is None:
            return False

        current_time = time.time()
        return (current_time - timestamp) < self._cache_ttl

    @retry(
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((aiohttp.ClientError,)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _get_current_rate(self) -> float:
        last_error = None

        for provider in self.providers:
            try:
                rate = await provider.get_rate()
                logger.info(f"Successfully fetched TAO rate from {provider.name}: {rate}")
                return rate
            except Exception as e:
                logger.warning(f"{provider.name} failed: {e}")
                last_error = e
                continue

        raise last_error or Exception("All currency providers failed")

    @retry(
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((Exception,)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _get_alpha_rate(self) -> float:
        return await asyncio.to_thread(self.subtensor_client.get_alpha_rate)

        # subtensor = AsyncSubtensor(config=settings.get_bittensor_config())
        # await subtensor.initialize()
        # price = await subtensor.get_subnet_price(netuid=settings.BITTENSOR_NETUID)
        # await subtensor.close()
        # return price.tao

    async def get_tao_price(self) -> Optional[float]:
        """
        Get the current TAO price in USD.

        Fetches from CoinGecko API with 15-minute caching. On fetch failures,
        falls back to cached value (even if expired) or returns default value if no cache exists.

        Returns:
            TAO price in USD (float, always returns a value)
        """
        # Check cache validity
        if self._is_cache_valid(self._tao_price_timestamp):
            logger.debug(
                _m(
                    "TAO price cache hit",
                    extra={"tao_price": self._cached_tao_price, "cache_age_seconds": time.time() - self._tao_price_timestamp}
                )
            )
            return self._cached_tao_price

        # Cache invalid or missing, fetch from API
        try:
            tao_price = await self._get_current_rate()
            self._cached_tao_price = tao_price
            self._tao_price_timestamp = time.time()
            logger.info(
                _m(
                    "Successfully fetched TAO price from CoinGecko",
                    extra={"tao_price": tao_price}
                )
            )
            return tao_price
        except Exception as e:
            logger.error(
                _m(
                    "Failed to fetch TAO price from CoinGecko API",
                    extra={"error": str(e), "error_type": type(e).__name__}
                )
            )

            # Fallback to cached value if it exists (even if expired)
            if self._cached_tao_price is not None:
                cache_age = time.time() - self._tao_price_timestamp if self._tao_price_timestamp else None
                logger.warning(
                    _m(
                        "Falling back to expired TAO price cache",
                        extra={
                            "tao_price": self._cached_tao_price,
                            "cache_age_seconds": cache_age
                        }
                    )
                )
                return self._cached_tao_price

            # No cache available, return default value
            logger.warning(
                _m(
                    "No TAO price cache available, falling back to default value",
                    extra={"default_tao_price": DEFAULT_TAO_PRICE}
                )
            )
            return DEFAULT_TAO_PRICE

    async def get_alpha_rate(self) -> Optional[float]:
        """
        Get the current alpha rate (TAO emission rate per block) for the subnet.

        Queries from subtensor blockchain with 15-minute caching. On query failures,
        falls back to cached value (even if expired) or returns default value if no cache exists.

        Returns:
            Alpha rate (TAO per block) (float, always returns a value)
        """
        # Check cache validity
        if self._is_cache_valid(self._alpha_rate_timestamp):
            logger.debug(
                _m(
                    "Alpha rate cache hit",
                    extra={"alpha_rate": self._cached_alpha_rate, "cache_age_seconds": time.time() - self._alpha_rate_timestamp}
                )
            )
            return self._cached_alpha_rate

        # Cache invalid or missing, query from subtensor
        try:
            alpha_rate = await self._get_alpha_rate()

            # update cache 
            self._cached_alpha_rate = alpha_rate
            self._alpha_rate_timestamp = time.time()

            logger.info(
                _m("Successfully queried alpha rate from subtensor", extra={"price": alpha_rate})
            )
            return alpha_rate
        except Exception as e:
            logger.error(
                _m(
                    "Failed to query alpha rate from subtensor",
                    extra={
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "netuid": settings.BITTENSOR_NETUID
                    }
                )
            )

            # Fallback to cached value if it exists (even if expired)
            if self._cached_alpha_rate is not None:
                cache_age = time.time() - self._alpha_rate_timestamp if self._alpha_rate_timestamp else None
                logger.warning(
                    _m(
                        "Falling back to expired alpha rate cache",
                        extra={
                            "alpha_rate": self._cached_alpha_rate,
                            "cache_age_seconds": cache_age
                        }
                    )
                )
                return self._cached_alpha_rate

            # No cache available, return default value
            logger.warning(
                _m(
                    "No alpha rate cache available, falling back to default value",
                    extra={"default_alpha_rate": DEFAULT_ALPHA_RATE}
                )
            )
            return DEFAULT_ALPHA_RATE

    async def refresh_cache(self) -> None:
        """
        Force refresh both TAO price and alpha rate caches.

        This method clears the cache and fetches fresh data from external sources.
        Useful for testing or manual cache refresh.
        """
        logger.info(_m("Force refreshing PriceProvider cache"))

        # Clear cache to force fresh fetch
        self.clear_cache()

        # Fetch fresh data
        await self.get_tao_price()
        await self.get_alpha_rate()

    def clear_cache(self) -> None:
        """
        Clear all cached values and timestamps.

        This method resets the cache to its initial empty state.
        """
        self._cached_tao_price = None
        self._tao_price_timestamp = None
        self._cached_alpha_rate = None
        self._alpha_rate_timestamp = None

        logger.debug(_m("PriceProvider cache cleared"))

    def set_mock_prices(self, tao_price: float, alpha_rate: float) -> None:
        """
        Set cached values directly for testing purposes.

        Args:
            tao_price: Mock TAO price in USD
            alpha_rate: Mock alpha rate (TAO per block)
        """
        self._cached_tao_price = tao_price
        self._tao_price_timestamp = time.time()
        self._cached_alpha_rate = alpha_rate
        self._alpha_rate_timestamp = time.time()

        logger.debug(
            _m(
                "Mock prices set for testing",
                extra={"tao_price": tao_price, "alpha_rate": alpha_rate}
            )
        )

    def get_cache_status(self) -> dict:
        """
        Get the current cache state for debugging.

        Returns:
            Dictionary containing cache values, timestamps, and validity status
        """
        current_time = time.time()

        return {
            "tao_price": {
                "value": self._cached_tao_price,
                "timestamp": self._tao_price_timestamp,
                "age_seconds": current_time - self._tao_price_timestamp if self._tao_price_timestamp else None,
                "is_valid": self._is_cache_valid(self._tao_price_timestamp)
            },
            "alpha_rate": {
                "value": self._cached_alpha_rate,
                "timestamp": self._alpha_rate_timestamp,
                "age_seconds": current_time - self._alpha_rate_timestamp if self._alpha_rate_timestamp else None,
                "is_valid": self._is_cache_valid(self._alpha_rate_timestamp)
            },
            "cache_ttl_seconds": self._cache_ttl
        }
