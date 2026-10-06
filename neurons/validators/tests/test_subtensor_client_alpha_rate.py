"""Regression tests for SubtensorClient.get_alpha_rate.

Background: the finney runtime migrated its `Swap` pallet to a new AMM and removed the
`Swap.AlphaSqrtPrice` storage item. `subtensor.get_subnet_price()` reads that item directly
and started raising `Storage function "Swap.AlphaSqrtPrice" not found` in production. The fix
routes through `subtensor.subnet()`, which falls back to the `tao_in / alpha_in` reserve ratio
when the storage query fails. These tests exercise the method against a mocked subtensor so the
heavy SubtensorClient init is bypassed.
"""

from unittest.mock import MagicMock

import pytest

from clients.subtensor_client import SubtensorClient


def _client_with_subnet(price_obj):
    """Build a stand-in `self` whose `.subtensor.subnet()` returns a configured object."""
    fake_self = MagicMock()
    fake_self.netuid = 51
    fake_self.subtensor.subnet.return_value = price_obj
    return fake_self


def test_get_alpha_rate_returns_subnet_price_tao():
    # Arrange: subnet() returns a DynamicInfo-like object with a Balance price
    price = MagicMock()
    price.tao = 0.054316
    subnet = MagicMock()
    subnet.price = price
    fake_self = _client_with_subnet(subnet)

    # Act
    result = SubtensorClient.get_alpha_rate(fake_self)

    # Assert: reads the alpha price in TAO and queries the configured netuid
    assert result == 0.054316
    fake_self.subtensor.subnet.assert_called_once_with(netuid=51)


def test_get_alpha_rate_raises_when_subnet_missing():
    # Arrange: subnet() returns None (subnet not found / decode failure)
    fake_self = _client_with_subnet(None)

    # Act / Assert: surfaces an error so the caller's retry+fallback path engages
    with pytest.raises(RuntimeError):
        SubtensorClient.get_alpha_rate(fake_self)


def test_get_alpha_rate_raises_when_price_missing():
    # Arrange: subnet() returns an object without a usable price
    subnet = MagicMock()
    subnet.price = None
    fake_self = _client_with_subnet(subnet)

    # Act / Assert
    with pytest.raises(RuntimeError):
        SubtensorClient.get_alpha_rate(fake_self)
