import logging
from unittest.mock import MagicMock, patch

import pydantic
import pytest
import requests

from lium_core.shared_config.client import SharedConfigClient
from lium_core.shared_config.defaults import DEFAULT_SHARED_CONFIG
from lium_core.shared_config.model import SharedConfig
from lium_core.shared_config.utils import dict_diff

API_URL = "http://fake-api/config"

SAMPLE_CONFIG_DATA = DEFAULT_SHARED_CONFIG.model_dump()

ALTERED_CONFIG_DATA = {
    **SAMPLE_CONFIG_DATA,
    "rental_fees_rate": 0.75,
    "collateral_days": 14,
}


def _make_response(json_data: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status.return_value = None
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(response=resp)
    return resp


def _build_client(mock_get: MagicMock) -> SharedConfigClient:
    """Create client with patched threading so no background loop runs."""
    with patch("lium_core.shared_config.client.threading"):
        client = SharedConfigClient(api_url=API_URL, refresh_interval=60)
    return client


# ==================== Model tests ====================


def test_shared_config_is_frozen() -> None:
    with pytest.raises(pydantic.ValidationError):
        DEFAULT_SHARED_CONFIG.rental_fees_rate = 0.5


def test_default_shared_config_has_all_fields() -> None:
    assert len(DEFAULT_SHARED_CONFIG.machine_prices) > 0
    assert len(DEFAULT_SHARED_CONFIG.required_deposit_amount) > 0
    assert len(DEFAULT_SHARED_CONFIG.gpu_architectures) > 0
    assert len(DEFAULT_SHARED_CONFIG.driver_cuda_map) > 0
    assert DEFAULT_SHARED_CONFIG.machine_max_price_rate == 3.0
    assert DEFAULT_SHARED_CONFIG.machine_min_price_rate == 0.5
    assert DEFAULT_SHARED_CONFIG.rental_fees_rate == 0.9
    assert DEFAULT_SHARED_CONFIG.collateral_days == 7
    assert DEFAULT_SHARED_CONFIG.collateral_contract_address == "0x7DCCb5659c70Ce2104A9bb79E9E257473ECbe628"
    assert DEFAULT_SHARED_CONFIG.bittensor_netuid == 51
    assert DEFAULT_SHARED_CONFIG.volume_gb_hour_price_usd == 0.00005
    assert DEFAULT_SHARED_CONFIG.max_initial_port_count == 200
    assert DEFAULT_SHARED_CONFIG.total_burn_emission == 0.91


def test_shared_config_serializes_to_json() -> None:
    data = DEFAULT_SHARED_CONFIG.model_dump()
    assert isinstance(data, dict)
    assert "machine_prices" in data
    assert "gpu_architectures" in data
    assert isinstance(data["gpu_architectures"]["NVIDIA B200"]["arch"], str)


# ==================== Utils tests (dict_diff) ====================


@pytest.mark.parametrize(
    "old, new, expected",
    [
        pytest.param({}, {}, [], id="both_empty"),
        pytest.param({"a": 1}, {"a": 1}, [], id="no_changes"),
        pytest.param(
            {"a": 1},
            {"a": 2},
            ["[a]: 1 -> 2"],
            id="top_level_change",
        ),
        pytest.param(
            {"a": {"b": 1}},
            {"a": {"b": 2}},
            ["[a.b]: 1 -> 2"],
            id="nested_change",
        ),
        pytest.param(
            {"a": {"b": {"c": 1}}},
            {"a": {"b": {"c": 99}}},
            ["[a.b.c]: 1 -> 99"],
            id="deep_nested_change",
        ),
        pytest.param(
            {},
            {"a": 1},
            ["[a]: None -> 1"],
            id="key_added",
        ),
        pytest.param(
            {"a": 1},
            {},
            ["[a]: 1 -> None"],
            id="key_removed",
        ),
        pytest.param(
            {"a": 1, "b": 2, "c": 3},
            {"a": 10, "b": 20, "c": 30},
            ["[a]: 1 -> 10", "[b]: 2 -> 20", "[c]: 3 -> 30"],
            id="multiple_changes_sorted",
        ),
        pytest.param(
            {"a": {"nested": 1}},
            {"a": "flat"},
            ["[a]: {'nested': 1} -> flat"],
            id="dict_becomes_scalar",
        ),
    ],
)
def test_dict_diff(old: dict, new: dict, expected: list[str]) -> None:
    assert dict_diff(old, new) == expected


# ==================== Client tests: _fetch ====================


def test_fetch_success() -> None:
    with patch("lium_core.shared_config.client.requests.get", return_value=_make_response(SAMPLE_CONFIG_DATA)):
        client = _build_client(MagicMock())

    assert isinstance(client._config, SharedConfig)
    assert client._config == DEFAULT_SHARED_CONFIG


def test_fetch_http_error() -> None:
    with patch("lium_core.shared_config.client.requests.get", return_value=_make_response({}, status_code=500)):
        client = _build_client(MagicMock())

    assert client._config == DEFAULT_SHARED_CONFIG


def test_fetch_network_error() -> None:
    with patch("lium_core.shared_config.client.requests.get", side_effect=requests.ConnectionError("no network")):
        client = _build_client(MagicMock())

    assert client._config == DEFAULT_SHARED_CONFIG


# ==================== Client tests: __init__ ====================


def test_init_with_successful_fetch() -> None:
    with patch("lium_core.shared_config.client.requests.get", return_value=_make_response(ALTERED_CONFIG_DATA)):
        client = _build_client(MagicMock())

    assert client._config.rental_fees_rate == 0.75
    assert client._config.collateral_days == 14


def test_init_fallback_to_default() -> None:
    with patch("lium_core.shared_config.client.requests.get", side_effect=Exception("boom")):
        client = _build_client(MagicMock())

    assert client._config is DEFAULT_SHARED_CONFIG


# ==================== Client tests: _refresh_loop ====================


def test_refresh_updates_config_on_change() -> None:
    mock_get = MagicMock(return_value=_make_response(SAMPLE_CONFIG_DATA))
    with patch("lium_core.shared_config.client.requests.get", mock_get):
        client = _build_client(mock_get)

    assert client._config == DEFAULT_SHARED_CONFIG

    mock_get.return_value = _make_response(ALTERED_CONFIG_DATA)

    def _stop_after_one_iteration(_interval: int) -> None:
        client._running = False

    with (
        patch("lium_core.shared_config.client.requests.get", mock_get),
        patch("lium_core.shared_config.client.time.sleep", side_effect=_stop_after_one_iteration),
    ):
        client._running = True
        client._refresh_loop()

    assert client._config.rental_fees_rate == 0.75
    assert client._config.collateral_days == 14


def test_refresh_skips_on_same_config(caplog: pytest.LogCaptureFixture) -> None:
    mock_get = MagicMock(return_value=_make_response(SAMPLE_CONFIG_DATA))
    with patch("lium_core.shared_config.client.requests.get", mock_get):
        client = _build_client(mock_get)

    def _stop_after_one_iteration(_interval: int) -> None:
        client._running = False

    with (
        patch("lium_core.shared_config.client.requests.get", mock_get),
        patch("lium_core.shared_config.client.time.sleep", side_effect=_stop_after_one_iteration),
        caplog.at_level(logging.DEBUG, logger="lium_core.shared_config.client"),
    ):
        client._running = True
        client._refresh_loop()

    assert "unchanged" in caplog.text


def test_refresh_skips_on_fetch_failure() -> None:
    mock_get = MagicMock(return_value=_make_response(SAMPLE_CONFIG_DATA))
    with patch("lium_core.shared_config.client.requests.get", mock_get):
        client = _build_client(mock_get)

    original_config = client._config
    mock_get.side_effect = requests.ConnectionError("down")

    def _stop_after_one_iteration(_interval: int) -> None:
        client._running = False

    with (
        patch("lium_core.shared_config.client.requests.get", mock_get),
        patch("lium_core.shared_config.client.time.sleep", side_effect=_stop_after_one_iteration),
    ):
        client._running = True
        client._refresh_loop()

    assert client._config is original_config


# ==================== Client tests: __getattr__ ====================


def test_getattr_delegates_to_config() -> None:
    mock_get = MagicMock(return_value=_make_response(SAMPLE_CONFIG_DATA))
    with patch("lium_core.shared_config.client.requests.get", mock_get):
        client = _build_client(mock_get)

    assert client.bittensor_netuid == DEFAULT_SHARED_CONFIG.bittensor_netuid
    assert client.rental_fees_rate == DEFAULT_SHARED_CONFIG.rental_fees_rate
    assert client.machine_prices == DEFAULT_SHARED_CONFIG.machine_prices
