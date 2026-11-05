import json
from unittest.mock import MagicMock, patch

import pytest

from lium_core.shared_config.client import SharedConfigClient
from lium_core.shared_config.defaults import DEFAULT_SHARED_CONFIG
from lium_core.shared_config.model import SharedConfig

API_URL = "http://fake-api/config"


def _build_client(mock_get: MagicMock) -> SharedConfigClient:
    """Create client with patched threading so no background loop runs."""
    with patch("lium_core.shared_config.client.threading"):
        client = SharedConfigClient(api_url=API_URL, refresh_interval=60)
    return client


# ==================== Model tests ====================


@pytest.mark.parametrize(
    "payment_minimums",
    [
        pytest.param({}, id="defaults_when_absent"),
        pytest.param({"miner_payment_min_tao": 0.003, "miner_payment_min_usd": 0.75}, id="supplied_values"),
    ],
)
def test_miner_payment_minimums_validate_from_json(payment_minimums: dict[str, float]) -> None:
    payload = {
        key: value
        for key, value in DEFAULT_SHARED_CONFIG.model_dump().items()
        if key not in {"miner_payment_min_tao", "miner_payment_min_usd"}
    }
    config = SharedConfig.model_validate_json(json.dumps({**payload, **payment_minimums}))

    assert config.miner_payment_min_tao == payment_minimums.get("miner_payment_min_tao", 0.0021)
    assert config.miner_payment_min_usd == payment_minimums.get("miner_payment_min_usd", 0.5)


# ==================== Utils tests (dict_diff) ====================


# ==================== Client tests: _fetch ====================


# ==================== Client tests: __init__ ====================


def test_init_fallback_to_default() -> None:
    with patch("lium_core.shared_config.client.requests.get", side_effect=Exception("boom")):
        client = _build_client(MagicMock())

    assert client.config is DEFAULT_SHARED_CONFIG


# ==================== Client tests: _refresh_loop ====================


# ==================== Client tests: .config property ====================


