"""
Shared test configuration for the miner test suite.

Sets up required environment variables before any miner src modules are
imported, avoiding import-time errors from Settings() validation and
SQLite engine creation.
"""

import os
import sys
import tempfile
from unittest.mock import patch

import pytest

# Required env vars consumed by core.config.Settings at import time.
os.environ.setdefault("BITTENSOR_WALLET_NAME", "test_wallet")
os.environ.setdefault("BITTENSOR_WALLET_HOTKEY_NAME", "test_hotkey")
os.environ.setdefault("SQLALCHEMY_DATABASE_URI", "sqlite:///tmp/test_miner.db")
os.environ.setdefault("EXTERNAL_IP_ADDRESS", "127.0.0.1")
# The collateral client's record of its last send; never the real one in the home directory, also for the CLI
# processes some tests start.
os.environ.setdefault(
    "COLLATERAL_SENT_RECORD", os.path.join(tempfile.mkdtemp(prefix="collateral-"), "sent.json")
)

# Prevent network calls during module-level SharedConfigClient instantiation.
# core/config.py creates `shared_client = SharedConfigClient(...)` at import time, which
# otherwise performs a blocking HTTP fetch against the live API. Forcing _fetch to return
# None keeps the offline fallback (DEFAULT_SHARED_CONFIG) so unit tests stay hermetic.
# This patch must run before test collection imports any miner src module.
_shared_config_patcher = patch(
    "lium_core.shared_config.client.SharedConfigClient._fetch",
    return_value=None,
)
_shared_config_patcher.start()

# Add src/ to sys.path so tests can import miner modules directly.
_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
if _src not in sys.path:
    sys.path.insert(0, _src)


@pytest.fixture(autouse=True)
def _fresh_collateral_sent_record(tmp_path, monkeypatch):
    from core import collateral
    from core.config import settings

    path = tmp_path / "collateral-sent.json"
    monkeypatch.setattr(collateral, "SENT_RECORD_PATH", path)
    monkeypatch.setattr(settings, "COLLATERAL_SENT_RECORD", str(path))
