"""
Shared test configuration for the miner test suite.

Sets up required environment variables before any miner src modules are
imported, avoiding import-time errors from Settings() validation and
SQLite engine creation.
"""

import os
import sys

# Required env vars consumed by core.config.Settings at import time.
os.environ.setdefault("BITTENSOR_WALLET_NAME", "test_wallet")
os.environ.setdefault("BITTENSOR_WALLET_HOTKEY_NAME", "test_hotkey")
os.environ.setdefault("SQLALCHEMY_DATABASE_URI", "sqlite:///tmp/test_miner.db")
os.environ.setdefault("EXTERNAL_IP_ADDRESS", "127.0.0.1")

# Add src/ to sys.path so tests can import miner modules directly.
_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
if _src not in sys.path:
    sys.path.insert(0, _src)
