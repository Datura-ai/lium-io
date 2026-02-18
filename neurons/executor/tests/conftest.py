"""
Shared test configuration for the executor test suite.

Sets up module-level mocks and required environment variables before any
executor src modules are imported, avoiding import-time side effects from
heavy system dependencies (docker, etc.).
"""

import os
import sys
from unittest.mock import MagicMock

# Mock docker before any src import so that routes/apis.py and related
# services can be imported without a running Docker daemon.
sys.modules["docker"] = MagicMock()

# Required env vars consumed by core.config.Settings at import time.
os.environ.setdefault("MINER_HOTKEY_SS58_ADDRESS", "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY")
os.environ.setdefault("DB_URI", "sqlite:///tmp/test.db")

# Add src/ to sys.path so tests can import executor modules directly.
_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
if _src not in sys.path:
    sys.path.insert(0, _src)
