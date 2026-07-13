"""
Shared test configuration for the executor test suite.

Sets up module-level mocks and required environment variables before any
executor src modules are imported, avoiding import-time side effects from
heavy system dependencies (docker, etc.).
"""

import os
import sys
import tempfile
from unittest.mock import MagicMock

# Mock docker before any src import so that routes/apis.py and related
# services can be imported without a running Docker daemon.
sys.modules["docker"] = MagicMock()

# Required env vars consumed by core.config.Settings at import time.
os.environ.setdefault("MINER_HOTKEY_SS58_ADDRESS", "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY")
# Absolute temp path: core.db creates the engine (with QueuePool args, which
# in-memory sqlite rejects) and runs create_all at import time, and the old
# relative "tmp/test.db" path broke any test importing core.db (e.g. monitor).
os.environ.setdefault("DB_URI", f"sqlite:///{os.path.join(tempfile.gettempdir(), 'executor-test.db')}")

# Add src/ to sys.path so tests can import executor modules directly.
_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
if _src not in sys.path:
    sys.path.insert(0, _src)
