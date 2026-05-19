"""
Shared test configuration for the executor test suite.

Sets up module-level mocks and required environment variables before any
executor src modules are imported, avoiding import-time side effects from
heavy system dependencies (docker, etc.).
"""

import os
import sys
import types
from unittest.mock import MagicMock

# Mock docker before any src import so that routes/apis.py and related
# services can be imported without a running Docker daemon.
sys.modules["docker"] = MagicMock()
sys.modules["pynvml"] = MagicMock()
sys.modules["pkg_resources"] = MagicMock()
sys.modules["dstack_sdk"] = MagicMock()


class _FakeSession:
    def __init__(self, *_args, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class _FakeSQLModel:
    metadata = types.SimpleNamespace(create_all=lambda *_args, **_kwargs: None)

    def __init_subclass__(cls, **_kwargs):
        super().__init_subclass__()


def _fake_field(*_args, **kwargs):
    if "default_factory" in kwargs:
        return kwargs["default_factory"]()
    return kwargs.get("default")


def _fake_create_engine(*_args, **_kwargs):
    return object()


def _fake_select(*_args, **_kwargs):
    return object()


sys.modules["sqlmodel"] = types.SimpleNamespace(
    Field=_fake_field,
    SQLModel=_FakeSQLModel,
    Session=_FakeSession,
    create_engine=_fake_create_engine,
    select=_fake_select,
)

# Required env vars consumed by core.config.Settings at import time.
os.environ.setdefault("MINER_HOTKEY_SS58_ADDRESS", "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY")
os.environ.setdefault("DB_URI", "sqlite:///tmp/test.db")

# Add src/ to sys.path so tests can import executor modules directly.
_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
if _src not in sys.path:
    sys.path.insert(0, _src)

# Add repo root so local workspace dependencies like datura/ resolve in tests.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

_datura_root = os.path.join(_repo_root, "datura")
if _datura_root not in sys.path:
    sys.path.insert(0, _datura_root)
