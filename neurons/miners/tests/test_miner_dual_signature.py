#!/usr/bin/env python3
"""Exercise miner dual-signature forwarding logic with real bittensor keys."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Awaitable, Callable, List, Optional, Sequence

import types

# ---------------------------------------------------------------------------
# Environment bootstrap
# ---------------------------------------------------------------------------

os.environ.setdefault("BITTENSOR_WALLET_NAME", "test-wallet")
os.environ.setdefault("BITTENSOR_WALLET_HOTKEY_NAME", "test-hotkey")
os.environ.setdefault("SQLALCHEMY_DATABASE_URI", "sqlite:///tmp/miner-test.db")
os.environ.setdefault("EXTERNAL_IP_ADDRESS", "127.0.0.1")
os.environ.setdefault("BT_LOGGING", "DISABLED")
os.environ.setdefault("BT_LOGGING_CONSOLE", "False")

REPO_ROOT = Path(__file__).resolve().parents[3]
MINER_SRC = REPO_ROOT / "neurons" / "miners" / "src"
DATURA_SRC = REPO_ROOT / "datura"
for path in (REPO_ROOT, MINER_SRC, DATURA_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

if "celium_collateral_contracts" not in sys.modules:
    celium_stub = types.ModuleType("celium_collateral_contracts")

    class CollateralContract:  # minimal placeholder
        def __init__(self, *_, **__):
            ...

    celium_stub.CollateralContract = CollateralContract
    sys.modules["celium_collateral_contracts"] = celium_stub

logging.getLogger("neurons.miners.src.services.executor_service").setLevel(logging.CRITICAL)

try:
    import bittensor
except ModuleNotFoundError as missing_bittensor:  # pragma: no cover
    sys.stderr.write(
        "Missing dependency 'bittensor'. Install it with `.venv/bin/python -m pip install bittensor`.\n"
    )
    raise SystemExit(1) from missing_bittensor

from neurons.miners.src.services import executor_service as executor_service_module  # type: ignore
from neurons.miners.src.services.executor_service import ExecutorService  # type: ignore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TEST_PUBLIC_KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAMinerTestKey miner@example"

VALIDATOR_SEED = "0x" + "11" * 32
UNAUTHORIZED_SEED = "0x" + "22" * 32
MINER_SEED = "0x" + "33" * 32

def _create_keypair(seed_hex: str) -> bittensor.Keypair:
    token = seed_hex[2:] if seed_hex.startswith("0x") else seed_hex
    seed_bytes = bytes.fromhex(token)
    try:
        return bittensor.Keypair.create_from_seed(seed_bytes)
    except TypeError:
        return bittensor.Keypair.create_from_seed(seed_hex)


VALIDATOR_KEYPAIR = _create_keypair(VALIDATOR_SEED)
UNAUTHORIZED_KEYPAIR = _create_keypair(UNAUTHORIZED_SEED)
MINER_KEYPAIR = _create_keypair(MINER_SEED)


class WalletStub:
    def __init__(self, keypair: bittensor.Keypair) -> None:
        self._keypair = keypair

    def get_hotkey(self) -> bittensor.Keypair:
        return self._keypair


def patch_settings_stub():
    original_settings = executor_service_module.settings
    stub = SimpleNamespace(
        get_bittensor_wallet=lambda: WalletStub(MINER_KEYPAIR),
        MINER_HOTKEY_SS58_ADDRESS=MINER_KEYPAIR.ss58_address,
        DEFAULT_MINER_HOTKEY=MINER_KEYPAIR.ss58_address,
    )
    executor_service_module.settings = stub
    return original_settings


def make_executor() -> SimpleNamespace:
    return SimpleNamespace(
        address="10.0.0.5",
        port=8001,
        uuid=uuid.uuid4(),
        price_per_hour=0.42,
    )


class FakeResponse:
    def __init__(self, status: int, payload: dict):
        self.status = status
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def json(self):
        return self._payload


class FakeSession:
    def __init__(self, captured: dict, status: int, payload: dict, *_, **__):
        self._captured = captured
        self._status = status
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def post(self, url: str, json: dict):
        self._captured["url"] = url
        self._captured["payload"] = json
        return FakeResponse(self._status, self._payload)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


async def send_pubkey_includes_validator_signature() -> None:
    captured: dict = {}
    response_payload = {
        "ssh_username": "miner",
        "ssh_port": 2200,
        "python_path": "/usr/bin/python3",
        "root_dir": "/srv/executor",
        "port_range": None,
        "port_mappings": None,
    }

    original_settings = patch_settings_stub()
    original_session = executor_service_module.aiohttp.ClientSession
    executor_service_module.aiohttp.ClientSession = lambda *a, **kw: FakeSession(
        captured, 200, response_payload
    )

    try:
        service = ExecutorService(SimpleNamespace())
        executor = make_executor()
        validator_sig = f"0x{VALIDATOR_KEYPAIR.sign(TEST_PUBLIC_KEY).hex()}"
        result = await service.send_pubkey_to_executor(executor, TEST_PUBLIC_KEY, validator_sig)

        assert result is not None, captured
        assert captured["payload"]["public_key"] == TEST_PUBLIC_KEY, captured
        assert captured["payload"]["data_to_sign"] == TEST_PUBLIC_KEY, captured
        assert MINER_KEYPAIR.verify(TEST_PUBLIC_KEY, captured["payload"]["signature"]), captured
        assert validator_sig == captured["payload"]["validator_signature"], captured
        assert result.price == executor.price_per_hour
        assert result.address == executor.address
        assert result.port == executor.port
    finally:
        executor_service_module.settings = original_settings
        executor_service_module.aiohttp.ClientSession = original_session


async def send_pubkey_omits_validator_signature() -> None:
    captured: dict = {}
    response_payload = {
        "ssh_username": "miner",
        "ssh_port": 2200,
        "python_path": "/usr/bin/python3",
        "root_dir": "/srv/executor",
        "port_range": None,
        "port_mappings": None,
    }

    original_settings = patch_settings_stub()
    original_session = executor_service_module.aiohttp.ClientSession
    executor_service_module.aiohttp.ClientSession = lambda *a, **kw: FakeSession(
        captured, 200, response_payload
    )

    try:
        service = ExecutorService(SimpleNamespace())
        executor = make_executor()
        result = await service.send_pubkey_to_executor(executor, TEST_PUBLIC_KEY, None)

        assert result is not None, captured
        assert "validator_signature" not in captured["payload"], captured
        assert MINER_KEYPAIR.verify(TEST_PUBLIC_KEY, captured["payload"]["signature"]), captured
    finally:
        executor_service_module.settings = original_settings
        executor_service_module.aiohttp.ClientSession = original_session


async def send_pubkey_handles_http_error() -> None:
    captured: dict = {}
    original_settings = patch_settings_stub()
    original_session = executor_service_module.aiohttp.ClientSession
    executor_service_module.aiohttp.ClientSession = lambda *a, **kw: FakeSession(
        captured, 500, {}
    )

    try:
        service = ExecutorService(SimpleNamespace())
        executor = make_executor()
        validator_sig = f"0x{VALIDATOR_KEYPAIR.sign(TEST_PUBLIC_KEY).hex()}"
        result = await service.send_pubkey_to_executor(executor, TEST_PUBLIC_KEY, validator_sig)
        assert result is None
    finally:
        executor_service_module.settings = original_settings
        executor_service_module.aiohttp.ClientSession = original_session


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


async def main() -> int:
    checks: Sequence[tuple[str, Callable[[], Awaitable[None]]]] = (
        ("send-pubkey-includes-validator-signature", send_pubkey_includes_validator_signature),
        ("send-pubkey-omits-validator-signature", send_pubkey_omits_validator_signature),
        ("send-pubkey-handles-http-error", send_pubkey_handles_http_error),
    )

    failures: List[str] = []
    for name, check in checks:
        try:
            await check()
            print(f"[PASS] {name}")
        except AssertionError as err:
            failures.append(f"{name}: {err}")
            print(f"[FAIL] {name}: {err}")
        except Exception as err:  # pragma: no cover - unexpected errors
            failures.append(f"{name}: {err}")
            print(f"[ERROR] {name}: {err}")

    if failures:
        print("\nChecks failed:")
        for failure in failures:
            print(f" - {failure}")
        return 1

    print("\nAll miner dual-signature checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
