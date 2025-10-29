#!/usr/bin/env python3
"""Exercise executor dual-signature SSH key handling with real keypairs.

This script drives the `MinerService.upload_ssh_key` flow and validates that
validator signatures are required (and correctly verified) whenever the
`VALIDATOR_HOTKEY_SS58_ADDRESS` setting is configured. The checks use
deterministically generated bittensor keypairs so they run entirely offline but
still exercise the real cryptography.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Awaitable, Callable, List, Sequence


# ---------------------------------------------------------------------------
# Environment bootstrap
# ---------------------------------------------------------------------------

os.environ.setdefault("MINER_HOTKEY_SS58_ADDRESS", "5TestMinerHotkeyAddress")
os.environ.setdefault("DB_URI", "sqlite:///tmp/test.db")

REPO_ROOT = Path(__file__).resolve().parents[3]
EXECUTOR_SRC = REPO_ROOT / "neurons" / "executor" / "src"
for path in (REPO_ROOT, EXECUTOR_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

logging.getLogger("neurons.executor.src.services.miner_service").setLevel(logging.CRITICAL)
middleware_logger = logging.getLogger("neurons.executor.src.middlewares.miner")
middleware_logger.setLevel(logging.CRITICAL)
middleware_logger.propagate = False
middleware_logger.disabled = True

try:
    from fastapi import HTTPException  # type: ignore
except ModuleNotFoundError as missing_fastapi:  # pragma: no cover
    sys.stderr.write(
        "Missing dependency 'fastapi'. Install project requirements or run inside the venv.\n"
    )
    raise SystemExit(1) from missing_fastapi

try:
    import bittensor
except ModuleNotFoundError as missing_bittensor:  # pragma: no cover
    sys.stderr.write(
        "Missing dependency 'bittensor'. Install it with `.venv/bin/python -m pip install bittensor`.\n"
    )
    raise SystemExit(1) from missing_bittensor

from core.config import settings  # type: ignore
from neurons.executor.src.middlewares.miner import MinerMiddleware  # type: ignore
from neurons.executor.src.payloads.miner import UploadSShKeyPayload  # type: ignore
from neurons.executor.src.services.miner_service import MinerService  # type: ignore
from starlette.requests import Request
from starlette.responses import JSONResponse


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

TEST_PUBLIC_KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAITestValidatorKey validator@example"
EXPECTED_MESSAGE = f"SSH_PUBKEY_INJECTION:{TEST_PUBLIC_KEY}"

# Deterministic seeds so the test output is stable across runs
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


class DummySSHService:
    """Minimal SSH service substitute that records injected keys."""

    def __init__(self) -> None:
        self.added_keys: List[str] = []
        self.host_key: str | None = "ssh-ed25519 AAAAC3NzaDummyHostKey executor-host"

    def add_pubkey_to_host(self, pub_key: str) -> None:
        self.added_keys.append(pub_key)

    def remove_pubkey_from_host(self, pub_key: str) -> None:  # pragma: no cover
        try:
            self.added_keys.remove(pub_key)
        except ValueError:
            pass

    def get_current_os_user(self) -> str:
        return "test-user"

    def get_host_public_key(self) -> str | None:
        return self.host_key


class DummyTDXQuoteService:
    """TDX quote stub used to avoid running external commands."""

    def __init__(self, quote: str | None = None) -> None:
        self.quote = quote
        self.calls: List[str | None] = []

    async def get_quote(self, host_key: str | None) -> str | None:
        self.calls.append(host_key)
        return self.quote


def make_payload(
    validator_signature: str | None,
    miner_signature: str | None = None,
) -> UploadSShKeyPayload:
    if miner_signature is None:
        miner_signature = f"0x{MINER_KEYPAIR.sign(TEST_PUBLIC_KEY).hex()}"
    return UploadSShKeyPayload(
        public_key=TEST_PUBLIC_KEY,
        data_to_sign=TEST_PUBLIC_KEY,
        signature=miner_signature,
        validator_signature=validator_signature,
    )


def build_request(payload: dict) -> Request:
    body = json.dumps(payload).encode("utf-8")
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "method": "POST",
        "path": "/upload_ssh_key",
        "raw_path": b"/upload_ssh_key",
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
        "scheme": "http",
    }

    async def receive() -> dict:
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(scope, receive)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


async def allows_without_validator_env() -> None:
    dummy = DummySSHService()
    service = MinerService(dummy, DummyTDXQuoteService())
    settings.MINER_HOTKEY_SS58_ADDRESS = MINER_KEYPAIR.ss58_address
    settings.DEFAULT_MINER_HOTKEY = MINER_KEYPAIR.ss58_address
    settings.VALIDATOR_HOTKEY_SS58_ADDRESS = None

    payload = make_payload(validator_signature=None)
    result = await service.upload_ssh_key(payload)

    assert dummy.added_keys == [TEST_PUBLIC_KEY]
    assert result["ssh_username"] == dummy.get_current_os_user()
    assert result["ssh_host_key"] == dummy.host_key
    assert "tdx_quote" not in result


async def accepts_valid_validator_signature() -> None:
    dummy = DummySSHService()
    service = MinerService(dummy, DummyTDXQuoteService('{"quote": "0xdeadbeef", "event_log": "[]"}'))
    settings.MINER_HOTKEY_SS58_ADDRESS = MINER_KEYPAIR.ss58_address
    settings.DEFAULT_MINER_HOTKEY = MINER_KEYPAIR.ss58_address
    settings.VALIDATOR_HOTKEY_SS58_ADDRESS = VALIDATOR_KEYPAIR.ss58_address

    signature = f"0x{VALIDATOR_KEYPAIR.sign(EXPECTED_MESSAGE).hex()}"
    payload = make_payload(validator_signature=signature)
    result = await service.upload_ssh_key(payload)

    assert dummy.added_keys == [TEST_PUBLIC_KEY]
    assert result["ssh_username"] == dummy.get_current_os_user()
    assert result["ssh_host_key"] == dummy.host_key
    assert result["tdx_quote"] == '{"quote": "0xdeadbeef", "event_log": "[]"}'


async def rejects_missing_validator_signature() -> None:
    dummy = DummySSHService()
    service = MinerService(dummy, DummyTDXQuoteService())
    settings.MINER_HOTKEY_SS58_ADDRESS = MINER_KEYPAIR.ss58_address
    settings.DEFAULT_MINER_HOTKEY = MINER_KEYPAIR.ss58_address
    settings.VALIDATOR_HOTKEY_SS58_ADDRESS = VALIDATOR_KEYPAIR.ss58_address

    payload = make_payload(validator_signature=None)

    try:
        await service.upload_ssh_key(payload)
    except HTTPException as exc:
        assert exc.status_code == 401
        assert "missing" in exc.detail.lower()
        assert dummy.added_keys == []
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected HTTPException for missing validator signature")


async def rejects_invalid_validator_signature() -> None:
    dummy = DummySSHService()
    service = MinerService(dummy, DummyTDXQuoteService())
    settings.MINER_HOTKEY_SS58_ADDRESS = MINER_KEYPAIR.ss58_address
    settings.DEFAULT_MINER_HOTKEY = MINER_KEYPAIR.ss58_address
    settings.VALIDATOR_HOTKEY_SS58_ADDRESS = VALIDATOR_KEYPAIR.ss58_address

    bad_signature = f"0x{UNAUTHORIZED_KEYPAIR.sign(EXPECTED_MESSAGE).hex()}"
    payload = make_payload(validator_signature=bad_signature)

    try:
        await service.upload_ssh_key(payload)
    except HTTPException as exc:
        assert exc.status_code == 401
        assert "invalid" in exc.detail.lower()
        assert dummy.added_keys == []
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected HTTPException for invalid validator signature")


async def middleware_allows_valid_signatures() -> None:
    dummy = DummySSHService()
    service = MinerService(dummy, DummyTDXQuoteService())
    settings.MINER_HOTKEY_SS58_ADDRESS = MINER_KEYPAIR.ss58_address
    settings.DEFAULT_MINER_HOTKEY = MINER_KEYPAIR.ss58_address
    settings.VALIDATOR_HOTKEY_SS58_ADDRESS = VALIDATOR_KEYPAIR.ss58_address

    payload_dict = {
        "public_key": TEST_PUBLIC_KEY,
        "data_to_sign": TEST_PUBLIC_KEY,
        "signature": f"0x{MINER_KEYPAIR.sign(TEST_PUBLIC_KEY).hex()}",
        "validator_signature": f"0x{VALIDATOR_KEYPAIR.sign(EXPECTED_MESSAGE).hex()}",
    }

    request = build_request(payload_dict)
    middleware = MinerMiddleware(lambda _: None)

    async def call_next(request: Request):
        payload_model = UploadSShKeyPayload.model_validate_json(await request.body())
        result = await service.upload_ssh_key(payload_model)
        return JSONResponse(result)

    response = await middleware.dispatch(request, call_next)
    assert response.status_code == 200
    assert dummy.added_keys == [TEST_PUBLIC_KEY]
    result_body = json.loads(response.body.decode("utf-8"))
    assert result_body["ssh_username"] == dummy.get_current_os_user()


async def middleware_rejects_invalid_miner_signature() -> None:
    dummy = DummySSHService()
    service = MinerService(dummy, DummyTDXQuoteService())
    settings.MINER_HOTKEY_SS58_ADDRESS = MINER_KEYPAIR.ss58_address
    settings.DEFAULT_MINER_HOTKEY = MINER_KEYPAIR.ss58_address
    settings.VALIDATOR_HOTKEY_SS58_ADDRESS = VALIDATOR_KEYPAIR.ss58_address

    payload_dict = {
        "public_key": TEST_PUBLIC_KEY,
        "data_to_sign": TEST_PUBLIC_KEY,
        "signature": f"0x{UNAUTHORIZED_KEYPAIR.sign(TEST_PUBLIC_KEY).hex()}",
        "validator_signature": f"0x{VALIDATOR_KEYPAIR.sign(EXPECTED_MESSAGE).hex()}",
    }

    request = build_request(payload_dict)
    middleware = MinerMiddleware(lambda _: None)
    call_next_called = False

    async def call_next(request: Request):
        nonlocal call_next_called
        call_next_called = True
        payload_model = UploadSShKeyPayload.model_validate_json(await request.body())
        result = await service.upload_ssh_key(payload_model)
        return JSONResponse(result)

    response = await middleware.dispatch(request, call_next)
    assert response.status_code == 401
    assert response.body.decode("utf-8").strip('"') == "Unauthorized"
    assert dummy.added_keys == []
    assert call_next_called is False


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


async def main() -> int:
    original_validator = settings.VALIDATOR_HOTKEY_SS58_ADDRESS
    original_miner = settings.MINER_HOTKEY_SS58_ADDRESS
    original_default_miner = settings.DEFAULT_MINER_HOTKEY
    checks: Sequence[tuple[str, Callable[[], Awaitable[None]]]] = (
        ("allows-without-validator-env", allows_without_validator_env),
        ("accepts-valid-validator-signature", accepts_valid_validator_signature),
        ("rejects-missing-validator-signature", rejects_missing_validator_signature),
        ("rejects-invalid-validator-signature", rejects_invalid_validator_signature),
        ("middleware-allows-valid-signatures", middleware_allows_valid_signatures),
        ("middleware-rejects-invalid-miner-signature", middleware_rejects_invalid_miner_signature),
    )

    failures: List[str] = []

    try:
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
    finally:
        settings.VALIDATOR_HOTKEY_SS58_ADDRESS = original_validator
        settings.MINER_HOTKEY_SS58_ADDRESS = original_miner
        settings.DEFAULT_MINER_HOTKEY = original_default_miner

    if failures:
        print("\nChecks failed:")
        for failure in failures:
            print(f" - {failure}")
        return 1

    print("\nAll dual-signature checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
