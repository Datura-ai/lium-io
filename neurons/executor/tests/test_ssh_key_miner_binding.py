"""/upload_ssh_key checks the validator issued the key for this executor's own miner hotkey.

Log-only by default (`ssh_key_miner_binding` warning, upload accepted);
REQUIRE_MINER_BOUND_SSH_KEY refuses instead. Uploads without the miner-bound
signature (older validators or miners) are logged as legacy.
"""

import logging
from unittest.mock import AsyncMock, MagicMock

import bittensor
import pytest
from datura.requests.validator_requests import ssh_pubkey_signing_blob
from fastapi import FastAPI
from fastapi.testclient import TestClient
from middlewares.miner import MinerMiddleware

from core.config import settings
from routes.apis import apis_router
from services.miner_service import MinerService

_SSH_KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAITestKey123456789abcdef user@host"
_NONCE = "ab" * 32
_LOG_KEY = "ssh_key_miner_binding"


@pytest.fixture(scope="module")
def validator_kp():
    return bittensor.Keypair.create_from_uri("//LiumTestValidator")


@pytest.fixture(scope="module")
def miner_kp():
    return bittensor.Keypair.create_from_uri("//LiumTestMiner")


@pytest.fixture(scope="module")
def other_miner_kp():
    return bittensor.Keypair.create_from_uri("//LiumTestOtherMiner")


@pytest.fixture()
def client(validator_kp, miner_kp, monkeypatch):
    monkeypatch.setattr(settings, "MINER_HOTKEY_SS58_ADDRESS", miner_kp.ss58_address)
    monkeypatch.setattr(settings, "DEFAULT_MINER_HOTKEY", miner_kp.ss58_address)
    monkeypatch.setattr(settings, "REQUIRE_MINER_BOUND_SSH_KEY", False)
    monkeypatch.setattr("dependencies.auth.VALIDATOR_HOTKEYS_SS58", {"current": validator_kp.ss58_address})
    app = FastAPI()
    app.add_middleware(MinerMiddleware)
    app.include_router(apis_router)
    service = MagicMock()
    service.upload_ssh_key = AsyncMock(return_value={"ssh_username": "u", "ssh_port": 2200})
    app.dependency_overrides[MinerService] = lambda: service
    return TestClient(app)


def _sign(kp, msg):
    return "0x" + kp.sign(msg).hex()


def _payload(miner_kp, validator_kp, bound_hotkey=None, nonce=None, signer=None):
    body = {
        "public_key": _SSH_KEY,
        "data_to_sign": _SSH_KEY,
        "signature": _sign(miner_kp, _SSH_KEY),
        "validator_signature": _sign(validator_kp, ssh_pubkey_signing_blob(_SSH_KEY, nonce)),
    }
    if nonce:
        body["nonce"] = nonce
    if bound_hotkey:
        body["miner_hotkey"] = bound_hotkey
        body["validator_miner_signature"] = _sign(
            signer or validator_kp, ssh_pubkey_signing_blob(_SSH_KEY, nonce, bound_hotkey)
        )
    return body


def _binding_logs(caplog):
    return [r for r in caplog.records if _LOG_KEY in r.getMessage()]


@pytest.mark.parametrize("nonce", [None, _NONCE])
def test_matching_hotkey_accepted_without_log(client, miner_kp, validator_kp, caplog, nonce):
    caplog.set_level(logging.INFO)
    resp = client.post("/upload_ssh_key", json=_payload(miner_kp, validator_kp, miner_kp.ss58_address, nonce))
    assert resp.status_code == 200
    assert _binding_logs(caplog) == []


def test_mismatched_hotkey_logged_and_accepted(client, miner_kp, validator_kp, other_miner_kp, caplog):
    caplog.set_level(logging.INFO)
    resp = client.post("/upload_ssh_key", json=_payload(miner_kp, validator_kp, other_miner_kp.ss58_address))
    assert resp.status_code == 200
    [rec] = _binding_logs(caplog)
    assert "mismatch" in rec.getMessage() and other_miner_kp.ss58_address in rec.getMessage()


def test_mismatched_hotkey_refused_when_enforced(client, miner_kp, validator_kp, other_miner_kp, monkeypatch):
    monkeypatch.setattr(settings, "REQUIRE_MINER_BOUND_SSH_KEY", True)
    resp = client.post("/upload_ssh_key", json=_payload(miner_kp, validator_kp, other_miner_kp.ss58_address))
    assert resp.status_code == 403


def test_hotkey_not_signed_by_validator_is_bad_signature(client, miner_kp, validator_kp, other_miner_kp, caplog, monkeypatch):
    # the miner-bound signature is not the validator's: the claimed hotkey proves nothing
    caplog.set_level(logging.INFO)
    body = _payload(miner_kp, validator_kp, miner_kp.ss58_address, signer=other_miner_kp)
    assert client.post("/upload_ssh_key", json=body).status_code == 200
    [rec] = _binding_logs(caplog)
    assert "bad_signature" in rec.getMessage()
    monkeypatch.setattr(settings, "REQUIRE_MINER_BOUND_SSH_KEY", True)
    assert client.post("/upload_ssh_key", json=body).status_code == 403


def test_legacy_payload_accepted_and_logged(client, miner_kp, validator_kp, caplog):
    caplog.set_level(logging.INFO)
    assert client.post("/upload_ssh_key", json=_payload(miner_kp, validator_kp)).status_code == 200
    [rec] = _binding_logs(caplog)
    assert "legacy" in rec.getMessage()


def test_legacy_payload_refused_when_enforced(client, miner_kp, validator_kp, monkeypatch):
    monkeypatch.setattr(settings, "REQUIRE_MINER_BOUND_SSH_KEY", True)
    assert client.post("/upload_ssh_key", json=_payload(miner_kp, validator_kp)).status_code == 403


def test_signing_blob_without_hotkey_unchanged():
    # the validator_signature format older executors verify must not move
    assert ssh_pubkey_signing_blob(_SSH_KEY) == _SSH_KEY
    assert ssh_pubkey_signing_blob(_SSH_KEY, _NONCE) == f"{_SSH_KEY}\nnonce:{_NONCE}"
    assert ssh_pubkey_signing_blob(_SSH_KEY, None, "5Abc") == f"{_SSH_KEY}\nminer:5Abc"
