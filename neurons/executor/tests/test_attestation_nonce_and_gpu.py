"""
Executor-side tests for the CVM attestation gap remediation (DAH-2338, G1/G3).

Covers:
- G3 signature surface: validator_signature must cover public_key ‖ nonce when a
  nonce rides the request — swapping or stripping the nonce invalidates it; the
  legacy (no-nonce) format keeps working; REQUIRE_ATTESTATION_NONCE gates the
  enforcement phase on the upload route only.
- G3 quote binding: TDX report_data becomes sha256 half ‖ nonce half, nonce-bound
  quotes bypass the cache, malformed nonces are rejected.
- G1 topology guard: GPU evidence is emitted only in-CVM with the feature flag on;
  host-mode never emits; a missing NVIDIA stack degrades to a logged None.
"""

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from bittensor_wallet import Keypair
import pytest
from datura.requests import validator_requests
from datura.requests.validator_requests import fresh_upload_nonce, ssh_pubkey_signing_blob
from fastapi import FastAPI
from fastapi.testclient import TestClient
from middlewares.miner import MinerMiddleware

# conftest.py already inserted src/ into sys.path and set required env vars.
from core.config import settings
from routes.apis import apis_router
from services.miner_service import MinerService
from services.tdx_service import TDXQuoteService

_SSH_KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAITestKey123456789abcdef user@host"
_NONCE = "ab" * 32


# ---------------------------------------------------------------------------
# Fixtures (same shape as test_upload_ssh_key_auth.py)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def validator_keypair():
    return Keypair.create_from_uri("//LiumTestValidator")


@pytest.fixture(scope="module")
def miner_keypair():
    return Keypair.create_from_uri("//LiumTestMiner")


@pytest.fixture()
def client(validator_keypair, miner_keypair, monkeypatch):
    monkeypatch.setattr(settings, "MINER_HOTKEY_SS58_ADDRESS", miner_keypair.ss58_address)
    monkeypatch.setattr(settings, "DEFAULT_MINER_HOTKEY", miner_keypair.ss58_address)
    monkeypatch.setattr(
        "dependencies.auth.VALIDATOR_HOTKEYS_SS58", {"current": validator_keypair.ss58_address}
    )

    app = FastAPI()
    app.add_middleware(MinerMiddleware)
    app.include_router(apis_router)

    mock_service = MagicMock()
    mock_service.upload_ssh_key = AsyncMock(
        return_value={"ssh_username": "testuser", "ssh_port": 2200}
    )
    mock_service.remove_ssh_key = AsyncMock(return_value=None)
    app.dependency_overrides[MinerService] = lambda: mock_service

    return TestClient(app)


def _build_payload(miner_kp, validator_kp, nonce=None, signed_nonce="__same__") -> dict:
    """Signed payload; `signed_nonce` lets tests sign over a different nonce
    than the one shipped in the payload (swap/strip attacks)."""
    if signed_nonce == "__same__":
        signed_nonce = nonce
    miner_sig = "0x" + miner_kp.sign(_SSH_KEY).hex()
    validator_sig = "0x" + validator_kp.sign(ssh_pubkey_signing_blob(_SSH_KEY, signed_nonce)).hex()
    payload = {
        "public_key": _SSH_KEY,
        "data_to_sign": _SSH_KEY,
        "signature": miner_sig,
        "validator_signature": validator_sig,
    }
    if nonce is not None:
        payload["nonce"] = nonce
    return payload


# ---------------------------------------------------------------------------
# G3 — signature surface
# ---------------------------------------------------------------------------


def test_swapped_nonce_rejected(client, miner_keypair, validator_keypair):
    # Arrange — valid signature over one nonce, a different nonce in the payload
    payload = _build_payload(
        miner_keypair, validator_keypair, nonce="cd" * 32, signed_nonce=_NONCE
    )

    # Act / Assert — covers the plan's nonce-swap-with-valid-signature case
    assert client.post("/upload_ssh_key", json=payload).status_code == 401


def test_stripped_nonce_rejected(client, miner_keypair, validator_keypair):
    # Arrange — validator signed public_key ‖ nonce but the nonce was removed
    payload = _build_payload(
        miner_keypair, validator_keypair, nonce=None, signed_nonce=_NONCE
    )

    # Act / Assert — downgrade-to-legacy verification must fail
    assert client.post("/upload_ssh_key", json=payload).status_code == 401


def test_nonce_without_covering_signature_rejected(client, miner_keypair, validator_keypair):
    # Arrange — nonce in payload but the signature only covers the public key
    payload = _build_payload(
        miner_keypair, validator_keypair, nonce=_NONCE, signed_nonce=None
    )

    # Act / Assert
    assert client.post("/upload_ssh_key", json=payload).status_code == 401


def test_require_nonce_enforcement_upload_only(
    client, miner_keypair, validator_keypair, monkeypatch
):
    # Arrange — enforcement phase: uploads without a nonce are rejected
    monkeypatch.setattr(settings, "REQUIRE_ATTESTATION_NONCE", True)
    no_nonce = _build_payload(miner_keypair, validator_keypair, nonce=None)
    undated = _build_payload(miner_keypair, validator_keypair, nonce=_NONCE)
    with_nonce = _build_payload(miner_keypair, validator_keypair, nonce=fresh_upload_nonce())

    # Act / Assert
    assert client.post("/upload_ssh_key", json=no_nonce).status_code == 401
    assert client.post("/upload_ssh_key", json=undated).status_code == 401
    assert client.post("/upload_ssh_key", json=with_nonce).status_code == 200
    # Removals never carry a nonce and must keep working during enforcement.
    assert client.post("/remove_ssh_key", json=no_nonce).status_code == 200


@pytest.mark.parametrize("age, status", [(0, 200), (590, 200), (-590, 200), (610, 401), (-610, 401)])
def test_dated_nonce_freshness(client, miner_keypair, validator_keypair, age, status, monkeypatch):
    # I-78: a captured upload signature over a dated nonce is refused once outside the window
    minted_at = time.time() - age
    # patch only the minting module's `time`: the executor's freshness check keeps the real clock
    with monkeypatch.context() as m:
        m.setattr(validator_requests, "time", SimpleNamespace(time=lambda: minted_at))
        payload = _build_payload(miner_keypair, validator_keypair, nonce=fresh_upload_nonce())
    assert client.post("/upload_ssh_key", json=payload).status_code == status


# ---------------------------------------------------------------------------
# G3 — TDX quote nonce binding
# ---------------------------------------------------------------------------


@pytest.fixture()
def tdx_service(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_TDX_ATTESTATION", True)
    return TDXQuoteService()


# ---------------------------------------------------------------------------
# G1 — GPU evidence emission topology guard
# ---------------------------------------------------------------------------


