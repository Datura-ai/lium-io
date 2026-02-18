"""
Integration tests for /upload_ssh_key and /remove_ssh_key authentication.

Both endpoints share the same two-layer auth model:
  1. MinerMiddleware  – verifies the miner's signature over `data_to_sign`
  2. _validate_validator_signature – verifies the validator's signature over `public_key`

Ephemeral bittensor keypairs are created inside the test fixtures so no real
production keys are needed and every test run uses fresh cryptographic material.
"""

from unittest.mock import AsyncMock, MagicMock

import bittensor
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# conftest.py already inserted src/ into sys.path and set required env vars.
from core.config import settings
from middlewares.miner import MinerMiddleware
from routes.apis import apis_router
from services.miner_service import MinerService


# A realistic-looking SSH public key used as the payload's public_key.
_SSH_KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAITestKey123456789abcdef user@host"

# Endpoints that share the same auth logic.
_ENDPOINTS = ["/upload_ssh_key", "/remove_ssh_key"]


# ---------------------------------------------------------------------------
# Ephemeral keypair fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def validator_keypair():
    """
    Ephemeral bittensor SR25519 keypair acting as the trusted validator.

    Created from a deterministic test URI so the ss58 address is stable
    within a test session without touching the file-system or a real wallet.
    """
    return bittensor.Keypair.create_from_uri("//LiumTestValidator")


@pytest.fixture(scope="module")
def miner_keypair():
    """
    Ephemeral bittensor SR25519 keypair acting as the authorized miner.

    Used to generate valid miner signatures and to configure the executor's
    MINER_HOTKEY_SS58_ADDRESS setting in the test app.
    """
    return bittensor.Keypair.create_from_uri("//LiumTestMiner")


# ---------------------------------------------------------------------------
# App / client fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(validator_keypair, miner_keypair, monkeypatch):
    """
    TestClient for the executor app with auth wired to the test keypairs.

    Patches applied for the duration of each test:
      - settings.MINER_HOTKEY_SS58_ADDRESS → test miner's ss58 address
      - settings.DEFAULT_MINER_HOTKEY      → same (prevents accidental match
                                             via the fallback hotkey)
      - routes.apis.VALIDATOR_HOTKEY_SS58  → test validator's ss58 address

    MinerService is overridden so no real SSH or TDX operations occur.
    """
    # --- patch miner auth ---
    monkeypatch.setattr(settings, "MINER_HOTKEY_SS58_ADDRESS", miner_keypair.ss58_address)
    # Prevent the DEFAULT_MINER_HOTKEY fallback from accidentally accepting
    # signatures from unrelated keypairs created in individual tests.
    monkeypatch.setattr(settings, "DEFAULT_MINER_HOTKEY", miner_keypair.ss58_address)

    # --- patch validator auth ---
    monkeypatch.setattr("routes.apis.VALIDATOR_HOTKEY_SS58", validator_keypair.ss58_address)

    # --- build minimal app ---
    app = FastAPI()
    app.add_middleware(MinerMiddleware)
    app.include_router(apis_router)

    # --- stub MinerService so no SSH/TDX side effects run ---
    mock_service = MagicMock()
    mock_service.upload_ssh_key = AsyncMock(
        return_value={"ssh_username": "testuser", "ssh_port": 2200}
    )
    mock_service.remove_ssh_key = AsyncMock(return_value=None)
    app.dependency_overrides[MinerService] = lambda: mock_service

    return TestClient(app)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _build_payload(public_key: str, miner_kp, validator_kp) -> dict:
    """Return a fully signed UploadSShKeyPayload dict."""
    miner_sig = "0x" + miner_kp.sign(public_key).hex()
    validator_sig = "0x" + validator_kp.sign(public_key).hex()
    return {
        "public_key": public_key,
        "data_to_sign": public_key,  # must equal public_key per Issue #744 check
        "signature": miner_sig,
        "validator_signature": validator_sig,
    }


# ---------------------------------------------------------------------------
# Tests: both auth layers valid
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("endpoint", _ENDPOINTS)
def test_valid_miner_and_validator_signatures_accepted(endpoint, client, miner_keypair, validator_keypair):
    """Both signatures are correct → endpoint returns 200."""
    # Arrange
    payload = _build_payload(_SSH_KEY, miner_keypair, validator_keypair)

    # Act
    response = client.post(endpoint, json=payload)

    # Assert — only when both auth layers pass should the request succeed
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Tests: miner signature failures (middleware layer)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("endpoint", _ENDPOINTS)
def test_miner_signature_from_wrong_keypair_rejected(endpoint, client, validator_keypair):
    """
    Miner signature produced by an unknown keypair → 401 from MinerMiddleware.

    Simulates a request that does not come from the registered miner.
    """
    # Arrange — sign with a completely different keypair
    unknown_miner = bittensor.Keypair.create_from_uri("//UnknownMiner")
    payload = _build_payload(_SSH_KEY, unknown_miner, validator_keypair)

    # Act
    response = client.post(endpoint, json=payload)

    # Assert — middleware must reject a signature from an unregistered miner
    assert response.status_code == 401


@pytest.mark.parametrize("endpoint", _ENDPOINTS)
def test_corrupted_miner_signature_rejected(endpoint, client, miner_keypair, validator_keypair):
    """
    Miner signature is syntactically present but cryptographically invalid → 401.

    Simulates a tampered or bit-flipped signature field.
    """
    # Arrange
    payload = _build_payload(_SSH_KEY, miner_keypair, validator_keypair)
    payload["signature"] = "0xdeadbeefdeadbeef"  # not a valid SR25519 signature

    # Act
    response = client.post(endpoint, json=payload)

    # Assert — corrupted bytes must not pass verification
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# Tests: validator signature failures (route layer)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("endpoint", _ENDPOINTS)
def test_validator_signature_from_wrong_keypair_rejected(endpoint, client, miner_keypair):
    """
    Validator signature produced by an attacker's keypair → 401.

    Simulates a request where the miner is legitimate but the validator_signature
    was not produced by the trusted validator.
    """
    # Arrange — miner is valid; attacker signs as validator
    attacker = bittensor.Keypair.create_from_uri("//Attacker")
    payload = _build_payload(_SSH_KEY, miner_keypair, attacker)

    # Act
    response = client.post(endpoint, json=payload)

    # Assert — only the registered validator's signature is accepted
    assert response.status_code == 401


@pytest.mark.parametrize("endpoint", _ENDPOINTS)
def test_corrupted_validator_signature_rejected(endpoint, client, miner_keypair, validator_keypair):
    """
    Miner auth passes but validator_signature is garbage bytes → 401.

    Verifies that the route-level validator check runs independently of
    the middleware and rejects invalid signatures on its own.
    """
    # Arrange
    payload = _build_payload(_SSH_KEY, miner_keypair, validator_keypair)
    payload["validator_signature"] = "0xdeadbeefdeadbeef"  # tampered

    # Act
    response = client.post(endpoint, json=payload)

    # Assert — route must reject the tampered validator signature
    assert response.status_code == 401


@pytest.mark.parametrize("endpoint", _ENDPOINTS)
def test_validator_signature_over_different_key_rejected(endpoint, client, miner_keypair, validator_keypair):
    """
    Replay attack: validator_signature is cryptographically valid but was
    produced for a *different* SSH public key → 401.

    Attack scenario:
      1. Validator legitimately signed key_A for another session.
      2. Attacker reuses that signature while substituting key_B as public_key.
      3. Because verify(key_B, sig_over_key_A) fails, the request is rejected.
    """
    # Arrange
    other_key = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIDifferentKeyForAnotherSession attacker@host"
    # Validator signed a *different* key
    replayed_sig = "0x" + validator_keypair.sign(other_key).hex()
    miner_sig = "0x" + miner_keypair.sign(_SSH_KEY).hex()
    payload = {
        "public_key": _SSH_KEY,
        "data_to_sign": _SSH_KEY,
        "signature": miner_sig,
        "validator_signature": replayed_sig,  # valid sig but over the wrong key
    }

    # Act
    response = client.post(endpoint, json=payload)

    # Assert — validator_signature must specifically cover the submitted public_key
    assert response.status_code == 401
