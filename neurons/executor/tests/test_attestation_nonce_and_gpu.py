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

import asyncio
from unittest.mock import AsyncMock, MagicMock

import bittensor
import pytest
from datura.requests.validator_requests import ssh_pubkey_signing_blob
from fastapi import FastAPI
from fastapi.testclient import TestClient
from middlewares.miner import MinerMiddleware

# conftest.py already inserted src/ into sys.path and set required env vars.
from core.config import settings
from routes.apis import apis_router
from services.gpu_attestation_service import GPUAttestationService
from services.miner_service import MinerService
from services.tdx_service import TDXQuoteService

_SSH_KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAITestKey123456789abcdef user@host"
_NONCE = "ab" * 32


# ---------------------------------------------------------------------------
# Fixtures (same shape as test_upload_ssh_key_auth.py)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def validator_keypair():
    return bittensor.Keypair.create_from_uri("//LiumTestValidator")


@pytest.fixture(scope="module")
def miner_keypair():
    return bittensor.Keypair.create_from_uri("//LiumTestMiner")


@pytest.fixture()
def client(validator_keypair, miner_keypair, monkeypatch):
    monkeypatch.setattr(settings, "MINER_HOTKEY_SS58_ADDRESS", miner_keypair.ss58_address)
    monkeypatch.setattr(settings, "DEFAULT_MINER_HOTKEY", miner_keypair.ss58_address)
    monkeypatch.setattr("routes.apis.VALIDATOR_HOTKEY_SS58", validator_keypair.ss58_address)

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


def test_nonce_covered_signature_accepted(client, miner_keypair, validator_keypair):
    # Arrange — signature over public_key ‖ nonce, nonce shipped in payload
    payload = _build_payload(miner_keypair, validator_keypair, nonce=_NONCE)

    # Act
    response = client.post("/upload_ssh_key", json=payload)

    # Assert
    assert response.status_code == 200


def test_legacy_no_nonce_signature_still_accepted(client, miner_keypair, validator_keypair):
    # Arrange — deployed-fleet format: bare public key signature, no nonce
    payload = _build_payload(miner_keypair, validator_keypair, nonce=None)

    # Act / Assert
    assert client.post("/upload_ssh_key", json=payload).status_code == 200


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
    with_nonce = _build_payload(miner_keypair, validator_keypair, nonce=_NONCE)

    # Act / Assert
    assert client.post("/upload_ssh_key", json=no_nonce).status_code == 401
    assert client.post("/upload_ssh_key", json=with_nonce).status_code == 200
    # Removals never carry a nonce and must keep working during enforcement.
    assert client.post("/remove_ssh_key", json=no_nonce).status_code == 200


# ---------------------------------------------------------------------------
# G3 — TDX quote nonce binding
# ---------------------------------------------------------------------------


@pytest.fixture()
def tdx_service(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_TDX_ATTESTATION", True)
    return TDXQuoteService()


def _fake_dstack_client(calls):
    client = MagicMock()

    async def fake_get_quote(report_bytes):
        calls.append(report_bytes)
        response = MagicMock()
        response.model_dump_json.return_value = '{"quote": "aa"}'
        return response

    client.get_quote = fake_get_quote
    return client


def test_report_data_layout_with_nonce(tdx_service):
    # Act
    _, legacy = tdx_service._report_data("host-key")
    _, bound = tdx_service._report_data("host-key", bytes.fromhex(_NONCE))

    # Assert — identity half preserved, nonce fills [32:64]
    assert len(legacy) == 32
    assert len(bound) == 64
    assert bound[:32] == legacy
    assert bound[32:] == bytes.fromhex(_NONCE)


def test_get_quote_binds_nonce_and_skips_cache(tdx_service, monkeypatch):
    # Arrange — the executor suite has no async pytest plugin; drive with asyncio.run
    calls = []
    client = _fake_dstack_client(calls)

    async def fake_get_client():
        return client

    monkeypatch.setattr(tdx_service, "_get_client", fake_get_client)

    async def scenario():
        # two nonce-bound calls and two legacy calls
        await tdx_service.get_quote("host-key", nonce=_NONCE)
        await tdx_service.get_quote("host-key", nonce=_NONCE)
        await tdx_service.get_quote("host-key")
        await tdx_service.get_quote("host-key")

    # Act
    asyncio.run(scenario())

    # Assert — nonce path regenerates per event (no cache), legacy path caches
    assert len(calls) == 3
    assert calls[0][32:] == bytes.fromhex(_NONCE)
    assert len(calls[2]) == 32


def test_get_quote_rejects_malformed_nonce(tdx_service, monkeypatch):
    # Arrange
    calls = []
    client = _fake_dstack_client(calls)

    async def fake_get_client():
        return client

    monkeypatch.setattr(tdx_service, "_get_client", fake_get_client)

    async def scenario():
        return (
            await tdx_service.get_quote("host-key", nonce="zz"),
            await tdx_service.get_quote("host-key", nonce="ab"),
        )

    # Act
    bad_hex, bad_length = asyncio.run(scenario())

    # Assert — bad hex and wrong length both refuse to mint a quote
    assert bad_hex is None
    assert bad_length is None
    assert calls == []


# ---------------------------------------------------------------------------
# G1 — GPU evidence emission topology guard
# ---------------------------------------------------------------------------


def test_gpu_collection_disabled_by_default():
    # Arrange — both flags default off
    service = GPUAttestationService()

    # Act / Assert
    assert asyncio.run(service.collect(_NONCE)) is None


def test_gpu_collection_skipped_in_host_mode(monkeypatch):
    # Arrange — flags on but no dstack guest socket (bare-metal topology)
    monkeypatch.setattr(settings, "ENABLE_TDX_ATTESTATION", True)
    monkeypatch.setattr(settings, "ENABLE_GPU_ATTESTATION", True)
    monkeypatch.setattr(settings, "DSTACK_SOCKET_PATH", "/nonexistent/dstack.sock")
    service = GPUAttestationService()

    # Act / Assert — host-mode must never emit CC evidence
    assert asyncio.run(service.collect(_NONCE)) is None


def test_gpu_collection_survives_missing_nvidia_stack(monkeypatch, tmp_path):
    # Arrange — in-CVM topology but the NVIDIA verifier stack is not installed
    # (this test environment has no nv_attestation_sdk — the real import fails)
    socket = tmp_path / "dstack.sock"
    socket.touch()
    monkeypatch.setattr(settings, "ENABLE_TDX_ATTESTATION", True)
    monkeypatch.setattr(settings, "ENABLE_GPU_ATTESTATION", True)
    monkeypatch.setattr(settings, "DSTACK_SOCKET_PATH", str(socket))
    service = GPUAttestationService()

    # Act / Assert — logged failure, never a crash of the upload path
    assert asyncio.run(service.collect(_NONCE)) is None


def test_gpu_nonce_normalisation():
    # Arrange
    service = GPUAttestationService()

    # Act / Assert — validator nonce passes through lowercased
    assert service._normalise_nonce("AB" * 32) == "ab" * 32
    # Malformed validator nonce is a protocol error → refuse (no silent self-nonce)
    assert service._normalise_nonce("zz") is None
    assert service._normalise_nonce("ab") is None
    # No nonce (phase-0) → self-generated 32-byte hex
    self_nonce = service._normalise_nonce(None)
    assert self_nonce is not None
    assert len(bytes.fromhex(self_nonce)) == 32
