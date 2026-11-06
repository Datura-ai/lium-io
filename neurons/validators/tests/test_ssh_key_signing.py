"""
Tests for MinerService._sign_validator_pubkey.

Focus: verify that the validator signs SSH public keys correctly and that
the resulting validator_signature is in the right format and is verifiable
by bittensor — matching the format expected by the executor's
_validate_validator_signature check.
"""

import logging
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import bittensor
import pytest
from datura.requests.validator_requests import upload_nonce_issued_at

from clients.subtensor_client import SubtensorClient
from payload_models.payloads import MinerJobEnryptedFiles, MinerJobRequestPayload
from protocol.vc_protocol.compute_requests import RentedExecutorsResponse
from services.attestation_service import AttestationNonce
from services.miner_service import MIN_DATED_NONCE_MINER_AXON_VERSION, MinerService

pytest_plugins = ["fixtures.rest_miner_fixtures"]

_SSH_KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAITestKey123456789abcdef user@host"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def validator_keypair():
    return bittensor.Keypair.create_from_uri("//LiumTestValidator")


@pytest.fixture(scope="module")
def miner_service():
    """MinerService instance with no dependencies — _sign_validator_pubkey
    only calls _normalize_public_key (a @staticmethod), so __init__ is skipped."""
    return MinerService.__new__(MinerService)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_sign_validator_pubkey_str_produces_verifiable_signature(miner_service, validator_keypair):
    """Signing a str pubkey produces a 0x-prefixed signature verifiable by bittensor."""
    # Arrange
    public_key = _SSH_KEY

    # Act
    signature = miner_service._sign_validator_pubkey(validator_keypair, public_key)

    # Assert — hex-encoded with 0x prefix (format expected by executor's verification)
    assert isinstance(signature, str)
    assert signature.startswith("0x")

    # Assert — the signature is cryptographically valid over the original key
    assert validator_keypair.verify(public_key, signature)


def _subtensor(monkeypatch, *, axon_versions=None, central=()):
    neurons = [
        SimpleNamespace(hotkey=hotkey, axon_info=SimpleNamespace(version=version))
        for hotkey, version in (axon_versions or {}).items()
    ]
    fake = SimpleNamespace(miners=neurons, hotkeys_routed_via_central_miner=frozenset(central))
    monkeypatch.setattr(SubtensorClient, "get_instance", classmethod(lambda cls: fake))


def test_upload_nonce_dated_for_a_miner_on_bittensor_10_5(miner_service, monkeypatch, caplog):
    # I-78: a miner announcing bittensor 10.5.0+ (miner-v1.003+) relays the dated nonce
    _subtensor(
        monkeypatch, axon_versions={"new": MIN_DATED_NONCE_MINER_AXON_VERSION, "newer": 10_006_001}
    )
    with caplog.at_level(logging.INFO):
        for hotkey in ("new", "newer"):
            assert (
                abs(
                    upload_nonce_issued_at(miner_service._upload_nonce_for_miner(hotkey))
                    - time.time()
                )
                < 5
            )
    assert "dated_upload_nonce_skipped" not in caplog.text


def test_upload_nonce_dated_for_the_central_miner_whatever_its_axon(miner_service, monkeypatch):
    _subtensor(monkeypatch, axon_versions={"opted": 0}, central={"opted"})
    assert upload_nonce_issued_at(miner_service._upload_nonce_for_miner("opted")) is not None


@pytest.mark.parametrize("hotkey", ["old", "zero", "unknown"])
def test_upload_nonce_skipped_and_logged_for_an_unproven_miner(
    miner_service, monkeypatch, caplog, hotkey
):
    _subtensor(monkeypatch, axon_versions={"old": 9_010_001, "zero": 0})
    with caplog.at_level(logging.INFO):
        assert miner_service._upload_nonce_for_miner(hotkey) is None
    [skipped] = [r.msg.extra for r in caplog.records if str(r.msg) == "dated_upload_nonce_skipped"]
    assert skipped["miner_hotkey"] == hotkey
    assert skipped["axon_version"] == {"old": 9_010_001, "zero": 0, "unknown": None}[hotkey]


@pytest.fixture
def miner_task_result():
    return lambda executor_id: None


@pytest.mark.asyncio
@pytest.mark.parametrize("attestation_due", [True, False])
async def test_old_miner_still_gets_the_attestation_nonce(
    rest_miner_service, monkeypatch, attestation_due
):
    # the attestation path keeps sending its own nonce to every miner; only the plain upload is gated
    _subtensor(monkeypatch, axon_versions={"miner-a": 9_010_001})
    attestation = AttestationNonce.issue()
    rest_miner_service.attestation_service.maybe_issue_nonce = AsyncMock(
        return_value=attestation if attestation_due else None
    )
    rest_miner_service.miner_returns()
    await rest_miner_service.request_job_to_miner(
        payload=MinerJobRequestPayload(
            job_batch_id="b",
            miner_hotkey="miner-a",
            miner_coldkey="c",
            miner_address="192.0.2.10",
            miner_port=8091,
        ),
        encrypted_files=MinerJobEnryptedFiles(
            encrypt_key="k",
            all_keys={},
            tmp_directory="/tmp",
            machine_scrape_file_name="s",
            machine_scrape_source="",
        ),
        rented_data=RentedExecutorsResponse(executors={}),
        default_docker_image_digests={},
    )
    [submitted] = [
        body for name, body in rest_miner_service.rest_calls if name == "ssh-pubkey-submit"
    ]
    assert submitted["nonce"] == (attestation.value_hex if attestation_due else None)
