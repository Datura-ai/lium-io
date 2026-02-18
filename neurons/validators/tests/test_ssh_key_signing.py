"""
Tests for MinerService._sign_validator_pubkey.

Focus: verify that the validator signs SSH public keys correctly and that
the resulting validator_signature is in the right format and is verifiable
by bittensor — matching the format expected by the executor's
_validate_validator_signature check.
"""

import bittensor
import pytest

from services.miner_service import MinerService


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


def test_sign_validator_pubkey_bytes_decodes_and_produces_verifiable_signature(
    miner_service, validator_keypair
):
    """Signing a bytes pubkey decodes it first; the signature verifies against the str form."""
    # Arrange — validator_requests sends public_key as bytes
    public_key_bytes = _SSH_KEY.encode("utf-8")

    # Act
    signature = miner_service._sign_validator_pubkey(validator_keypair, public_key_bytes)

    # Assert — hex-encoded with 0x prefix
    assert isinstance(signature, str)
    assert signature.startswith("0x")

    # Assert — verifiable against the decoded str form (executor receives public_key as str)
    assert validator_keypair.verify(_SSH_KEY, signature)
