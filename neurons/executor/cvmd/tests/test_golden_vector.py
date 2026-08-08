"""The cross-version wire-compatibility guard.

cvmd pins bittensor 11.x. Its signers — the validator and the connector — stay on 9.x. This test
is the only thing standing between that split and a silent wire break: if a future bittensor bump
changes the signature format or the sr25519 primitive, every other test in this suite still
passes, because they sign and verify with the same library.

The vector is generated once, by hand, under a real bittensor 9.x venv:

    /path/to/neurons/validators/.venv/bin/python tests/fixtures/generate_golden_vector.py

Do not regenerate it in CI. A vector produced by the implementation under test proves nothing.
"""

import json
from pathlib import Path

import bittensor
from bittensor.sp_core import Keypair
from cvmd.auth.blob import signing_blob

VECTOR_PATH = Path(__file__).parent / "fixtures" / "golden_vector.json"


def _vector() -> dict:
    return json.loads(VECTOR_PATH.read_text())


def test_vector_was_signed_by_a_9x_signer():
    """Guards the fixture itself: regenerated under 11.x, this file would be a tautology."""
    signed_with = _vector()["signed_with_bittensor"]
    assert signed_with.startswith("9."), (
        f"the golden vector was signed with bittensor {signed_with}, not a 9.x line — "
        "it no longer proves cross-version compatibility"
    )
    assert bittensor.__version__.startswith("11."), (
        f"cvmd is verifying under bittensor {bittensor.__version__}, not 11.x"
    )


def test_blob_construction_matches_the_9x_signer():
    """cvmd must derive the exact bytes the 9.x signer signed."""
    vector = _vector()
    blob = signing_blob(
        method=vector["method"],
        request_target=vector["request_target"],
        body=vector["body_utf8"].encode(),
        timestamp=vector["timestamp"],
        nonce=vector["nonce"],
    )
    assert blob.hex() == vector["blob_sha256_hex"], (
        "the signed-blob construction changed — this is a breaking protocol change for the "
        "DAH-2576 connector and the DAH-2580 validator, not a fixture to refresh"
    )


def test_9x_signature_verifies_under_11x():
    """The claim the version split rests on."""
    vector = _vector()
    blob = bytes.fromhex(vector["blob_sha256_hex"])
    signature = bytes.fromhex(vector["signature_hex"])

    assert Keypair(ss58_address=vector["ss58_address"]).verify(blob, signature) is True


def test_tampered_vector_does_not_verify():
    """Proves the assertion above is a real check and not a constant True."""
    vector = _vector()
    blob = bytearray(bytes.fromhex(vector["blob_sha256_hex"]))
    blob[0] ^= 0x01
    signature = bytes.fromhex(vector["signature_hex"])

    assert Keypair(ss58_address=vector["ss58_address"]).verify(bytes(blob), signature) is False


def test_primitive_is_sr25519():
    """crypto_type 1 and a 64-byte signature — unchanged from 9.x."""
    vector = _vector()
    assert vector["crypto_type"] == 1
    assert len(bytes.fromhex(vector["signature_hex"])) == 64
