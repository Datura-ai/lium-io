"""Generate the cross-version golden vector. Run once, by hand, under a bittensor 9.x venv.

    /path/to/neurons/validators/.venv/bin/python tests/fixtures/generate_golden_vector.py

The point of this fixture is that cvmd pins bittensor 11.x while its signers — the validator and
the connector — stay on 9.x. That split is only safe because a 9.x signature verifies under 11.x,
and this vector is the committed proof. It is generated once and committed; CI verifies it, never
regenerates it. Regenerating under 11.x would turn the one test that can catch a wire-format
break into a tautology.

The blob construction below is re-implemented rather than imported: cvmd's own code cannot run on
the 9.x venv, and a vector produced by the implementation under test proves nothing.
"""

import hashlib
import json
from pathlib import Path

import bittensor

DOMAIN_SEPARATOR = b"cvmd-v1\x00"

# Fixed inputs — a realistic renter-create request. Nothing here is time-sensitive: the vector
# exercises signature verification, not the freshness window.
METHOD = "POST"
REQUEST_TARGET = "/v1/cvm"
BODY = b'{"kind":"renter","image":"sha256:abc","kernel":"sha256:def","initrd":"sha256:012"}'
TIMESTAMP = "1754500000000000000"
NONCE = "9f2c1a7b4e6d80f35c9a1e4b7d2f6083"
URI = "//Alice"


def _lp(value: bytes) -> bytes:
    return len(value).to_bytes(4, "big") + value


def main() -> None:
    keypair = bittensor.Keypair.create_from_uri(URI)

    blob = hashlib.sha256(
        DOMAIN_SEPARATOR
        + _lp(METHOD.encode())
        + _lp(REQUEST_TARGET.encode())
        + _lp(BODY)
        + _lp(TIMESTAMP.encode())
        + _lp(NONCE.encode())
    ).digest()

    signature = keypair.sign(blob)
    assert keypair.verify(blob, signature), "9.x cannot verify its own signature"

    vector = {
        "_comment": (
            "Signed once under the bittensor version named below. Verify it; never regenerate it "
            "in CI. If this fails after a bittensor bump, the new version has broken wire "
            "compatibility with the 9.x signers — that is the finding, not a stale fixture."
        ),
        "signed_with_bittensor": bittensor.__version__,
        "ss58_address": keypair.ss58_address,
        "crypto_type": keypair.crypto_type,
        "method": METHOD,
        "request_target": REQUEST_TARGET,
        "body_utf8": BODY.decode(),
        "timestamp": TIMESTAMP,
        "nonce": NONCE,
        "blob_sha256_hex": blob.hex(),
        "signature_hex": signature.hex(),
    }

    out = Path(__file__).with_name("golden_vector.json")
    out.write_text(json.dumps(vector, indent=2) + "\n")
    print(f"wrote {out} (bittensor {bittensor.__version__}, {keypair.ss58_address})")


if __name__ == "__main__":
    main()
