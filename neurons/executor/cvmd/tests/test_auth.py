"""Signature verification, the body size cap, and blob canonicalization."""

import json

import pytest
from bittensor.sp_core import Keypair
from conftest import sign_headers, signed_request
from cvmd.auth.blob import signing_blob
from cvmd.auth.middleware import (
    HOTKEY_HEADER,
    NONCE_HEADER,
    SIGNATURE_HEADER,
    TIMESTAMP_HEADER,
)

STATE_PATH = "/v1/state"


def test_valid_signature_is_accepted(client, validator_key):
    response = signed_request(client, validator_key, "GET", STATE_PATH)
    assert response.status_code == 200
    assert response.json()["state"] == "RECONCILING"


def test_unsigned_request_is_rejected(client):
    assert client.get(STATE_PATH).status_code == 401


def test_unknown_key_is_rejected(client, stranger_key):
    """A well-formed signature from a hotkey that is not in the authorized-clients file."""
    assert signed_request(client, stranger_key, "GET", STATE_PATH).status_code == 401


def test_tampered_body_is_rejected(client, platform_key):
    body = json.dumps({"kind": "renter"}).encode()
    headers = sign_headers(platform_key, method="POST", request_target="/v1/cvm", body=body)

    tampered = json.dumps({"kind": "renter", "extra": "added after signing"}).encode()
    response = client.post("/v1/cvm", content=tampered, headers=headers)
    assert response.status_code == 401


def test_signature_for_another_method_is_rejected(client, platform_key):
    """The doc's formula omits the method; without it this replay would succeed.

    The platform key holds both POST and DELETE on /v1/cvm, so a captured create signature would
    otherwise be a usable teardown signature inside the freshness window.
    """
    body = json.dumps({"kind": "renter"}).encode()
    headers = sign_headers(platform_key, method="POST", request_target="/v1/cvm", body=body)

    assert client.request("DELETE", "/v1/cvm", content=body, headers=headers).status_code == 401


def test_signature_for_another_query_is_rejected(client, validator_key):
    """The query string is signed — it is attacker-controlled input on DAH-2578's /v1/catalog."""
    headers = sign_headers(validator_key, method="GET", request_target=f"{STATE_PATH}?verbose=1")

    assert client.get(f"{STATE_PATH}?verbose=2", headers=headers).status_code == 401


@pytest.mark.parametrize(
    "missing", [HOTKEY_HEADER, TIMESTAMP_HEADER, NONCE_HEADER, SIGNATURE_HEADER]
)
def test_missing_auth_header_is_rejected(client, validator_key, missing):
    headers = sign_headers(validator_key, method="GET", request_target=STATE_PATH)
    del headers[missing]
    assert client.get(STATE_PATH, headers=headers).status_code == 401


def test_non_hex_signature_is_rejected_not_500(client, validator_key):
    headers = sign_headers(validator_key, method="GET", request_target=STATE_PATH)
    headers[SIGNATURE_HEADER] = "not-hex-at-all"
    assert client.get(STATE_PATH, headers=headers).status_code == 401


def test_short_nonce_is_rejected(client, validator_key):
    """A nonce is the replay store's uniqueness key; a short or constant one degrades it."""
    response = signed_request(client, validator_key, "GET", STATE_PATH, nonce="abcd")
    assert response.status_code == 401


def test_verify_is_bytes_only_in_v11(validator_key):
    """The v9 -> v11 API break, pinned so it cannot regress into a 500.

    v9 accepted a str message and a hex-string signature. v11 raises TypeError for both, which is
    why the middleware decodes the wire values itself instead of passing them through the way
    neurons/executor/src/middlewares/miner.py does.
    """
    blob = signing_blob(
        method="GET", request_target=STATE_PATH, body=b"", timestamp="1", nonce="0" * 32
    )
    signature = validator_key.sign(blob)
    verifier = Keypair(ss58_address=validator_key.ss58_address)

    assert verifier.verify(blob, signature) is True
    with pytest.raises(TypeError):
        verifier.verify(blob.hex(), signature)
    with pytest.raises(TypeError):
        verifier.verify(blob, signature.hex())


class TestBodySizeCap:
    """The cap runs before the body is buffered and before the /health exemption."""

    def test_oversize_body_returns_413(self, client, platform_key):
        oversize = b"x" * (64 * 1024 + 1)
        response = signed_request(client, platform_key, "POST", "/v1/cvm", body=oversize)
        assert response.status_code == 413

    def test_oversize_body_is_rejected_before_signature_work(
        self, client, platform_key, monkeypatch
    ):
        """An oversize body must not reach signature verification — cheap check first."""
        import cvmd.auth.middleware as middleware

        def fail(*args, **kwargs):
            raise AssertionError("signature verification ran on an oversize body")

        monkeypatch.setattr(middleware, "_verify_signature", fail)

        oversize = b"x" * (64 * 1024 + 1)
        response = signed_request(client, platform_key, "POST", "/v1/cvm", body=oversize)
        assert response.status_code == 413

    def test_cap_applies_to_health_too(self, client):
        """/health is exempt from auth, not from the cap."""
        response = client.post("/health", content=b"x" * (64 * 1024 + 1))
        assert response.status_code == 413

    def test_lying_content_length_does_not_bypass_the_cap(self, client, platform_key):
        """Content-Length is a claim; the stream walk is the enforcement."""
        oversize = b"x" * (64 * 1024 + 1)
        headers = sign_headers(platform_key, method="POST", request_target="/v1/cvm", body=oversize)
        headers["Content-Length"] = "10"

        response = client.post("/v1/cvm", content=oversize, headers=headers)
        assert response.status_code in (400, 413)


class TestBlobCanonicalization:
    """Length prefixes make the encoding injective — one blob has exactly one parse."""

    def test_field_boundary_shift_changes_the_blob(self):
        """Under bare concatenation these two tuples produce identical bytes."""
        first = signing_blob(
            method="POST", request_target="/v1/cvm", body=b"ab", timestamp="12", nonce="0" * 32
        )
        second = signing_blob(
            method="POST", request_target="/v1/cvm", body=b"a", timestamp="b12", nonce="0" * 32
        )
        assert first != second

    def test_target_boundary_shift_changes_the_blob(self):
        first = signing_blob(
            method="GET", request_target="/v1/state", body=b"", timestamp="1", nonce="0" * 32
        )
        second = signing_blob(
            method="GE", request_target="T/v1/state", body=b"", timestamp="1", nonce="0" * 32
        )
        assert first != second

    def test_signature_does_not_transfer_between_colliding_tuples(self, client, platform_key):
        """The canonicalization difference is enforced end-to-end, not just in the helper."""
        headers = sign_headers(
            platform_key, method="POST", request_target="/v1/cvm", body=b'{"kind":"renter"}'
        )
        response = client.post("/v1/cvm", content=b'{"kind":"renter"} ', headers=headers)
        assert response.status_code == 401

    def test_domain_separator_is_present(self):
        """A cvmd blob must never be a valid message in another protocol signed by the same key."""
        import hashlib

        from cvmd.auth.blob import DOMAIN_SEPARATOR

        undomained = hashlib.sha256(
            (0).to_bytes(4, "big") * 5  # five empty length-prefixed fields, no separator
        ).digest()
        actual = signing_blob(method="", request_target="", body=b"", timestamp="", nonce="")
        assert actual != undomained
        assert DOMAIN_SEPARATOR == b"cvmd-v1\x00"
