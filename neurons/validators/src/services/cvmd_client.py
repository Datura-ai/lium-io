"""Sign and make this validator's own cvmd calls (DAH-2629).

Until now the validator only *relayed* cvmd calls the backend signed (DAH-2580). The
validation-CVM lifecycle is different: cvmd's scope table gives ``validation`` to the
validator key alone, so launching a validation CVM is a call this process signs itself,
with the same wallet it signs everything else with.

The signing half is a mirror of ``lium-io-backend: services/cvmd_request.py`` — one wire
protocol, two signers with two scopes. cvmd's ``tests/test_golden_vector.py`` is the
committed fixture that pins the blob formula across both repos; the formula test in this
repo's DAH-2629 suite pins this mirror to the same bytes.

What this client can and cannot do is decided by cvmd, not here:

    POST /v1/cvm {kind: validation}   allowed — the validator's own scope
    GET  /v1/state, /v1/catalog       allowed — reads are open to either authorized key
    POST /v1/cvm {kind: renter}       403 — platform scope, this key is refused
    DELETE /v1/cvm                    403 — teardown is the platform's alone

So "the validator never destroys a CVM" stays a property of what this process can sign.

Transport is the existing ``CvmdRelay`` — same TLS stance (the host's certificate is
self-signed; the request signature and the measured launch report are what authenticate
this exchange), same no-retry stance (a repeated launch lands a second request on a node
already committed to the first; cvmd's own 409 is the better answer).
"""

import hashlib
import json
import logging
import secrets
import time
from dataclasses import dataclass

from services.cvmd_relay import (
    DEFAULT_PROVISION_TIMEOUT_SECONDS,
    CvmdRelay,
    CvmdRelayError,
    RelayResult,
)

logger = logging.getLogger(__name__)

DOMAIN_SEPARATOR = b"cvmd-v1\x00"

HOTKEY_HEADER = "X-Cvmd-Hotkey"
TIMESTAMP_HEADER = "X-Cvmd-Timestamp"
NONCE_HEADER = "X-Cvmd-Nonce"
SIGNATURE_HEADER = "X-Cvmd-Signature"

NONCE_BYTES = 16

CVM_PATH = "/v1/cvm"
STATE_PATH = "/v1/state"

KIND_VALIDATION = "validation"

# A state read answers from memory; only the network is being waited on.
DEFAULT_STATE_TIMEOUT_SECONDS = 30


def _lp(value: bytes) -> bytes:
    return len(value).to_bytes(4, "big") + value


def signing_blob(*, method: str, request_target: str, body: bytes, timestamp: str, nonce: str) -> bytes:
    """The 32-byte digest cvmd recomputes and verifies. Byte-identical to the backend's."""
    return hashlib.sha256(
        DOMAIN_SEPARATOR
        + _lp(method.encode())
        + _lp(request_target.encode())
        + _lp(body)
        + _lp(timestamp.encode())
        + _lp(nonce.encode())
    ).digest()


@dataclass(frozen=True)
class SignedCall:
    method: str
    path: str
    body: str
    headers: dict[str, str]


def sign_request(keypair, *, method: str, path: str, body: str = "") -> SignedCall:
    """Sign one cvmd call with this validator's keypair.

    ``path`` is the request target — path plus query, exactly as sent — and it is inside
    the blob, so a captured call cannot be redirected or replayed as a different verb.
    """
    timestamp = str(time.time_ns())
    nonce = secrets.token_hex(NONCE_BYTES)
    blob = signing_blob(
        method=method,
        request_target=path,
        body=body.encode(),
        timestamp=timestamp,
        nonce=nonce,
    )
    return SignedCall(
        method=method,
        path=path,
        body=body,
        headers={
            HOTKEY_HEADER: keypair.ss58_address,
            TIMESTAMP_HEADER: timestamp,
            NONCE_HEADER: nonce,
            SIGNATURE_HEADER: keypair.sign(blob).hex(),
        },
    )


def canonical_body(payload: dict) -> str:
    """Serialize once, with pinned separators — the exact bytes that are signed and sent."""
    return json.dumps(payload, separators=(",", ":"))


class CvmdClient:
    """This validator's own cvmd calls: read a node's state, launch its validation CVM."""

    def __init__(self, keypair, *, relay: CvmdRelay | None = None) -> None:
        self._keypair = keypair
        self._relay = relay or CvmdRelay()

    async def state(self, base_url: str) -> RelayResult:
        """Signed ``GET /v1/state``: the node's mode, its CVM if any, and its last switch."""
        signed = sign_request(self._keypair, method="GET", path=STATE_PATH)
        return await self._relay.forward(
            base_url=base_url,
            method=signed.method,
            path=signed.path,
            body=signed.body,
            headers=signed.headers,
            timeout_seconds=DEFAULT_STATE_TIMEOUT_SECONDS,
        )

    async def launch_validation(
        self, base_url: str, *, qemu: str, os_image_hash: str, compose_hash: str
    ) -> RelayResult:
        """Signed ``POST /v1/cvm {kind: validation}`` naming the pinned triple.

        The triple is required by cvmd's contract — a host that chose the stack itself
        would give this validator no way to tell "the CVM I asked for" from "whatever the
        host felt like running". Slow on purpose: the call holds until the guest boots.
        """
        body = canonical_body(
            {
                "kind": KIND_VALIDATION,
                "qemu": qemu,
                "os_image_hash": os_image_hash,
                "compose_hash": compose_hash,
            }
        )
        signed = sign_request(self._keypair, method="POST", path=CVM_PATH, body=body)
        return await self._relay.forward(
            base_url=base_url,
            method=signed.method,
            path=signed.path,
            body=signed.body,
            headers=signed.headers,
            timeout_seconds=DEFAULT_PROVISION_TIMEOUT_SECONDS,
        )


__all__ = [
    "CVM_PATH",
    "STATE_PATH",
    "KIND_VALIDATION",
    "CvmdClient",
    "CvmdRelayError",
    "SignedCall",
    "canonical_body",
    "sign_request",
    "signing_blob",
]
