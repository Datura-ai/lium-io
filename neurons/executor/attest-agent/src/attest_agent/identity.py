"""What a renter CVM proves it is, and which GPUs it is holding while it proves it.

A TDX quote carries 64 bytes of caller-supplied `report_data`, and those 64 bytes are the
only thing binding the hardware's signature to anything outside the hardware. DAH-2579
spends them like this:

    report_data = sha256(agent_tls_public_key ‖ gpu_uuid_digest) ‖ nonce
                  |________________ 32 bytes _________________|  |_ 32 _|
                          identity                                freshness

**Why the TLS key.** The validator's only channel into a renter CVM is this agent's TLS
endpoint (FR-F3). Without the key in the quote, a quote proves that *some* TDX guest exists
and says nothing about whether it is the one on the other end of the connection — an
attacker who can terminate TLS relays a genuine quote from a genuine CVM it does not own.
Hashing the key in makes the proof and the channel the same thing. It is the renter-side
counterpart of the validation CVM's `SSH_HOST_KEY:` binding in
`executor/src/services/tdx_service.py`, and it exists for the same reason.

**Why the GPU UUIDs.** FR-G6: "every trust check must state which physical GPUs the CVM
holds", because today those identifiers travel on a channel no proof covers and a node can
claim GPUs it does not have. Folding their digest into the same 32 bytes means the GPU set
is attested by the hardware rather than asserted by software the host controls.

**Why a nonce in the other half.** FR-G1: old proofs cannot be replayed. dstack zero-pads
`report_data` when it is short, so the second half is free and is exactly 32 bytes wide.

The three inputs are ordered and length-fixed — a 32-byte digest and a 32-byte nonce — so
the encoding is unambiguous without length prefixes. `gpu_uuid_digest` folds a variable
number of identifiers into a fixed width before it gets here.
"""

import hashlib

NONCE_BYTES = 32
DIGEST_BYTES = 32
REPORT_DATA_BYTES = NONCE_BYTES + DIGEST_BYTES

# The separator between UUIDs inside the digest. A character that cannot occur in an NVIDIA
# GPU UUID, so no two different GPU sets can produce the same joined string — with a
# separator that could appear in a UUID, {"a,b"} and {"a", "b"} would digest identically.
_UUID_SEPARATOR = ","


class IdentityError(Exception):
    """The agent cannot state who it is. Never answered around — a quote without a correct
    binding is worse than no quote, because it looks like proof."""


def gpu_uuid_digest(uuids: list[str]) -> bytes:
    """A fixed-width digest of the GPU set this CVM holds.

    Sorted, so the digest depends on which GPUs are present and not on the order NVML
    happened to enumerate them in — an ordering that varies with PCI topology and would
    otherwise make the same node attest differently after a reboot.

    A CVM with no GPUs is legal (a validation CVM on a host with none), and it digests to
    the sha256 of the empty string rather than to a special case, so every quote carries a
    GPU binding of the same shape.
    """
    for uuid in uuids:
        if not uuid or _UUID_SEPARATOR in uuid:
            raise IdentityError(
                f"{uuid!r} is not a usable GPU identifier: it is empty or contains "
                f"{_UUID_SEPARATOR!r}, which would make two different GPU sets digest alike"
            )
    joined = _UUID_SEPARATOR.join(sorted(uuids))
    return hashlib.sha256(joined.encode()).digest()


def identity_digest(tls_public_key: bytes, uuids: list[str]) -> bytes:
    """The 32-byte identity half: this agent's TLS key bound to this CVM's GPUs."""
    if not tls_public_key:
        raise IdentityError("the agent has no TLS public key, so it can prove nothing")
    return hashlib.sha256(tls_public_key + gpu_uuid_digest(uuids)).digest()


def report_data(tls_public_key: bytes, uuids: list[str], nonce: bytes) -> bytes:
    """The full 64 bytes handed to the TDX quote."""
    if len(nonce) != NONCE_BYTES:
        raise IdentityError(f"the nonce must be exactly {NONCE_BYTES} bytes, got {len(nonce)}")
    return identity_digest(tls_public_key, uuids) + nonce


def parse_nonce(value: str) -> bytes:
    """Decode a caller-issued nonce, or refuse it.

    Refused rather than padded or truncated. A nonce is the verifier's challenge, and an
    agent that quietly reshapes one produces a quote the verifier cannot match against what
    it issued — which reads as a failed attestation rather than as a malformed request.
    """
    try:
        nonce = bytes.fromhex(value)
    except (ValueError, TypeError) as exc:
        raise IdentityError(f"the nonce must be hex, got {value!r}") from exc
    if len(nonce) != NONCE_BYTES:
        raise IdentityError(
            f"the nonce must be exactly {NONCE_BYTES} bytes ({NONCE_BYTES * 2} hex chars), "
            f"got {len(nonce)}"
        )
    return nonce
