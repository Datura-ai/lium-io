"""What a renter CVM proves it is, and which GPUs it says it is holding while it proves it.

A TDX quote carries 64 bytes of caller-supplied `report_data`, and those 64 bytes are the
only thing binding the hardware's signature to anything outside the hardware. DAH-2579
spends them like this:

    report_data = sha256("LIUM_RENTER_ATTEST_TLS_V1\\0" ‖ tls_spki_der ‖ gpu_uuid_digest) ‖ nonce
                  |_____________________________ 32 bytes ____________________________|  |_ 32 _|
                                        identity                                          freshness

**Why a versioned domain tag.** A bare `sha256(key ‖ digest)` states no purpose, so nothing
distinguishes it from any other hash anyone ever takes over the same key material — and this
key is a TLS key, which lives to be hashed by other protocols. The tag gives these 32 bytes
one meaning, and carries the version *inside* the hashed bytes: a future recipe is a new tag
rather than a silent reinterpretation of the old one, so a verifier accepts the constructions
it knows and cannot be talked into reading one as another. The trailing NUL terminates the
tag — without it, a tag that is a prefix of a longer future tag could be made to collide by
shifting bytes across the boundary. The validator mirrors the constant in
`validators/src/services/attestation_service.py`; the two must agree byte for byte.

**Why the TLS key, and in which encoding.** The validator's only channel into a renter CVM is
this agent's TLS endpoint (FR-F3). Without the key in the quote, a quote proves that *some*
TDX guest exists and says nothing about whether it is the one on the other end of the
connection — an attacker who can terminate TLS relays a genuine quote from a genuine CVM it
does not own. Hashing the key in makes the proof and the channel the same thing. The key is
hashed as its **SubjectPublicKeyInfo DER**, which is what `tls.py` reads back out of the
certificate and therefore what a client sees on the wire; the encoding is part of the recipe
because two encodings of one key are two different identities to a verifier that only holds
the wire bytes. It is the renter-side counterpart of the validation CVM's `SSH_HOST_KEY:`
binding in `executor/src/services/tdx_service.py`, and it exists for the same reason.

**Why the GPU UUIDs — and what they are not.** FR-G6: "every trust check must state which
physical GPUs the CVM holds", because those identifiers otherwise travel on a channel no
proof covers. Folding their digest in means the *host* cannot swap the GPU claim in flight
and the claim cannot be lifted onto another quote: what the measured guest read is what the
hardware signed.

That is the whole of it. NVML UUIDs are read by the guest; they are not authenticated by
NVIDIA's attestation evidence, so this binding does not establish that those GPUs exist, are
genuine, are in confidential-compute mode, or are not also answering for another node. The
identifier that does is the per-GPU `ueid` claim in the NRAS-verified evidence this agent
returns alongside the quote — the one GPU identifier here that a node cannot choose. Existence,
authenticity, counting and cross-node uniqueness are decided on the ueids, on the validator's
side; the digest below is an observation, bound so that it cannot be tampered with in transit.

**Why a nonce in the other half.** FR-G1: old proofs cannot be replayed. dstack zero-pads
`report_data` when it is short, so the second half is free and is exactly 32 bytes wide.

The inputs are ordered and length-fixed — a constant tag, one DER blob, a 32-byte digest, a
32-byte nonce — so the encoding is unambiguous without length prefixes. `gpu_uuid_digest`
folds a variable number of identifiers into a fixed width before it gets here.
"""

import hashlib

NONCE_BYTES = 32
DIGEST_BYTES = 32
REPORT_DATA_BYTES = NONCE_BYTES + DIGEST_BYTES

# The domain-separation tag for the renter recipe, hashed ahead of everything else. See
# "Why a versioned domain tag" above. Mirrored by the validator; changing either side alone
# fails every honest node.
RENTER_IDENTITY_DOMAIN_V1 = b"LIUM_RENTER_ATTEST_TLS_V1\x00"

# The separator between UUIDs inside the digest. A character that cannot occur in an NVIDIA
# GPU UUID, so no two different GPU sets can produce the same joined string — with a
# separator that could appear in a UUID, {"a,b"} and {"a", "b"} would digest identically.
_UUID_SEPARATOR = ","


class IdentityError(Exception):
    """The agent cannot state who it is. Never answered around — a quote without a correct
    binding is worse than no quote, because it looks like proof."""


def gpu_uuid_digest(uuids: list[str]) -> bytes:
    """A fixed-width digest of the GPU set this CVM *observes* through NVML.

    Observes, not proves — see the module docstring. Binding it makes the observation
    tamper-evident in transit; it does not make it an attested identity.

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


def identity_digest(tls_public_key_der: bytes, uuids: list[str]) -> bytes:
    """The 32-byte identity half: this agent's TLS channel, and the GPU set it observes.

    `tls_public_key_der` is the SubjectPublicKeyInfo DER — the encoding `tls.py` reads out
    of the certificate, and the one a client sees on the wire. Not "some encoding of the
    key": the recipe fixes it, because a verifier holding the wire bytes can only reproduce
    this digest if it hashes them the same way.
    """
    if not tls_public_key_der:
        raise IdentityError("the agent has no TLS public key, so it can prove nothing")
    return hashlib.sha256(
        RENTER_IDENTITY_DOMAIN_V1 + tls_public_key_der + gpu_uuid_digest(uuids)
    ).digest()


def report_data(tls_public_key_der: bytes, uuids: list[str], nonce: bytes) -> bytes:
    """The full 64 bytes handed to the TDX quote."""
    if len(nonce) != NONCE_BYTES:
        raise IdentityError(f"the nonce must be exactly {NONCE_BYTES} bytes, got {len(nonce)}")
    return identity_digest(tls_public_key_der, uuids) + nonce


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
