"""The canonical signed blob.

This module is the whole wire contract between cvmd and its clients (DAH-2576 connector,
DAH-2580 validator). Anything that changes here is a breaking protocol change — the golden
vectors in tests/ double as the client-side reference implementation.

    blob = sha256(
        b"cvmd-v1\\x00"                     # domain separator
      | lp(method) | lp(request_target)     # request_target = path + query, exactly as received
      | lp(body_bytes)
      | lp(timestamp_ascii) | lp(nonce_ascii)
    )
        where lp(x) = uint32_be(len(x)) | x

Two deviations from the architecture doc §03 formula, both deliberate:

  * **Method and request target are signed.** The doc omits them. Without them a captured
    `POST /v1/cvm` signature replays as `DELETE /v1/cvm` inside the freshness window — live,
    not theoretical, because the platform key holds both verbs. Signing the query matters for
    DAH-2578's `/v1/catalog`, whose parameters would otherwise be attacker-controlled.
  * **Fields are length-prefixed.** The doc writes bare concatenation, which is not injective:
    a body ending in digits followed by a short timestamp can produce the same bytes as a
    shorter body and a longer timestamp. Length prefixes give one blob exactly one parse.

The domain separator stops a cvmd blob from ever being a valid message in another protocol
signed by the same hotkey.
"""

import hashlib

DOMAIN_SEPARATOR = b"cvmd-v1\x00"

# uint32 big-endian length prefix — a field longer than this cannot be signed. The body cap is
# orders of magnitude below it; the bound exists so the encoding is total, not as a policy.
MAX_FIELD_BYTES = 0xFFFFFFFF


def _lp(value: bytes) -> bytes:
    if len(value) > MAX_FIELD_BYTES:
        raise ValueError(f"field of {len(value)} bytes exceeds the length prefix")
    return len(value).to_bytes(4, "big") + value


def signing_blob(
    *,
    method: str,
    request_target: str,
    body: bytes,
    timestamp: str,
    nonce: str,
) -> bytes:
    """Return the 32-byte digest that a client signs and cvmd verifies.

    `timestamp` and `nonce` are the header values verbatim, as ASCII — not re-formatted. A
    client that sends "007" signs "007"; normalizing here would let two distinct headers produce
    one blob.
    """
    return hashlib.sha256(
        DOMAIN_SEPARATOR
        + _lp(method.encode())
        + _lp(request_target.encode())
        + _lp(body)
        + _lp(timestamp.encode())
        + _lp(nonce.encode())
    ).digest()
