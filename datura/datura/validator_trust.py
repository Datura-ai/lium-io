"""Which validator a node trusts: the one named in a message signed by the SN51 subnet owner coldkey.

The subnet owner signs one line, `lium-validator:<validator hotkey ss58>:<revision>`, built by
`trust_line` so every component signs and checks the same text. A node accepts a message whose
signature matches the owner coldkey and whose revision is higher than the one it holds; on any error
it keeps the last good one. The owner coldkey is read from chain (`SubtensorModule.SubnetOwner`)
on public nodes only, with one plain `state_getStorage` request (the caller sends it and bounds it), so a validator key change needs a new
signed message and no release.
"""

from __future__ import annotations

import re
from typing import NamedTuple

NETUID = 51
TRUST_LABEL = "lium-validator"
# The owner known at release time: a fresh node starts from it and needs no chain read to begin.
RELEASE_SUBNET_OWNER_COLDKEY = "5FqACMtcegZxxopgu1g7TgyrnyD8skurr9QDPLPhxNQzsThe"  # gitleaks:allow public on chain: the SN51 owner ss58
# Public chain nodes only: this code runs on provider machines.
PUBLIC_CHAIN_NODES = (
    "https://entrypoint-finney.opentensor.ai",
    "https://lite.chain.opentensor.ai",
)
# twox128("SubtensorModule") + twox128("SubnetOwner"); the map key (netuid, u16) is Identity-hashed.
_SUBNET_OWNER_COLDKEY_PREFIX = "0x658faa385070e074c85bf6b568cf055536e3e82152c8758267395fe524fbbd16"  # gitleaks:allow public on chain: storage prefix
_SS58 = "[1-9A-HJ-NP-Za-km-z]{46,48}"
_TRUST_LINE = re.compile(rf"{TRUST_LABEL}:({_SS58}):([0-9]{{1,18}})")
_SIGNATURE = re.compile(r"0x[0-9a-fA-F]{128}")


class TrustedValidator(NamedTuple):
    validator: str
    revision: int


class SignedTrustMessage(NamedTuple):
    message: str
    signature: str


def trust_line(validator_ss58: str, revision: int) -> str:
    """The exact text the subnet owner signs."""
    return f"{TRUST_LABEL}:{validator_ss58}:{revision}"


def verify_trust_message(signed: SignedTrustMessage, owner_ss58: str) -> TrustedValidator:
    """The validator and revision a message names once its signature is checked against `owner_ss58`.

    Raises ValueError on a malformed line or signature, an invalid validator address, or a signature
    that is not the owner's.
    """
    from bittensor_wallet import Keypair

    match = _TRUST_LINE.fullmatch(signed.message)
    if match is None:
        raise ValueError(f"not a {TRUST_LABEL} line")
    if _SIGNATURE.fullmatch(signed.signature) is None:
        raise ValueError("the signature is not 0x and 64 bytes of hex")
    validator, revision = match.group(1), int(match.group(2))
    Keypair(ss58_address=validator)  # raises ValueError on a bad checksum
    if trust_line(validator, revision) != signed.message:
        raise ValueError("the line is not in its canonical form")
    if not Keypair(ss58_address=owner_ss58).verify(signed.message, signed.signature):
        raise ValueError("not signed by the subnet owner coldkey")
    return TrustedValidator(validator, revision)


def subnet_owner_storage_key(netuid: int = NETUID) -> str:
    return _SUBNET_OWNER_COLDKEY_PREFIX + netuid.to_bytes(2, "little").hex()


def subnet_owner_request(netuid: int = NETUID) -> dict:
    """The JSON-RPC request a chain node answers with the SN `netuid` owner coldkey."""
    return {"jsonrpc": "2.0", "id": 1, "method": "state_getStorage", "params": [subnet_owner_storage_key(netuid)]}


def subnet_owner_from_reply(reply: dict) -> str:
    """The owner coldkey (ss58) in a chain node's reply to `subnet_owner_request`. Raises ValueError on anything else."""
    from scalecodec.utils.ss58 import ss58_encode

    result = reply.get("result") if isinstance(reply, dict) else None
    if not isinstance(result, str) or not re.fullmatch(r"0x[0-9a-fA-F]{64}", result):
        raise ValueError(f"unexpected SubnetOwner value: {result!r}")
    return ss58_encode(result, 42)


def override_trust(module: str) -> dict:
    """The ValidatorTrust arguments for this build. A release build has no `module` (its config_override) and gets {}:
    the release owner, re-read from chain. A non-prod build that names a staging owner (`_STAGING_TRUST_OWNER_SS58`)
    follows messages that key signed, never reads the chain, and starts from its own signed message
    (`_STAGING_TRUST_MESSAGE`, {"message", "signature"}) instead of the image copy, and reads new ones from the backend
    that serves them (`_STAGING_TRUST_API_URL`), never the host's own backend setting. A non-prod build naming no owner
    keeps message trust off and trusts its baked validator, as before. A malformed or missing owner, message or URL
    fails the start."""
    import importlib
    import importlib.util

    from bittensor_wallet import Keypair

    if importlib.util.find_spec(module) is None:
        return {}
    config = importlib.import_module(module)
    owner = getattr(config, "_STAGING_TRUST_OWNER_SS58", None)
    if not owner:
        return {"enabled": False}
    Keypair(ss58_address=owner)  # raises ValueError on a bad address
    message = getattr(config, "_STAGING_TRUST_MESSAGE", None)
    signed = SignedTrustMessage(message=str(message["message"]), signature=str(message["signature"])) if message else None
    if signed is not None:
        verify_trust_message(signed, owner)
    api_url = getattr(config, "_STAGING_TRUST_API_URL", None)
    if not isinstance(api_url, str) or not re.fullmatch(r"https://[A-Za-z0-9.-]+(/[A-Za-z0-9._-]+)*", api_url):
        raise ValueError(f"{module}: a staging owner needs _STAGING_TRUST_API_URL, the https backend of its messages")
    return {"owner": owner, "read_chain": False, "message": signed, "api_url": api_url}
