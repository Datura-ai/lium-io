"""The authorized-clients file: which hotkeys may call cvmd, and in which scope.

Installed by day-zero Ansible (DAH-2544 role `cvmd`). Loading is fail-closed in every direction:
a missing file, malformed JSON, an unknown scope, or an ss58 that does not checksum all refuse
startup rather than yielding a daemon that authorizes a subset of what the operator intended.
"""

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from bittensor.sp_core import Keypair


class Scope(StrEnum):
    """What a key is allowed to do. The scopes are disjoint by design.

    ATTEST holds no launch or teardown right at all: a key in this scope can make the reads
    open to any authorized key, plus `POST /v1/attest` — which is itself open to any
    authorized key, so the scope GRANTS nothing the other two lack. It exists for the
    opposite direction: an operator can authorize a key that can request a renter CVM's
    quote and nothing else (DAH-2675).
    """

    VALIDATION = "validation"
    RENTER = "renter"
    ATTEST = "attest"


class AuthorizedClientsError(Exception):
    """Raised when the authorized-clients file cannot be loaded exactly as written. Always fatal."""


@dataclass(frozen=True)
class AuthorizedClients:
    by_hotkey: dict[str, Scope]

    def scope_for(self, hotkey: str) -> Scope | None:
        return self.by_hotkey.get(hotkey)

    def __len__(self) -> int:
        return len(self.by_hotkey)


def _validate_ss58(hotkey: str) -> None:
    """Reject a malformed address at load time rather than on the first request that uses it.

    bittensor's Keypair constructor does the ss58 decode and checksum for us, so there is no
    hand-written base58/blake2b anywhere in cvmd.
    """
    try:
        Keypair(ss58_address=hotkey)
    except (ValueError, TypeError) as exc:
        raise AuthorizedClientsError(f"not a valid ss58 address: {hotkey!r} ({exc})") from exc


def load_authorized_clients(path: Path) -> AuthorizedClients:
    try:
        with path.open("rb") as handle:
            raw = json.load(handle)
    except OSError as exc:
        raise AuthorizedClientsError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise AuthorizedClientsError(f"{path} is not valid JSON: {exc}") from exc

    if not isinstance(raw, list):
        raise AuthorizedClientsError(f"{path} must be a JSON list of {{hotkey, scope}} objects")

    by_hotkey: dict[str, Scope] = {}
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise AuthorizedClientsError(f"{path}[{index}] is not an object")

        hotkey = entry.get("hotkey")
        scope = entry.get("scope")
        if not isinstance(hotkey, str) or not hotkey:
            raise AuthorizedClientsError(f"{path}[{index}] has no `hotkey`")
        if not isinstance(scope, str):
            raise AuthorizedClientsError(f"{path}[{index}] has no `scope`")

        try:
            parsed_scope = Scope(scope)
        except ValueError as exc:
            allowed = ", ".join(s.value for s in Scope)
            raise AuthorizedClientsError(
                f"{path}[{index}] has unknown scope {scope!r} (allowed: {allowed})"
            ) from exc

        _validate_ss58(hotkey)

        # A hotkey listed twice is ambiguous — one of the two scopes would silently win.
        if hotkey in by_hotkey:
            raise AuthorizedClientsError(f"{path} lists hotkey {hotkey} more than once")
        by_hotkey[hotkey] = parsed_scope

    if not by_hotkey:
        raise AuthorizedClientsError(f"{path} authorizes no keys — cvmd would accept nothing")

    return AuthorizedClients(by_hotkey=by_hotkey)
