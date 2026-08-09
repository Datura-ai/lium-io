"""The signed catalog manifest: what the platform approved, and proof that it said so.

The manifest is the wire contract between the backend's artifact catalog (DAH-2578) and every
host. It is one JSON object:

    {
      "schema":    "lium-cvm-catalog/1",
      "payload":   "<the catalog document, as JSON text>",
      "signer":    "<ss58 of the platform's signing hotkey>",
      "signature": "0x<sr25519 signature>"
    }

    signed bytes = b"lium-cvm-catalog-v1\\x00" | payload.encode()

**`payload` is a string, not an object.** The signature covers exactly the bytes the platform
serialized, so nothing here has to reproduce the backend's JSON formatting to check it. The
alternative — signing a re-serialized object — makes correctness depend on key order, unicode
escaping and float spelling agreeing across two languages and two libraries, and every one of
those is a silent verification bypass when it drifts.

The domain separator stops a catalog manifest from ever being a valid message in another
protocol signed by the same hotkey — the same reasoning as `auth/blob.py`.

Four things are checked, and each one closes a different attack:

  signer      The signature is verified against the ss58 **the host was configured with**, never
              against the one the document carries. A document checked against its own claimed
              signer proves only that someone owns a key.
  freshness   `expires_at` must be in the future. A manifest that never expired would let a
              revocation be defeated by simply never delivering the next one.
  serial      Monotonic. Without it, a validly signed *older* manifest replays a revoked
              artifact back into the catalog.
  floors      Monotonic too, per artifact class. The floor is the version ratchet the backend
              maintains; letting a host accept a lower one would undo the ratchet remotely.

Anything wrong is a `CatalogError`, and a `CatalogError` refuses every launch rather than some.
A manifest is approved as a whole or not at all: dropping the entries that look wrong and
keeping the rest would leave a host launching from a document nobody wrote.
"""

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from bittensor.sp_core import Keypair

from cvmd.catalog.artifacts import CatalogError, assert_hex64

SCHEMA = "lium-cvm-catalog/1"
DOMAIN_SEPARATOR = b"lium-cvm-catalog-v1\x00"

# The artifact classes the backend ratchets independently. A stack entry names its version of
# each; the manifest's floors say how low each may go. Kept as a frozen set rather than derived
# from the document so an unknown class is a refusal, not a silently unratcheted component.
COMPONENT_CLASSES = ("os_image", "qemu", "compose")

# How far ahead of this host's clock a manifest may claim to have been issued. Hosts and the
# backend both run NTP; this exists so a second of skew is not a fleet-wide outage, not as a
# licence to accept a document from next week.
ISSUED_AT_SKEW_SECONDS = 300


@dataclass(frozen=True)
class Entry:
    """One launchable stack, exactly as the platform published it.

    Everything that ends up inside the measurement travels here as **content**, not as a path:
    the compose text and both scripts. A host that supplied its own copy of any of them could
    launch a CVM measuring as something the platform never approved, and the compose hash in the
    triple would be the only thing that noticed — after the fact, on the next attestation.

    The OS image is the exception, and it has to be: it is gigabytes, and it is staged on the
    host by day-zero Ansible. `os_image_name` names which staged image to use and
    `os_image_hash` is what it must measure to, which `cvm/measure.py` checks against the
    image's own `digest.txt` before QEMU is started.
    """

    id: str
    kind: str
    qemu: str
    os_image_hash: str
    os_image_name: str
    compose_hash: str
    compose: str
    versions: dict[str, int]
    init_script: str | None = None
    pre_launch_script: str | None = None
    local_key_provider: bool = True
    enable_logs: bool = False
    enable_sysinfo: bool = False


@dataclass(frozen=True)
class Manifest:
    serial: int
    issued_at: datetime
    expires_at: datetime
    floors: dict[str, int]
    entries: list[Entry] = field(default_factory=list)
    signer: str = ""
    # The bytes this manifest was read from, so a client can cache exactly what it verified.
    raw: bytes = b""

    def report(self) -> dict:
        """What `/v1/catalog` and the logs say about the manifest in force."""
        return {
            "serial": self.serial,
            "issued_at": self.issued_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "signer": self.signer,
            "floors": dict(self.floors),
            "entries": [
                {
                    "id": entry.id,
                    "kind": entry.kind,
                    "qemu": entry.qemu,
                    "os_image_hash": entry.os_image_hash,
                    "compose_hash": entry.compose_hash,
                    "versions": dict(entry.versions),
                }
                for entry in self.entries
            ],
        }

    def is_expired(self, now: datetime | None = None) -> bool:
        return (now or datetime.now(UTC)) >= self.expires_at


def _require(raw: dict, key: str, where: str):
    if key not in raw:
        raise CatalogError(f"{where}: `{key}` is missing")
    return raw[key]


def _require_str(raw: dict, key: str, where: str) -> str:
    value = _require(raw, key, where)
    if not isinstance(value, str) or not value.strip():
        raise CatalogError(f"{where}: `{key}` must be a non-empty string")
    return value.strip()


def _require_text(raw: dict, key: str, where: str) -> str:
    """Like `_require_str`, but returns the value **exactly**.

    For the two fields whose bytes are the point: the signed `payload`, and the `compose` whose
    sha256 is a third of the attested triple. `_require_str` strips, and a stripped trailing
    newline is a different compose hash — a launch that would then be refused as "not approved",
    with nothing anywhere naming whitespace as the cause.
    """
    value = _require(raw, key, where)
    if not isinstance(value, str) or not value.strip():
        raise CatalogError(f"{where}: `{key}` must be a non-empty string")
    return value


def _optional_str(raw: dict, key: str, where: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise CatalogError(f"{where}: `{key}` must be a string when present")
    return value


def _require_bool(raw: dict, key: str, default: bool, where: str) -> bool:
    value = raw.get(key, default)
    if not isinstance(value, bool):
        raise CatalogError(f"{where}: `{key}` must be a boolean, got {value!r}")
    return value


def _require_int(raw: dict, key: str, where: str) -> int:
    value = _require(raw, key, where)
    # bool is an int in Python, and `True` reaching a version comparison as 1 is exactly the
    # kind of thing that reads correct and ratchets nothing.
    if isinstance(value, bool) or not isinstance(value, int):
        raise CatalogError(f"{where}: `{key}` must be an integer, got {value!r}")
    return value


def _require_time(raw: dict, key: str, where: str) -> datetime:
    text = _require_str(raw, key, where)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise CatalogError(f"{where}: `{key}` is not an ISO-8601 timestamp: {text!r}") from exc
    if parsed.tzinfo is None:
        # A naive timestamp would be compared against an aware `now` and raise, or worse be
        # coerced to the host's local zone. Neither is a freshness check.
        raise CatalogError(f"{where}: `{key}` must carry a timezone, got {text!r}")
    return parsed.astimezone(UTC)


def _decode_versions(raw: dict, where: str) -> dict[str, int]:
    versions = _require(raw, "versions", where)
    if not isinstance(versions, dict):
        raise CatalogError(f"{where}: `versions` must be an object")
    decoded = {}
    for component in COMPONENT_CLASSES:
        decoded[component] = _require_int(versions, component, f"{where}.versions")
    unknown = sorted(set(versions) - set(COMPONENT_CLASSES))
    if unknown:
        raise CatalogError(
            f"{where}.versions carries {', '.join(unknown)}, which this cvmd does not ratchet — "
            f"refusing rather than accepting a component it cannot floor"
        )
    return decoded


def _decode_entry(raw, index: int) -> Entry:
    where = f"artifacts[{index}]"
    if not isinstance(raw, dict):
        raise CatalogError(f"{where} must be an object")

    entry = Entry(
        id=_require_str(raw, "id", where),
        kind=_require_str(raw, "kind", where),
        qemu=_require_str(raw, "qemu", where),
        os_image_hash=_require_str(raw, "os_image_hash", where),
        os_image_name=_require_str(raw, "os_image_name", where),
        compose_hash=_require_str(raw, "compose_hash", where),
        compose=_require_text(raw, "compose", where),
        versions=_decode_versions(raw, where),
        init_script=_optional_str(raw, "init_script", where),
        pre_launch_script=_optional_str(raw, "pre_launch_script", where),
        local_key_provider=_require_bool(raw, "local_key_provider", True, where),
        enable_logs=_require_bool(raw, "enable_logs", False, where),
        enable_sysinfo=_require_bool(raw, "enable_sysinfo", False, where),
    )
    assert_hex64(entry.os_image_hash, field="os_image_hash", where=where)
    assert_hex64(entry.compose_hash, field="compose_hash", where=where)

    # `os_image_name` is joined onto a host-local directory. A name with a separator or a `..`
    # in it escapes that directory, which would let the manifest choose any path on the host as
    # the image — including one whose digest.txt says whatever the attacker wants.
    if "/" in entry.os_image_name or entry.os_image_name in (".", ".."):
        raise CatalogError(
            f"{where}: `os_image_name` must be a single directory name under the host's image "
            f"directory, got {entry.os_image_name!r}"
        )
    return entry


def _decode_payload(payload: str) -> tuple[int, datetime, datetime, dict[str, int], list[Entry]]:
    try:
        raw = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise CatalogError(f"the manifest payload is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise CatalogError("the manifest payload must be a JSON object")

    version = raw.get("version")
    if version != 1:
        raise CatalogError(
            f"manifest payload version {version!r} is not the supported 1 — refusing rather "
            f"than guessing at the shape"
        )

    serial = _require_int(raw, "serial", "payload")
    if serial < 0:
        raise CatalogError(f"payload: `serial` must not be negative, got {serial}")

    issued_at = _require_time(raw, "issued_at", "payload")
    expires_at = _require_time(raw, "expires_at", "payload")
    if expires_at <= issued_at:
        raise CatalogError(
            f"payload: `expires_at` ({expires_at.isoformat()}) is not after `issued_at` "
            f"({issued_at.isoformat()}), so this manifest was never valid"
        )

    floors_raw = _require(raw, "floors", "payload")
    if not isinstance(floors_raw, dict):
        raise CatalogError("payload: `floors` must be an object")
    floors = {c: _require_int(floors_raw, c, "payload.floors") for c in COMPONENT_CLASSES}

    entries_raw = raw.get("artifacts")
    if not isinstance(entries_raw, list):
        raise CatalogError("payload: `artifacts` must be a list")
    entries = [_decode_entry(entry, index) for index, entry in enumerate(entries_raw)]

    # The backend excludes below-floor artifacts when it builds a manifest. Re-checking here is
    # not distrust of the backend so much as of the path between them: this is the one place a
    # host can tell that the ratchet it was told about and the list it was given disagree.
    for index, entry in enumerate(entries):
        for component, floor in floors.items():
            if entry.versions[component] < floor:
                raise CatalogError(
                    f"artifacts[{index}] ({entry.id}) carries {component} version "
                    f"{entry.versions[component]}, below this manifest's own floor of {floor}"
                )

    seen: dict[str, int] = {}
    for index, entry in enumerate(entries):
        if entry.id in seen:
            raise CatalogError(
                f"artifacts[{index}] repeats the id {entry.id!r} from artifacts[{seen[entry.id]}]"
            )
        seen[entry.id] = index

    return serial, issued_at, expires_at, floors, entries


def _verify_signature(*, signer: str, payload: str, signature: str) -> None:
    text = signature[2:] if signature.startswith("0x") else signature
    try:
        raw_signature = bytes.fromhex(text)
    except ValueError as exc:
        raise CatalogError(f"the manifest signature is not hex: {signature!r}") from exc

    try:
        keypair = Keypair(ss58_address=signer)
    except (ValueError, TypeError) as exc:
        raise CatalogError(f"the configured manifest signer is not a valid ss58: {exc}") from exc

    blob = DOMAIN_SEPARATOR + payload.encode()
    try:
        verified = keypair.verify(blob, raw_signature)
    except (ValueError, TypeError) as exc:
        raise CatalogError(f"the manifest signature could not be checked: {exc}") from exc
    if not verified:
        raise CatalogError(
            f"the manifest is not signed by {signer} — it was rejected without being read"
        )


def parse_manifest(
    raw: bytes,
    *,
    signer: str,
    now: datetime | None = None,
    require_fresh: bool = True,
) -> Manifest:
    """Verify a manifest and decode it, or raise `CatalogError` saying which check failed.

    `signer` is the ss58 the *host* was configured with. The `signer` field inside the document
    is only ever compared against it — never used in its place.

    `require_fresh=False` is for the cache-loading path, which needs to be able to read an
    expired manifest in order to say so precisely; every launch path passes the default.
    """
    now = now or datetime.now(UTC)

    try:
        envelope = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise CatalogError(f"the manifest is not valid JSON: {exc}") from exc
    if not isinstance(envelope, dict):
        raise CatalogError("the manifest must be a JSON object")

    schema = envelope.get("schema")
    if schema != SCHEMA:
        raise CatalogError(
            f"manifest schema {schema!r} is not the supported {SCHEMA!r} — refusing rather than "
            f"guessing at the shape"
        )

    payload = _require_text(envelope, "payload", "manifest")
    claimed_signer = _require_str(envelope, "signer", "manifest")
    signature = _require_str(envelope, "signature", "manifest")

    if not signer:
        raise CatalogError(
            "this host has no configured manifest signer, so no manifest can be trusted; set "
            "`catalog_signer` in /etc/cvmd/config.toml"
        )
    if claimed_signer != signer:
        raise CatalogError(
            f"the manifest says it was signed by {claimed_signer}, but this host only trusts "
            f"{signer}"
        )

    _verify_signature(signer=signer, payload=payload, signature=signature)

    serial, issued_at, expires_at, floors, entries = _decode_payload(payload)

    if issued_at > now.astimezone(UTC) + timedelta(seconds=ISSUED_AT_SKEW_SECONDS):
        raise CatalogError(
            f"the manifest claims to have been issued at {issued_at.isoformat()}, which is "
            f"further ahead of this host's clock than {ISSUED_AT_SKEW_SECONDS}s"
        )
    if require_fresh and now >= expires_at:
        raise CatalogError(
            f"the manifest expired at {expires_at.isoformat()} and this host will not launch "
            f"from a catalog it can no longer confirm is current"
        )

    return Manifest(
        serial=serial,
        issued_at=issued_at,
        expires_at=expires_at,
        floors=floors,
        entries=entries,
        signer=signer,
        raw=raw,
    )


def assert_not_rollback(candidate: Manifest, current: Manifest | None) -> None:
    """Refuse a manifest that would move this host backwards.

    A signature proves the platform published a document; it says nothing about *when*. Both
    guards below exist because an attacker who can serve bytes to a host — a compromised CDN, a
    stale cache, a rewound backend replica — can serve a genuinely signed manifest from before
    a revocation and undo it without forging anything.
    """
    if current is None:
        return
    if candidate.serial < current.serial:
        raise CatalogError(
            f"the fetched manifest has serial {candidate.serial}, older than the {current.serial} "
            f"this host already holds — refused as a rollback"
        )
    for component, floor in current.floors.items():
        if candidate.floors[component] < floor:
            raise CatalogError(
                f"the fetched manifest lowers the {component} floor from {floor} to "
                f"{candidate.floors[component]}; the floor only goes up"
            )
