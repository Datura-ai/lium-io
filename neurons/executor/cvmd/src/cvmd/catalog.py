"""The local artifact catalog — which software stacks this host is allowed to launch.

A catalog entry pins a **triple**: the QEMU build, the OS image hash, and the compose hash.
Those three values are the entire input to the CVM's measurements, so an entry is a statement
that "this host may produce exactly these MRTD/RTMR values". Everything else in the entry is
the local path needed to actually produce them.

The caller names a triple; cvmd refuses anything the catalog does not carry. That is the whole
point: a validator can only attest a stack it asked for, so a host quietly serving a different
one has to fail the request rather than the attestation.

**This is DAH-2576's stub.** DAH-2578 replaces the loader with a client for the backend's
signed manifest — fetch, verify the signature, cache, refuse unsigned/tampered/stale. The
*shape* below is what that client will produce, so only `load_catalog` changes. Nothing here
verifies a signature, so the file's integrity rests on its filesystem permissions today.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

SCHEMA_VERSION = 1

# What a hash looks like on both sides of the gate: the catalog's pinned values and the request
# body's. Owned here, where the values are compared against digests cvmd computes itself, and
# imported by the request model so the two cannot drift into accepting different spellings.
HEX64 = r"^[0-9a-f]{64}$"


class CatalogError(Exception):
    """The catalog is missing, unreadable, or malformed. Always refuses a launch."""


class TripleNotFound(Exception):
    """No catalog entry matches the requested triple. The message names which part failed."""


@dataclass(frozen=True)
class Artifact:
    """One approved stack: the triple it produces, and the local inputs that produce it."""

    id: str
    kind: str
    qemu: str
    os_image_hash: str
    compose_hash: str
    os_image_path: Path
    compose_path: Path
    init_script: Path | None = None
    pre_launch_script: Path | None = None
    local_key_provider: bool = True
    enable_logs: bool = False
    enable_sysinfo: bool = False

    @property
    def triple(self) -> tuple[str, str, str]:
        return (self.qemu, self.os_image_hash, self.compose_hash)


def _require_str(raw: dict, key: str, where: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise CatalogError(f"{where}: `{key}` must be a non-empty string")
    return value.strip()


def _optional_path(raw: dict, key: str, where: str) -> Path | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise CatalogError(f"{where}: `{key}` must be a non-empty string when present")
    return Path(value.strip())


def _bool(raw: dict, key: str, default: bool, where: str) -> bool:
    value = raw.get(key, default)
    if not isinstance(value, bool):
        raise CatalogError(f"{where}: `{key}` must be a boolean, got {value!r}")
    return value


def _decode_artifact(raw, index: int) -> Artifact:
    where = f"artifacts[{index}]"
    if not isinstance(raw, dict):
        raise CatalogError(f"{where} must be an object")

    artifact = Artifact(
        id=_require_str(raw, "id", where),
        kind=_require_str(raw, "kind", where),
        qemu=_require_str(raw, "qemu", where),
        os_image_hash=_require_str(raw, "os_image_hash", where),
        compose_hash=_require_str(raw, "compose_hash", where),
        os_image_path=Path(_require_str(raw, "os_image_path", where)),
        compose_path=Path(_require_str(raw, "compose_path", where)),
        init_script=_optional_path(raw, "init_script", where),
        pre_launch_script=_optional_path(raw, "pre_launch_script", where),
        local_key_provider=_bool(raw, "local_key_provider", True, where),
        enable_logs=_bool(raw, "enable_logs", False, where),
        enable_sysinfo=_bool(raw, "enable_sysinfo", False, where),
    )
    # Hashes are compared against values cvmd computes itself, which are lowercase hex. A
    # catalog written with uppercase or a `sha256:` prefix would never match and the failure
    # would read as "this stack is not approved" — refuse it here, where the cause is legible.
    for field in ("os_image_hash", "compose_hash"):
        value = getattr(artifact, field)
        if not re.fullmatch(HEX64, value):
            raise CatalogError(f"{where}: `{field}` must be 64 lowercase hex digits, got {value!r}")
    return artifact


def load_catalog(path: Path) -> list[Artifact]:
    """Read and fully validate the catalog. Any problem refuses every launch, not some."""
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise CatalogError(f"no catalog at {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise CatalogError(f"cannot read catalog at {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise CatalogError(f"{path}: catalog must be a JSON object")
    if raw.get("version") != SCHEMA_VERSION:
        raise CatalogError(
            f"{path}: catalog version {raw.get('version')!r} is not the supported "
            f"{SCHEMA_VERSION} — refusing rather than guessing at the shape"
        )

    entries = raw.get("artifacts")
    if not isinstance(entries, list):
        raise CatalogError(f"{path}: `artifacts` must be a list")

    artifacts = [_decode_artifact(entry, index) for index, entry in enumerate(entries)]

    seen: dict[tuple[str, str, str, str], str] = {}
    for artifact in artifacts:
        key = (artifact.kind, *artifact.triple)
        if key in seen:
            raise CatalogError(
                f"{path}: {artifact.id} and {seen[key]} pin the same kind and triple, so a "
                f"request could resolve to either"
            )
        seen[key] = artifact.id
    return artifacts


def resolve(
    artifacts: list[Artifact], *, kind: str, qemu: str, os_image_hash: str, compose_hash: str
) -> Artifact:
    """Return the single entry matching the triple, or raise naming what did not match.

    The message narrows one component at a time so an operator learns *which* part of their
    triple is unapproved — "compose hash is not in the catalog" is actionable, "not found" is
    not.
    """
    requested = {"qemu": qemu, "os_image_hash": os_image_hash, "compose_hash": compose_hash}

    candidates = [a for a in artifacts if a.kind == kind]
    if not candidates:
        kinds = sorted({a.kind for a in artifacts}) or ["(none)"]
        raise TripleNotFound(
            f"the catalog carries no {kind!r} artifact (it has: {', '.join(kinds)})"
        )

    for field, value in requested.items():
        narrowed = [a for a in candidates if getattr(a, field) == value]
        if not narrowed:
            approved = sorted({getattr(a, field) for a in candidates})
            raise TripleNotFound(
                f"{field}={value!r} is not approved for kind={kind!r}; the catalog allows: "
                f"{', '.join(approved)}"
            )
        candidates = narrowed

    # load_catalog already rejected duplicate (kind, triple) pairs, so exactly one survives.
    return candidates[0]
