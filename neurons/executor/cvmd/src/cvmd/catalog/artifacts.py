"""What an approved stack *is*, and how a request is matched against one.

A catalog entry pins a **triple**: the QEMU build, the OS image hash, and the compose hash.
Those three values are the entire input to the CVM's measurements, so an entry is a statement
that "this host may produce exactly these MRTD/RTMR values". Everything else in the entry is
the local path needed to actually produce them.

The caller names a triple; cvmd refuses anything the catalog does not carry. That is the whole
point: a validator can only attest a stack it asked for, so a host quietly serving a different
one has to fail the request rather than the attestation.

This module is deliberately pure — it holds no I/O and no policy about *where* an approved set
comes from. `manifest.py` decides what is approved and proves it was the platform that said so;
`store.py` turns that into the local paths below.
"""

import re
from dataclasses import dataclass
from pathlib import Path

# What a hash looks like on both sides of the gate: the catalog's pinned values and the request
# body's. Owned here, where the values are compared against digests cvmd computes itself, and
# imported by the request model so the two cannot drift into accepting different spellings.
HEX64 = r"^[0-9a-f]{64}$"

_HEX64 = re.compile(HEX64)


class CatalogError(Exception):
    """The catalog is missing, unusable, or not trustworthy. Always refuses a launch."""


class TripleNotFound(Exception):
    """No catalog entry matches the requested triple. The message names which part failed."""


def assert_hex64(value: str, *, field: str, where: str) -> str:
    """Reject any spelling of a hash that cvmd's own digests could never equal.

    cvmd compares these against values it computes itself, which are lowercase hex with no
    prefix. A catalog written with uppercase or a `sha256:` prefix would never match, and the
    failure would read as "this stack is not approved" — which sends an operator looking at the
    wrong thing entirely.
    """
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise CatalogError(f"{where}: `{field}` must be 64 lowercase hex digits, got {value!r}")
    return value


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


def assert_unambiguous(artifacts: list[Artifact], *, where: str) -> None:
    """Refuse a set in which one request could resolve to either of two entries."""
    seen: dict[tuple[str, str, str, str], str] = {}
    for artifact in artifacts:
        key = (artifact.kind, *artifact.triple)
        if key in seen:
            raise CatalogError(
                f"{where}: {artifact.id} and {seen[key]} pin the same kind and triple, so a "
                f"request could resolve to either"
            )
        seen[key] = artifact.id


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

    # `assert_unambiguous` already rejected duplicate (kind, triple) pairs, so exactly one
    # survives.
    return candidates[0]
