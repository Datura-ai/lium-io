"""What a CVM is allowed to measure as, taken from the platform instead of from a constant.

DAH-2581. Until now the answer lived in `services/const.py:TDX_WHITELIST` — a dict edited by
hand and shipped with the validator, which meant a revoked image stayed acceptable everywhere
nobody remembered to redeploy, and a new release could not be approved without a release.

The platform's signed manifest (DAH-2578) already carries exactly this set, and every CVM host
already fetches and verifies it. This module makes the validator a third reader of the same
document, so "what a host may launch" and "what this validator will accept" stop being two
lists that drift.

    lium-io-backend: services/cvm_catalog.py   builds and signs it
    lium-io:  cvmd/src/cvmd/catalog/manifest.py  the reference implementation of every check
    here                                        the verifying side of the same protocol

The checks below are the same ones cvmd makes, and for the same reasons — but they are written
out rather than imported, because cvmd is a separate package on a separate Python and a
separate bittensor pin. Where the two must agree exactly is the signed bytes, and that is one
line: the domain separator and the payload text.

**Two things are deliberately NOT done here.**

There is no rollback ratchet across process restarts. cvmd keeps a cached manifest on disk and
refuses a serial lower than the one it holds; this validator holds its catalog in memory, so a
restart legitimately starts from nothing and a disk cache would be a second thing to keep
honest. What survives is a within-process ratchet, which is what defeats a backend that starts
replaying an old manifest at a running validator.

And a fetch failure never empties the whitelist. The manifest in force stays in force until a
newer one verifies. A validator that failed every CVM in the fleet because the backend was
briefly unreachable would be a far worse outage than one accepting a slightly stale catalog —
and the catalog has its own expiry for the case where "slightly" stops being true.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

import aiohttp
import bittensor

from core.config import settings
from core.utils import _m, get_extra_info

logger = logging.getLogger(__name__)

SCHEMA = "lium-cvm-catalog/1"
DOMAIN_SEPARATOR = b"lium-cvm-catalog-v1\x00"
PAYLOAD_VERSION = 1


class CvmCatalogError(Exception):
    """The manifest is missing, malformed, or not the platform's. Never accepted."""


@dataclass(frozen=True)
class CvmCatalog:
    """One verified manifest: the approved set, and when it stops being usable."""

    serial: int
    expires_at: datetime
    os_image_hashes: frozenset[str]
    compose_hashes: frozenset[str]
    # Keyed by kind, so a renter compose cannot satisfy a validation check or the reverse.
    compose_hashes_by_kind: dict[str, frozenset[str]] = field(default_factory=dict)

    def is_expired(self, now: datetime | None = None) -> bool:
        return (now or datetime.now(UTC)) >= self.expires_at

    def approves(self, *, os_image_hash: str, compose_hash: str, kind: str | None = None) -> bool:
        if os_image_hash not in self.os_image_hashes:
            return False
        if kind is None:
            return compose_hash in self.compose_hashes
        return compose_hash in self.compose_hashes_by_kind.get(kind, frozenset())


def parse_manifest(raw: bytes, *, signer: str, now: datetime | None = None) -> CvmCatalog:
    """Verify a manifest completely and return what it approves. Raises on any refusal.

    The signature is checked against the ss58 this validator was **configured** with, never
    against the `signer` field inside the document. A document that names its own signer proves
    only that whoever wrote it also chose who to blame.
    """
    now = now or datetime.now(UTC)
    try:
        envelope = json.loads(raw)
    except ValueError as exc:
        raise CvmCatalogError(f"the manifest is not JSON: {exc}") from exc
    if not isinstance(envelope, dict):
        raise CvmCatalogError("the manifest is not a JSON object")

    if envelope.get("schema") != SCHEMA:
        raise CvmCatalogError(
            f"the manifest declares schema {envelope.get('schema')!r}, not {SCHEMA!r}"
        )

    payload = envelope.get("payload")
    signature = envelope.get("signature")
    if not isinstance(payload, str) or not isinstance(signature, str):
        raise CvmCatalogError("the manifest is missing its payload or its signature")

    try:
        signature_bytes = bytes.fromhex(signature[2:] if signature.startswith("0x") else signature)
    except ValueError as exc:
        raise CvmCatalogError(f"the manifest's signature is not hex: {exc}") from exc

    # Verified over the payload TEXT, exactly as it arrived. Re-serializing the decoded object
    # first would make verification depend on key order and float spelling agreeing across two
    # languages — and every one of those is a silent bypass the day it drifts.
    try:
        keypair = bittensor.Keypair(ss58_address=signer)
        verified = keypair.verify(DOMAIN_SEPARATOR + payload.encode(), signature_bytes)
    except Exception as exc:  # noqa: BLE001 - a malformed ss58 or signature raises many types
        raise CvmCatalogError(f"the manifest's signature could not be checked: {exc}") from exc
    if not verified:
        raise CvmCatalogError(f"the manifest is not signed by the configured signer {signer}")

    return _decode_payload(payload, now=now)


def _decode_payload(payload: str, *, now: datetime) -> CvmCatalog:
    try:
        document = json.loads(payload)
    except ValueError as exc:
        raise CvmCatalogError(f"the manifest payload is not JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise CvmCatalogError("the manifest payload is not a JSON object")

    version = document.get("version")
    if version != PAYLOAD_VERSION:
        raise CvmCatalogError(
            f"manifest payload version {version} is not the supported {PAYLOAD_VERSION} — "
            f"refusing rather than guessing at the shape"
        )

    try:
        serial = int(document["serial"])
        expires_at = datetime.fromisoformat(document["expires_at"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CvmCatalogError(f"the manifest payload is missing a required field: {exc}") from exc
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=UTC)

    if now >= expires_at:
        raise CvmCatalogError(
            f"the manifest expired at {expires_at.isoformat()}; an expired catalog is how a "
            f"revocation would be defeated by simply never delivering the next one"
        )

    artifacts = document.get("artifacts")
    if not isinstance(artifacts, list):
        raise CvmCatalogError("the manifest payload carries no `artifacts` list")

    os_images: set[str] = set()
    composes: set[str] = set()
    by_kind: dict[str, set[str]] = {}
    for entry in artifacts:
        if not isinstance(entry, dict):
            raise CvmCatalogError("a manifest artifact is not an object")
        os_image_hash = entry.get("os_image_hash")
        compose_hash = entry.get("compose_hash")
        kind = entry.get("kind")
        if not isinstance(os_image_hash, str) or not isinstance(compose_hash, str):
            raise CvmCatalogError(f"artifact {entry.get('id')!r} has no usable hashes")
        os_images.add(os_image_hash)
        composes.add(compose_hash)
        if isinstance(kind, str):
            by_kind.setdefault(kind, set()).add(compose_hash)

    return CvmCatalog(
        serial=serial,
        expires_at=expires_at,
        os_image_hashes=frozenset(os_images),
        compose_hashes=frozenset(composes),
        compose_hashes_by_kind={k: frozenset(v) for k, v in by_kind.items()},
    )


class CvmWhitelistSource:
    """Holds the manifest in force and refreshes it on a timer.

    One instance per validator process. `current()` never raises and never blocks — a check
    that has to decide right now gets whatever verified last, and `None` means this validator
    has never held a usable manifest, which callers treat as "fall back to the static list"
    rather than as "approve nothing".
    """

    def __init__(self, *, session_factory=None) -> None:
        self._catalog: CvmCatalog | None = None
        self._fetched_at: float = 0.0
        self._session_factory = session_factory or aiohttp.ClientSession

    @property
    def enabled(self) -> bool:
        return bool(
            settings.ENABLE_DYNAMIC_CVM_WHITELIST
            and settings.CVM_CATALOG_MANIFEST_URL
            and settings.CVM_CATALOG_SIGNER_SS58
        )

    def current(self) -> CvmCatalog | None:
        """The manifest in force, or None. Expiry is checked here, not only at fetch time."""
        catalog = self._catalog
        if catalog is None:
            return None
        if catalog.is_expired():
            logger.warning(_m(
                "The CVM catalog in force has expired; falling back until a newer one verifies",
                extra=get_extra_info({"serial": catalog.serial}),
            ))
            return None
        return catalog

    def is_due(self) -> bool:
        return (time.time() - self._fetched_at) >= settings.CVM_CATALOG_REFRESH_SECONDS

    async def refresh(self, *, force: bool = False) -> CvmCatalog | None:
        """Fetch and verify once. Returns the manifest in force afterwards.

        A failure of any kind leaves the previous manifest untouched and is logged, never
        raised: the caller is a validation cycle, and a briefly unreachable backend must not
        turn into a fleet-wide attestation failure.
        """
        if not self.enabled or (not force and not self.is_due()):
            return self._catalog

        # Recorded before the attempt so a backend that is down does not get re-polled on every
        # single executor of every single cycle.
        self._fetched_at = time.time()
        try:
            raw = await self._fetch()
            candidate = parse_manifest(raw, signer=settings.CVM_CATALOG_SIGNER_SS58)
        except Exception as exc:  # noqa: BLE001 - a refusal and a network error are both just "no newer catalog"
            logger.warning(_m(
                "Could not refresh the CVM catalog; the manifest in force is unchanged",
                extra=get_extra_info({
                    "error": str(exc),
                    "held_serial": self._catalog.serial if self._catalog else None,
                }),
            ))
            return self._catalog

        held = self._catalog
        if held is not None and candidate.serial < held.serial:
            # Within-process rollback ratchet. It is what stops a backend that starts serving an
            # older manifest from re-approving something this validator has already seen retired.
            logger.warning(_m(
                "Refusing a CVM catalog older than the one in force",
                extra=get_extra_info({"held_serial": held.serial, "offered": candidate.serial}),
            ))
            return held

        if held is None or candidate.serial != held.serial:
            logger.info(_m(
                "CVM catalog in force",
                extra=get_extra_info({
                    "serial": candidate.serial,
                    "os_images": len(candidate.os_image_hashes),
                    "composes": len(candidate.compose_hashes),
                    "expires_at": candidate.expires_at.isoformat(),
                }),
            ))
        self._catalog = candidate
        return candidate

    async def _fetch(self) -> bytes:
        timeout = aiohttp.ClientTimeout(total=settings.CVM_CATALOG_FETCH_TIMEOUT_SECONDS)
        async with self._session_factory(timeout=timeout) as session:
            async with session.get(settings.CVM_CATALOG_MANIFEST_URL) as response:
                if response.status >= 400:
                    raise CvmCatalogError(
                        f"the catalog endpoint answered {response.status}"
                    )
                return await response.read()
