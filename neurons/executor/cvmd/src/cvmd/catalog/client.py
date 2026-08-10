"""Fetching the signed manifest from the backend.

Deliberately built on `urllib.request` rather than a client library. This is one GET with a
timeout against one URL, and cvmd's dependency list is a supply-chain surface on every CVM host
in the fleet — a package added here has to be worth that. The standard library verifies TLS
against the system trust store, which is the only property that matters on the wire; everything
that makes the *content* trustworthy is the signature, checked in `manifest.py` after the bytes
have landed.

Fetch failures are not errors here. A host that cannot reach the backend keeps launching from
the catalog it already holds until that manifest expires, and then refuses — which is the
behaviour a revocation needs. Turning a transient network failure into a refusal instead would
make the platform's availability the fleet's availability.
"""

import logging
import urllib.error
import urllib.request
from datetime import datetime

from cvmd.catalog.artifacts import CatalogError
from cvmd.catalog.manifest import Manifest
from cvmd.catalog.store import CatalogStore

logger = logging.getLogger(__name__)

# A manifest is a few tens of KiB: a handful of composes and their scripts. The cap stops a
# hostile or broken endpoint from making cvmd hold arbitrary memory before a single check runs —
# the same reasoning as the request body cap in the auth middleware.
MAX_MANIFEST_BYTES = 4 * 1024 * 1024

USER_AGENT = "cvmd-catalog/1"


class FetchError(Exception):
    """The manifest could not be retrieved. Never fatal — the cached one stays in force."""


def fetch(url: str, *, timeout: int) -> bytes:
    """GET the manifest, or raise `FetchError`. Verifies nothing about the content."""
    request = urllib.request.Request(  # noqa: S310 - the scheme is checked below
        url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
    )
    if request.type not in ("http", "https"):
        raise FetchError(f"the catalog manifest URL must be http or https, got {url!r}")

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            raw = response.read(MAX_MANIFEST_BYTES + 1)
    except urllib.error.HTTPError as exc:
        raise FetchError(f"{url} answered {exc.code} {exc.reason}") from exc
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise FetchError(f"cannot reach {url}: {exc}") from exc

    if len(raw) > MAX_MANIFEST_BYTES:
        raise FetchError(f"{url} returned more than {MAX_MANIFEST_BYTES} bytes")
    if not raw:
        raise FetchError(f"{url} returned an empty body")
    return raw


def refresh_once(store: CatalogStore, *, now: datetime | None = None) -> Manifest | None:
    """One refresh cycle: adopt a newer seed, then fetch. Returns the manifest that took effect.

    Never raises: every outcome is a log line, because this runs on a timer and the only thing a
    caller could do with an exception is ignore it. The distinction the logs preserve is the one
    that matters — could not *reach* the backend (warning, cache stands) versus reached it and
    the bytes were *not acceptable* (error, cache stands, someone should look).

    The seed is re-read every cycle rather than only at startup, so an operator who stages a
    newer one does not also have to restart the daemon for it to matter. `install` refuses a
    rollback either way, so re-reading an old seed forever is free.
    """
    config = store.config
    adopted = store.install_seed_if_newer()

    if not config.manifest_url:
        return adopted

    try:
        raw = fetch(config.manifest_url, timeout=config.fetch_timeout_seconds)
    except FetchError as exc:
        logger.warning("catalog refresh skipped: %s", exc)
        return adopted

    try:
        return store.install(raw, source=config.manifest_url, now=now)
    except CatalogError as exc:
        logger.error(
            "catalog from %s REFUSED, keeping the one in force: %s", config.manifest_url, exc
        )
        return adopted
