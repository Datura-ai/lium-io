"""Self-check of the executor stack's update state (DAH-3419).

The validator scores a node on the digest of the executor image it runs. The digest a
node *should* run is decided by the runner image the validator signs at
``<COMPUTE_REST_API_URL>/watchtower/digest``; the updater (``watchtower/`` in this
repository) pulls that digest. This service lets the node itself say where it stands:
the digest of the runner container it runs, the signed digest, and whether an update is
pending, so a provider or the portal can read "update pending" from the node instead of
inferring it from a zero score.

Everything here blocks (docker-py, HTTP) and runs in the route's metrics pool with the
pool's timeout. The signed digest is cached for ``EXPECTED_DIGEST_CACHE_SECONDS``, a
failed fetch included, so the endpoint is not hit on every call. A docker daemon that is
down or wedged is reported in ``runner.error``; the route does not fail.
"""

from __future__ import annotations

import logging
import os
import re
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import docker
import requests

from core.config import VALIDATOR_HOTKEYS_SS58, settings
from dependencies.auth import match_validator_hotkey

logger = logging.getLogger(__name__)

RUNNER_CVM_NAME = "executor-runner"
RUNNER_SERVICE_LABEL = "com.docker.compose.service=executor-runner"
EXPECTED_DIGEST_CACHE_SECONDS = 300
ENDPOINT_TIMEOUT_SECONDS = 5.0
# Same window the updater applies to the signed timestamp (watchtower/src/watchtower.py).
TIMESTAMP_MAX_SKEW_SECONDS = 10 * 60
_SHA256_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
# The updater renames the old runner to `<name>-previous-<short id>` while it creates the new one and removes it
# last (watchtower.py `previous_runner_name`). A leftover of an interrupted update keeps the runner label; it is
# never the runner this report describes.
_PREVIOUS_RUNNER_NAME = re.compile(r"^.+-previous-[0-9a-f]+$")


def digest_endpoint_url() -> str | None:
    base_url = settings.COMPUTE_REST_API_URL
    if not base_url:
        return None
    return f"{base_url.rstrip('/')}/watchtower/digest"


def verify_signed_digest(payload: dict[str, Any], now: int | None = None) -> str:
    """Return the digest from a ``/watchtower/digest`` response after checking its signature.

    The validator signs ``"<digest>:<timestamp>"`` with its hotkey; the timestamp must be
    within ``TIMESTAMP_MAX_SKEW_SECONDS`` of now. Raises ``ValueError`` otherwise.
    """
    digest = payload.get("digest")
    timestamp = payload.get("timestamp")
    signature = payload.get("signature")
    if not isinstance(digest, str) or not _SHA256_DIGEST.fullmatch(digest):
        raise ValueError("digest response has no sha256 digest")
    if not isinstance(timestamp, int) or not isinstance(signature, str) or not signature:
        raise ValueError("digest response has no timestamp or signature")
    if now is None:
        now = int(datetime.now(UTC).timestamp())
    if abs(timestamp - now) > TIMESTAMP_MAX_SKEW_SECONDS:
        raise ValueError(f"digest response timestamp out of range (now={now}, ts={timestamp})")
    # DAH-3394: the executor trusts two validator hotkeys during the rotation (current, next);
    # match_validator_hotkey is the one list every validator-signature check reads.
    if match_validator_hotkey(f"{digest}:{timestamp}", signature) is None:
        raise ValueError(
            "digest response signature is not from a configured validator hotkey "
            f"({', '.join(VALIDATOR_HOTKEYS_SS58)})"
        )
    return digest


@dataclass
class ExpectedDigestCache:
    """The signed runner digest, refreshed at most every ``ttl_seconds``."""

    ttl_seconds: float = EXPECTED_DIGEST_CACHE_SECONDS
    _digest: str | None = None
    _error: str | None = None
    _fetched_at: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def get(self, now: float | None = None) -> tuple[str | None, str | None]:
        """``(digest, error)``: the last good digest survives a failed refresh, with the error beside it."""
        if now is None:
            now = time.monotonic()
        with self._lock:
            if now - self._fetched_at < self.ttl_seconds and (self._digest or self._error):
                return self._digest, self._error
            self._fetched_at = now
            url = digest_endpoint_url()
            if url is None:
                self._error = "COMPUTE_REST_API_URL not set"
                return self._digest, self._error
            try:
                response = requests.get(url, timeout=ENDPOINT_TIMEOUT_SECONDS)
                response.raise_for_status()
                self._digest = verify_signed_digest(response.json())
                self._error = None
            except Exception as exc:  # a bad response, a bad signature, or a key error: never the caller's 500
                self._error = f"{type(exc).__name__}: {exc}"
                logger.warning("Signed runner digest not refreshed from %s: %s", url, self._error)
            return self._digest, self._error


class RunnerLookupError(Exception):
    """More than one container carries the runner label."""


def find_runner_container(client: docker.DockerClient):
    """The runner container: the CVM name first, then the compose service label (one match).

    A ``<name>-previous-<id>`` leftover of an interrupted update is skipped (the updater removes it).
    Returns None when there is none. Raises ``RunnerLookupError`` for two or more live matches.
    """
    try:
        return client.containers.get(RUNNER_CVM_NAME)
    except docker.errors.NotFound:
        pass
    matches = [
        c
        for c in client.containers.list(all=True, filters={"label": RUNNER_SERVICE_LABEL})
        if not _PREVIOUS_RUNNER_NAME.match(c.name)
    ]
    if len(matches) > 1:
        raise RunnerLookupError(f"{len(matches)} containers carry {RUNNER_SERVICE_LABEL}")
    return matches[0] if matches else None


def container_image_digest(client: docker.DockerClient, container) -> str | None:
    """``sha256:…`` from the RepoDigests of the image a container runs, or None (local build)."""
    image_id = container.attrs.get("Image")
    if not image_id:
        return None
    repo_digests = client.images.get(image_id).attrs.get("RepoDigests") or []
    for entry in repo_digests:
        digest = entry.split("@")[-1]
        if _SHA256_DIGEST.fullmatch(digest):
            return digest
    return None


def own_container_id() -> str | None:
    """The executor's own container id: docker sets the hostname to it."""
    return os.environ.get("HOSTNAME") or socket.gethostname() or None


def collect_update_status(
    client_factory: Callable[[], docker.DockerClient], cache: ExpectedDigestCache
) -> dict[str, Any]:
    """The self-check: runner digest running vs signed, and the executor's own image digest.

    ``update_pending`` is ``None`` when either digest is unknown, so a missing runner, a
    daemon that does not answer or an unreachable endpoint reads as "unknown", not as
    "current". ``runner.error`` carries every reason, joined with ``" | "``.

    ``client_factory`` is called here, inside the pool, so a daemon that refuses the
    connection is an error string and not a 500. docker-py raises its own
    ``DockerException`` family for API errors and ``requests`` exceptions for a socket
    that does not answer, so both are caught.
    """
    expected_digest, expected_error = cache.get()

    runner_digest: str | None = None
    runner_name: str | None = None
    executor_digest: str | None = None
    errors: list[str] = []
    try:
        client = client_factory()
        runner = find_runner_container(client)
        if runner is None:
            errors.append("runner container not found")
        else:
            runner_name = runner.name
            runner_digest = container_image_digest(client, runner)
        container_id = own_container_id()
        if container_id:
            try:
                executor_digest = container_image_digest(client, client.containers.get(container_id))
            except (docker.errors.DockerException, requests.RequestException) as exc:
                logger.debug("Own image digest not resolved: %s", exc)
    except (docker.errors.DockerException, requests.RequestException, RunnerLookupError) as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    if expected_error:
        errors.append(expected_error)

    update_pending: bool | None = None
    if runner_digest and expected_digest:
        update_pending = runner_digest != expected_digest

    return {
        "runner": {
            "container": runner_name,
            "running_digest": runner_digest,
            "expected_digest": expected_digest,
            "update_pending": update_pending,
            "error": " | ".join(errors) or None,
        },
        "executor": {"running_digest": executor_digest},
    }
