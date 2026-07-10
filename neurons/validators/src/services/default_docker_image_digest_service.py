"""Fetch manifest digests for default cache-template docker images (DAH-2380).

The validator resolves the recommended image from the backend (image + tag only) and
compares the executor's local RepoDigest against a digest snapshot built here from Docker
Hub. The set of default images is read from the backend shared config (``SharedConfig``),
whose source of truth is the backend ``DOCKER_IMAGES`` list — no hardcoded copy here.
Uses the manifest-list media type so the digest matches what `docker pull` records in
RepoDigests on amd64 GPU nodes.

``fetch_default_image_digests`` returns a plain ``dict`` snapshot for one job cycle rather
than holding state: the validator fetches it at job-cycle start and threads it into the
pipeline context, so there is no shared mutable cache to keep in sync.
"""

from __future__ import annotations

import logging

import aiohttp

logger = logging.getLogger(__name__)

_MANIFEST_ACCEPT = (
    "application/vnd.docker.distribution.manifest.list.v2+json,"
    " application/vnd.docker.distribution.manifest.v2+json"
)
_REGISTRY_AUTH_URL = "https://auth.docker.io/token"
_REGISTRY_API = "https://registry-1.docker.io/v2"


def _shared_config_image_refs() -> tuple[str, ...]:
    """Default cache-template image refs served by the backend via shared config.

    The single source of truth is the backend ``DOCKER_IMAGES`` list, published on
    ``/v1/shared-config``; ``SharedConfigClient`` falls back to the packaged lium-core
    default when the backend is unreachable, so the returned tuple is always valid.
    """
    from core.config import shared_client

    return tuple(shared_client.config.default_docker_image_refs)


async def fetch_registry_digest(session: aiohttp.ClientSession, image_ref: str) -> str | None:
    """Return the registry manifest digest for ``repository:tag``, or None on any error."""
    repository, _, tag = image_ref.partition(":")
    if not repository or not tag:
        return None
    try:
        async with session.get(
            _REGISTRY_AUTH_URL,
            params={
                "service": "registry.docker.io",
                "scope": f"repository:{repository}:pull",
            },
        ) as token_response:
            token_response.raise_for_status()
            token = (await token_response.json()).get("token")
        if not token:
            return None

        async with session.head(
            f"{_REGISTRY_API}/{repository}/manifests/{tag}",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": _MANIFEST_ACCEPT,
            },
        ) as manifest_response:
            manifest_response.raise_for_status()
            # The Docker-Content-Digest response header is already a bare ``sha256:…``.
            return manifest_response.headers.get("Docker-Content-Digest") or None
    except Exception as exc:
        logger.warning("Failed to fetch registry digest for %s: %s", image_ref, exc)
        return None


async def fetch_default_image_digests() -> dict[str, str]:
    """Snapshot of ``image_ref -> bare sha256 digest`` for one job cycle.

    Fail-open per ref: a ref whose fetch fails this cycle is simply *absent* from the
    snapshot, so the verification check finds no digest to compare against and skips
    (rather than mismatching a re-pushed tag and zeroing an honest miner's score —
    DAH-2380). The list of refs comes from the backend shared config.
    """
    image_refs = _shared_config_image_refs()
    digests: dict[str, str] = {}
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for image_ref in image_refs:
            digest = await fetch_registry_digest(session, image_ref)
            if digest:
                digests[image_ref] = digest

    failed = [ref for ref in image_refs if ref not in digests]
    if failed:
        logger.warning(
            "No default docker image digest for %d ref(s) this cycle "
            "(check fails open for these): %s",
            len(failed),
            ", ".join(failed),
        )
    if digests:
        logger.info(
            "Fetched %d default docker image digest(s) from Docker Hub",
            len(digests),
        )
    return digests
