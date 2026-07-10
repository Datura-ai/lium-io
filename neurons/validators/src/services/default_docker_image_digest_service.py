"""Fetch and cache manifest digests for default cache-template docker images (DAH-2380).

The validator resolves the recommended image from the backend (image + tag only) and
compares the executor's local RepoDigest against a digest map built here from Docker Hub.
Uses the manifest-list media type so the digest matches what `docker pull` records in
RepoDigests on amd64 GPU nodes.
"""

from __future__ import annotations

import asyncio
import logging
from functools import lru_cache

import aiohttp
from services.const import DEFAULT_DOCKER_IMAGE_REFS

from core.config import settings

logger = logging.getLogger(__name__)

_MANIFEST_ACCEPT = (
    "application/vnd.docker.distribution.manifest.list.v2+json,"
    " application/vnd.docker.distribution.manifest.v2+json"
)
_REGISTRY_AUTH_URL = "https://auth.docker.io/token"
_REGISTRY_API = "https://registry-1.docker.io/v2"


def _bare_digest(digest: str | None) -> str | None:
    if not digest:
        return None
    return digest.rpartition("@")[2] or digest


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
            return _bare_digest(manifest_response.headers.get("Docker-Content-Digest"))
    except Exception as exc:
        logger.warning("Failed to fetch registry digest for %s: %s", image_ref, exc)
        return None


class DefaultDockerImageDigestService:
    """In-memory image_ref → bare sha256 digest map, refreshed from Docker Hub."""

    def __init__(
        self,
        image_refs: tuple[str, ...] = DEFAULT_DOCKER_IMAGE_REFS,
        refresh_seconds: int | None = None,
    ) -> None:
        self._image_refs = image_refs
        self._refresh_seconds = (
            refresh_seconds
            if refresh_seconds is not None
            else settings.DEFAULT_DOCKER_IMAGE_DIGEST_REFRESH_SECONDS
        )
        self._digests: dict[str, str] = {}
        self._lock = asyncio.Lock()

    def get_digest(self, image_ref: str) -> str | None:
        """Bare ``sha256:…`` for a previously refreshed ``repo:tag``, or None."""
        return self._digests.get(image_ref)

    async def refresh(self) -> None:
        """Pull current digests for every configured default image ref.

        Replaces the cache wholesale: a ref whose fetch fails this round is
        *dropped* rather than left pointing at a stale digest. A missing digest
        makes the verification check fail open (skip); a stale one would produce
        a false ``DIGEST_MISMATCH`` against a re-pushed tag and zero an honest
        miner's score (DAH-2380).
        """
        updated: dict[str, str] = {}
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for image_ref in self._image_refs:
                digest = await fetch_registry_digest(session, image_ref)
                if digest:
                    updated[image_ref] = digest

        async with self._lock:
            self._digests = updated

        failed = [ref for ref in self._image_refs if ref not in updated]
        if failed:
            logger.warning(
                "Dropped %d default docker image digest(s) after a failed refresh "
                "(check fails open for these): %s",
                len(failed),
                ", ".join(failed),
            )
        if updated:
            logger.info(
                "Refreshed %d default docker image digest(s) from Docker Hub",
                len(updated),
            )

    async def run_refresh_loop(self) -> None:
        """Background loop: refresh on boot, then every ``refresh_seconds``."""
        while True:
            try:
                await self.refresh()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("Unexpected error refreshing default image digests: %s", exc)
            await asyncio.sleep(self._refresh_seconds)


@lru_cache
def get_default_docker_image_digest_service() -> DefaultDockerImageDigestService:
    """Process-wide singleton used by the validator and FastAPI DI."""
    return DefaultDockerImageDigestService()
