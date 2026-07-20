"""Digest-pinned cache template refresh (DAH-2461).

The backend serves the expected Docker Hub digest for the default cache template;
``_ensure_template`` must treat it as the source of truth (pull by digest + retag on
mismatch) and only fall back to the daemon's remote-digest compare when the backend
sends no digest — the daemon path resolves through provider registry mirrors, which
can serve stale manifests for a tag.
"""

import asyncio
from unittest.mock import MagicMock

import docker

# conftest replaces the docker module with a MagicMock; except-clauses in the
# service need a real exception class to catch.
docker.errors.ImageNotFound = type("ImageNotFound", (Exception,), {})

from services import cache_template_service  # noqa: E402

EXPECTED_DIGEST = "sha256:70bd5fa697877594b753a146e207ca4de66d9b875d606ae09e6ee7bac8f4f423"
STALE_DIGEST = "sha256:2d19c94ce8a37c6fa364f8a6211d8b6dc1a44ece574c4c22ab5579925ce7a4c8"
REPO = "daturaai/pytorch"
TAG = "2.12.0-py3.12-cuda13.0.2-devel-ubuntu24.04-dind"


def _make_client(local_digests: list[str] | None = None, image_absent: bool = False) -> MagicMock:
    client = MagicMock()
    if image_absent:
        client.images.get.side_effect = docker.errors.ImageNotFound("absent")
    else:
        image = MagicMock()
        image.attrs = {"RepoDigests": [f"{REPO}@{digest}" for digest in (local_digests or [])]}
        client.images.get.return_value = image
    client.images.list.return_value = []
    return client


def _template(digest: str | None = None) -> dict:
    data = {
        "docker_image": REPO,
        "docker_image_tag": TAG,
        "docker_image_size": 0,
    }
    if digest is not None:
        data["docker_image_digest"] = digest
    return data


def test_backend_digest_match_skips_pull():
    # Arrange
    client = _make_client(local_digests=[EXPECTED_DIGEST])

    # Act
    asyncio.run(cache_template_service._ensure_template(client, _template(EXPECTED_DIGEST)))

    # Assert
    client.images.pull.assert_not_called()
    client.images.get_registry_data.assert_not_called()


def test_backend_digest_mismatch_pulls_pinned_and_retags():
    # Arrange: local tag poisoned by a stale mirror, backend knows the fresh digest
    client = _make_client(local_digests=[STALE_DIGEST])
    pulled_image = MagicMock()
    client.images.pull.return_value = pulled_image

    # Act
    asyncio.run(cache_template_service._ensure_template(client, _template(EXPECTED_DIGEST)))

    # Assert: content-addressed pull, then the tag is re-pointed at the pinned build
    client.images.pull.assert_called_once_with(f"{REPO}@{EXPECTED_DIGEST}")
    pulled_image.tag.assert_called_once_with(REPO, TAG)
    client.images.get_registry_data.assert_not_called()


def test_backend_digest_image_absent_pulls_pinned():
    # Arrange: nothing cached locally yet
    client = _make_client(image_absent=True)
    pulled_image = MagicMock()
    client.images.pull.return_value = pulled_image

    # Act
    asyncio.run(cache_template_service._ensure_template(client, _template(EXPECTED_DIGEST)))

    # Assert: first pull is digest-pinned too (bypasses the mirror from the start)
    client.images.pull.assert_called_once_with(f"{REPO}@{EXPECTED_DIGEST}")
    pulled_image.tag.assert_called_once_with(REPO, TAG)


def test_no_backend_digest_matching_remote_skips_pull():
    # Arrange: legacy path — daemon reports the same digest as cached
    client = _make_client(local_digests=[STALE_DIGEST])
    registry_data = MagicMock()
    registry_data.id = STALE_DIGEST
    client.images.get_registry_data.return_value = registry_data

    # Act
    asyncio.run(cache_template_service._ensure_template(client, _template()))

    # Assert
    client.images.pull.assert_not_called()


def test_no_backend_digest_changed_remote_pulls_by_tag():
    # Arrange: legacy path — daemon reports a new digest
    client = _make_client(local_digests=[STALE_DIGEST])
    registry_data = MagicMock()
    registry_data.id = EXPECTED_DIGEST
    client.images.get_registry_data.return_value = registry_data

    # Act
    asyncio.run(cache_template_service._ensure_template(client, _template()))

    # Assert: legacy pull by repo + tag, no digest pinning involved
    client.images.pull.assert_called_once_with(REPO, TAG)


def test_no_backend_digest_unreadable_remote_keeps_local():
    # Arrange: legacy path — daemon cannot resolve the remote digest
    client = _make_client(local_digests=[STALE_DIGEST])
    client.images.get_registry_data.side_effect = RuntimeError("registry unreachable")

    # Act
    asyncio.run(cache_template_service._ensure_template(client, _template()))

    # Assert
    client.images.pull.assert_not_called()
