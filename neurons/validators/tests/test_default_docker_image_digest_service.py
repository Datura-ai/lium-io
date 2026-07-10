"""Tests for DefaultDockerImageDigestService (DAH-2380)."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from neurons.validators.src.services.default_docker_image_digest_service import (
    DefaultDockerImageDigestService,
    _bare_digest,
    fetch_registry_digest,
)


def test_bare_digest_strips_repo_prefix():
    assert _bare_digest("daturaai/pytorch@sha256:abc") == "sha256:abc"
    assert _bare_digest("sha256:abc") == "sha256:abc"


@pytest.mark.asyncio
async def test_fetch_registry_digest_returns_bare_digest():
    session = AsyncMock()
    token_response = AsyncMock()
    token_response.raise_for_status = Mock()
    token_response.json = AsyncMock(return_value={"token": "tok"})
    token_cm = AsyncMock()
    token_cm.__aenter__.return_value = token_response

    manifest_response = AsyncMock()
    manifest_response.raise_for_status = Mock()
    manifest_response.headers = {"Docker-Content-Digest": "sha256:deadbeef"}
    manifest_cm = AsyncMock()
    manifest_cm.__aenter__.return_value = manifest_response

    session.get = Mock(return_value=token_cm)
    session.head = Mock(return_value=manifest_cm)

    digest = await fetch_registry_digest(session, "daturaai/pytorch:1.0")

    assert digest == "sha256:deadbeef"
    session.head.assert_called_once()
    assert "manifests/1.0" in session.head.call_args.args[0]


@pytest.mark.asyncio
async def test_refresh_populates_digest_map():
    service = DefaultDockerImageDigestService(
        image_refs=("daturaai/pytorch:test",),
        refresh_seconds=60,
    )

    with patch(
        "neurons.validators.src.services.default_docker_image_digest_service.fetch_registry_digest",
        new=AsyncMock(return_value="sha256:abc"),
    ):
        await service.refresh()

    assert service.get_digest("daturaai/pytorch:test") == "sha256:abc"


@pytest.mark.asyncio
async def test_refresh_drops_ref_when_fetch_fails():
    """A ref that fails to refresh is cleared, not left stale (fail open).

    Regression for the DIGEST_MISMATCH false-positive: a stale digest kept after
    a failed fetch would mismatch a re-pushed tag and zero an honest miner.
    """
    service = DefaultDockerImageDigestService(
        image_refs=("daturaai/pytorch:test",),
        refresh_seconds=60,
    )

    with patch(
        "neurons.validators.src.services.default_docker_image_digest_service.fetch_registry_digest",
        new=AsyncMock(return_value="sha256:abc"),
    ):
        await service.refresh()
    assert service.get_digest("daturaai/pytorch:test") == "sha256:abc"

    with patch(
        "neurons.validators.src.services.default_docker_image_digest_service.fetch_registry_digest",
        new=AsyncMock(return_value=None),
    ):
        await service.refresh()
    assert service.get_digest("daturaai/pytorch:test") is None


@pytest.mark.asyncio
async def test_get_digest_returns_none_for_unknown_image():
    service = DefaultDockerImageDigestService(image_refs=(), refresh_seconds=60)
    assert service.get_digest("daturaai/pytorch:missing") is None
