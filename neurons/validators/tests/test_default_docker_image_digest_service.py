"""Tests for the default docker image digest snapshot (DAH-2380)."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from neurons.validators.src.services.default_docker_image_digest_service import (
    fetch_default_image_digests,
    fetch_registry_digest,
)

_MODULE = "neurons.validators.src.services.default_docker_image_digest_service"


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
async def test_fetch_default_image_digests_builds_snapshot():
    with (
        patch(f"{_MODULE}._shared_config_image_refs", return_value=("daturaai/pytorch:test",)),
        patch(f"{_MODULE}.fetch_registry_digest", new=AsyncMock(return_value="sha256:abc")),
    ):
        digests = await fetch_default_image_digests()

    assert digests == {"daturaai/pytorch:test": "sha256:abc"}


@pytest.mark.asyncio
async def test_fetch_default_image_digests_drops_ref_when_fetch_fails():
    """A ref that fails to fetch is absent from the snapshot, not stale (fail open).

    Regression for the DIGEST_MISMATCH false-positive: a stale digest kept after a
    failed fetch would mismatch a re-pushed tag and zero an honest miner's score.
    """
    with (
        patch(f"{_MODULE}._shared_config_image_refs", return_value=("daturaai/pytorch:test",)),
        patch(f"{_MODULE}.fetch_registry_digest", new=AsyncMock(return_value=None)),
    ):
        digests = await fetch_default_image_digests()

    assert digests == {}
    # An absent ref makes the verification check find no digest to compare (skip).
    assert digests.get("daturaai/pytorch:test") is None


@pytest.mark.asyncio
async def test_fetch_default_image_digests_reads_refs_from_shared_config():
    """The ref list is the backend single-source-of-truth via shared config.

    No hardcoded copy lives in the validator: whatever ``_shared_config_image_refs``
    returns is exactly what gets fetched and keyed in the snapshot.
    """
    with (
        patch(f"{_MODULE}._shared_config_image_refs", return_value=("daturaai/pytorch:shared",)),
        patch(f"{_MODULE}.fetch_registry_digest", new=AsyncMock(return_value="sha256:shared")),
    ):
        digests = await fetch_default_image_digests()

    assert digests == {"daturaai/pytorch:shared": "sha256:shared"}
