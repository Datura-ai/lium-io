"""Tests for the default docker image digest snapshot (DAH-2380)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from neurons.validators.src.services.default_docker_image_digest_service import (
    _shared_config_image_refs,
    fetch_default_image_digests,
    fetch_docker_hub_digest,
    fetch_executor_image_digest,
    fetch_registry_digest,
)

_MODULE = "neurons.validators.src.services.default_docker_image_digest_service"


def _shared_client_with(images) -> SimpleNamespace:
    return SimpleNamespace(config=SimpleNamespace(default_docker_images=images))


def test_shared_config_image_refs_derives_repo_tag_from_structured_entries():
    """Refs are derived locally from the structured default_docker_images field.

    The backend serves its DOCKER_IMAGES entries verbatim (full metadata); only
    image/tag participate in the ref, extra keys are ignored.
    """
    images = (
        {"image": "daturaai/pytorch", "tag": "cuda12.8-dind", "cuda": 12.8, "size": 123},
        {"image": "daturaai/pytorch", "tag": "cuda13.0-dind", "cuda": 13.0, "size": 456},
    )
    with patch("core.config.shared_client", _shared_client_with(images)):
        refs = _shared_config_image_refs()

    assert refs == (
        "daturaai/pytorch:cuda12.8-dind",
        "daturaai/pytorch:cuda13.0-dind",
    )


def test_shared_config_image_refs_skips_malformed_entries():
    """An entry missing image or tag is dropped (fail open), not a broken ref."""
    images = (
        {"image": "daturaai/pytorch", "tag": "cuda12.8-dind"},
        {"image": "daturaai/pytorch"},  # no tag
        {"tag": "orphan-tag"},  # no image
        {},
    )
    with patch("core.config.shared_client", _shared_client_with(images)):
        refs = _shared_config_image_refs()

    assert refs == ("daturaai/pytorch:cuda12.8-dind",)


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


@pytest.mark.asyncio
async def test_fetch_executor_image_digest_uses_executor_image_ref():
    with patch(
        f"{_MODULE}.fetch_registry_digest",
        new=AsyncMock(return_value=f"sha256:{'a' * 64}"),
    ) as fetch_registry:
        digest = await fetch_executor_image_digest()

    assert digest == f"sha256:{'a' * 64}"
    fetch_registry.assert_awaited_once()
    assert fetch_registry.await_args.args[1] == "daturaai/compute-subnet-executor:latest"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("image", "looked_up"),
    [
        ("daturaai/pytorch:2.11.0-dind-lium1", "daturaai/pytorch:2.11.0-dind-lium1"),
        ("docker.io/daturaai/pytorch:prod", "daturaai/pytorch:prod"),
        ("ubuntu:24.04", "library/ubuntu:24.04"),
        ("ghcr.io/org/app:prod", None),
        ("ghcr.io/app:prod", None),
        ("localhost:5000/app:prod", None),
        ("localhost/app:prod", None),
        ("daturaai/pytorch@sha256:abc", None),
        ("daturaai/pytorch", None),
        ("daturaai/../v2/x:tag", None),
        ("daturaai/pytorch:tag?x=1", None),
    ],
)
async def test_fetch_docker_hub_digest_asks_docker_hub_only_for_a_plain_hub_tag(image, looked_up):
    with patch(f"{_MODULE}.fetch_registry_digest", AsyncMock(return_value="sha256:d")) as fetch:
        digest = await fetch_docker_hub_digest(image)

    if looked_up is None:
        assert digest is None
        fetch.assert_not_awaited()
    else:
        assert digest == "sha256:d"
        assert fetch.await_args.args[1] == looked_up


@pytest.mark.asyncio
async def test_fetch_docker_hub_digest_gives_up_at_its_bound_when_docker_hub_hangs(monkeypatch):
    monkeypatch.setattr(f"{_MODULE}._RENT_PATH_DIGEST_TIMEOUT_SECONDS", 0.05)
    async def docker_hub_hangs(*args: object) -> str | None:
        await asyncio.Event().wait()

    with patch(f"{_MODULE}.fetch_registry_digest", docker_hub_hangs):
        digest = await asyncio.wait_for(fetch_docker_hub_digest("daturaai/pytorch:prod"), timeout=2)

    assert digest is None
