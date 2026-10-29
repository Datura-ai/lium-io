"""A filler's image digests are recorded at create, best effort, for the GPU usage check."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from helpers import FakeRedis
from services.docker_service import DockerService
from services.redis_service import FILLER_IMAGE_DIGESTS_PREFIX


def _docker(repo_digests=None, error: Exception | None = None):
    return SimpleNamespace(
        container_image_id=AsyncMock(return_value="sha256:imageid"),
        local_image_repo_digests=AsyncMock(return_value=repo_digests, side_effect=error),
    )


@pytest.mark.asyncio
async def test_records_bare_digests_of_the_filler_image():
    redis = FakeRedis()
    docker = _docker(("daturaai/filler@sha256:aaa", "mirror/filler@sha256:bbb"))

    await DockerService._record_filler_image_best_effort(
        SimpleNamespace(redis_service=redis), docker, "container-id", "filler_x", {}
    )

    assert (
        await redis.get(f"{FILLER_IMAGE_DIGESTS_PREFIX}filler_x") == '["sha256:aaa", "sha256:bbb"]'
    )
    # read from the created container's own image, not the tag, which can move during the bootstrap
    docker.container_image_id.assert_awaited_once_with(container="container-id")
    docker.local_image_repo_digests.assert_awaited_once_with(image="sha256:imageid")


@pytest.mark.parametrize(
    "repo_digests, error", [((), None), (None, None), (None, OSError("ssh gone"))]
)
@pytest.mark.asyncio
async def test_records_nothing_and_never_raises_without_a_digest(repo_digests, error):
    redis = FakeRedis()

    await DockerService._record_filler_image_best_effort(
        SimpleNamespace(redis_service=redis), _docker(repo_digests, error), "img", "filler_x", {}
    )

    assert await redis.get(f"{FILLER_IMAGE_DIGESTS_PREFIX}filler_x") is None
