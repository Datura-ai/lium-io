"""A filler's image digests are recorded at create, best effort, for the GPU usage check."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from helpers import FakeRedis
from services.docker_service import DockerService
from services.redis_service import FILLER_IMAGE_DIGESTS_PREFIX


def _ssh(stdout: str | None = None, error: Exception | None = None):
    return SimpleNamespace(
        run=AsyncMock(return_value=SimpleNamespace(stdout=stdout), side_effect=error)
    )


@pytest.mark.asyncio
async def test_records_bare_digests_of_the_filler_image():
    redis = FakeRedis()
    ssh = _ssh('["daturaai/filler@sha256:aaa","mirror/filler@sha256:bbb"]\n')

    await DockerService._record_filler_image_best_effort(
        SimpleNamespace(redis_service=redis), ssh, "filler_x", {}
    )

    assert (
        await redis.get(f"{FILLER_IMAGE_DIGESTS_PREFIX}filler_x") == '["sha256:aaa", "sha256:bbb"]'
    )
    assert "filler_x" in ssh.run.await_args.args[0]


@pytest.mark.parametrize(
    "stdout, error", [("[]", None), ("not json", None), (None, OSError("ssh gone"))]
)
@pytest.mark.asyncio
async def test_records_nothing_and_never_raises_without_a_digest(stdout, error):
    redis = FakeRedis()

    await DockerService._record_filler_image_best_effort(
        SimpleNamespace(redis_service=redis), _ssh(stdout, error), "filler_x", {}
    )

    assert await redis.get(f"{FILLER_IMAGE_DIGESTS_PREFIX}filler_x") is None
