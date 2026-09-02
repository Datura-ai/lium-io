"""DAH-2620 — a busy WireGuard port must say so, instead of surfacing docker's bind error.

DAH-2842: the port checked is the one the backend allocated for this node, because that is the one
the create binds on the host.
"""

from unittest.mock import AsyncMock

import pytest

from services.docker_service import DockerService

# A port out of the executor's own verified range, which is what the backend allocates.
_OVERLAY_HOST_PORT = 10113


def _ssh(stdout: str) -> AsyncMock:
    ssh = AsyncMock()
    ssh.run = AsyncMock(return_value=AsyncMock(stdout=stdout))
    return ssh


@pytest.mark.asyncio
async def test_a_free_port_passes_silently() -> None:
    # Arrange
    ssh = _ssh("")

    # Act / Assert — no raise
    await DockerService._assert_cluster_overlay_port_free(ssh, _OVERLAY_HOST_PORT, {})


@pytest.mark.asyncio
async def test_a_busy_port_names_the_holder() -> None:
    # Arrange
    ssh = _ssh("pod_abc123\n")

    # Act / Assert
    with pytest.raises(RuntimeError) as failure:
        await DockerService._assert_cluster_overlay_port_free(ssh, _OVERLAY_HOST_PORT, {})

    assert str(_OVERLAY_HOST_PORT) in str(failure.value)
    assert "pod_abc123" in str(failure.value)


@pytest.mark.asyncio
async def test_the_probe_asks_about_the_port_this_node_was_allocated() -> None:
    # Arrange
    ssh = _ssh("")

    # Act
    await DockerService._assert_cluster_overlay_port_free(ssh, _OVERLAY_HOST_PORT, {})

    # Assert
    assert f"publish={_OVERLAY_HOST_PORT}" in ssh.run.call_args.args[0]
