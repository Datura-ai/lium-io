"""DAH-2620 — a busy WireGuard port must say so, instead of surfacing docker's bind error."""

from unittest.mock import AsyncMock

import pytest

from services.cluster_fabric import WIREGUARD_LISTEN_PORT
from services.docker_service import DockerService


def _ssh(stdout: str) -> AsyncMock:
    ssh = AsyncMock()
    ssh.run = AsyncMock(return_value=AsyncMock(stdout=stdout))
    return ssh


@pytest.mark.asyncio
async def test_a_free_port_passes_silently() -> None:
    # Arrange
    ssh = _ssh("")

    # Act / Assert — no raise
    await DockerService._assert_cluster_overlay_port_free(ssh, {})


@pytest.mark.asyncio
async def test_a_busy_port_names_the_holder() -> None:
    # Arrange
    ssh = _ssh("pod_abc123\n")

    # Act / Assert
    with pytest.raises(RuntimeError) as failure:
        await DockerService._assert_cluster_overlay_port_free(ssh, {})

    assert str(WIREGUARD_LISTEN_PORT) in str(failure.value)
    assert "pod_abc123" in str(failure.value)


@pytest.mark.asyncio
async def test_the_probe_asks_about_the_overlay_port() -> None:
    # Arrange
    ssh = _ssh("")

    # Act
    await DockerService._assert_cluster_overlay_port_free(ssh, {})

    # Assert — UDP only: the rental's own TCP mappings can hold the same number.
    assert f"publish={WIREGUARD_LISTEN_PORT}/udp" in ssh.run.call_args.args[0]
