import pytest
from unittest.mock import AsyncMock, MagicMock

from services.executor_connectivity.cleanup_service import ContainerCleanupService


@pytest.mark.asyncio
async def test_cleanup_removes_orphaned_containers():
    """Containers not in active list should be removed."""
    # Arrange
    ssh = AsyncMock()
    ssh.run = AsyncMock(return_value=MagicMock(stdout="pod_abc123\npod_def456\n"))

    service = ContainerCleanupService()

    # Act
    await service.cleanup(ssh, ["pod_abc123"], "pod_")

    # Assert - pod_def456 should be removed (not in active list)
    assert ssh.run.call_count == 3  # ps_filter, docker rm, volume prune
    rm_call = ssh.run.call_args_list[1]
    assert "docker rm" in rm_call.args[0]
    assert "pod_def456" in rm_call.args[0]
    assert "pod_abc123" not in rm_call.args[0]


@pytest.mark.asyncio
async def test_cleanup_skips_when_all_containers_active():
    """No removal when all containers are in active list."""
    # Arrange
    ssh = AsyncMock()
    ssh.run = AsyncMock(return_value=MagicMock(stdout="pod_abc123\npod_def456\n"))

    service = ContainerCleanupService()

    # Act
    await service.cleanup(ssh, ["pod_abc123", "pod_def456"], "pod_")

    # Assert - only ps_filter called, no rm
    assert ssh.run.call_count == 1


@pytest.mark.asyncio
async def test_cleanup_handles_empty_stdout():
    """Empty docker ps output should not cause errors."""
    # Arrange
    ssh = AsyncMock()
    ssh.run = AsyncMock(return_value=MagicMock(stdout=""))

    service = ContainerCleanupService()

    # Act
    await service.cleanup(ssh, [], "pod_")

    # Assert - only ps_filter called
    assert ssh.run.call_count == 1


@pytest.mark.asyncio
async def test_cleanup_handles_exception_gracefully():
    """Exceptions should be logged but not raised."""
    # Arrange
    ssh = AsyncMock()
    ssh.run = AsyncMock(side_effect=Exception("SSH connection failed"))

    service = ContainerCleanupService()

    # Act - should not raise
    await service.cleanup(ssh, [], "pod_")

    # Assert - no exception raised
    assert ssh.run.call_count == 1
