from unittest.mock import AsyncMock, Mock, patch

import pytest

from services.docker_service import DockerService, is_host_fault_create_failure
from services.task.checks.rental_probe import STEP_CONTAINER_START

EXECUTOR_UUID = "0b7c1e1e-8f43-4c3e-9f0e-2f4a1c9d5e11"


def _service() -> tuple[DockerService, Mock]:
    redis_service = Mock()
    redis_service.set = AsyncMock()
    redis_service.delete = AsyncMock()
    service = DockerService(ssh_service=Mock(), redis_service=redis_service, attestation_service=Mock())
    return service, redis_service


@pytest.mark.parametrize(
    ("current_step", "expected"),
    [
        ("volume_creation", True),
        ("gpu_flags", True),
        # runc echoes the renter's command, so a docker run error can carry any text the renter chose
        ("docker_run", False),
        ("docker_pull", False),
        ("ssh_connect", False),
    ],
)
def test_only_failures_on_the_node_itself_are_host_faults(current_step, expected):
    assert is_host_fault_create_failure(current_step) is expected


@pytest.mark.asyncio
async def test_a_host_fault_stands_as_a_failed_probe_and_drops_the_interval_stamp():
    service, redis_service = _service()

    with patch("services.docker_service.settings.RENTAL_PROBE_ENABLED", True):
        await service.hold_node_until_probe_passes(EXECUTOR_UUID, "volume_creation", {})

    redis_service.set.assert_awaited_once_with(
        f"rental_probe_failed:{EXECUTOR_UUID}", f"{STEP_CONTAINER_START}:volume_creation"
    )
    redis_service.delete.assert_awaited_once_with(f"rental_probe_ok:{EXECUTOR_UUID}")


@pytest.mark.asyncio
async def test_nothing_is_stamped_while_the_rental_probe_is_off():
    service, redis_service = _service()

    with patch("services.docker_service.settings.RENTAL_PROBE_ENABLED", False):
        await service.hold_node_until_probe_passes(EXECUTOR_UUID, "volume_creation", {})

    redis_service.set.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_renter_side_failure_is_not_stamped():
    service, redis_service = _service()

    with patch("services.docker_service.settings.RENTAL_PROBE_ENABLED", True):
        await service.hold_node_until_probe_passes(EXECUTOR_UUID, "docker_pull", {})

    redis_service.set.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_redis_error_while_stamping_does_not_raise():
    service, redis_service = _service()
    redis_service.set.side_effect = ConnectionError("redis down")

    with patch("services.docker_service.settings.RENTAL_PROBE_ENABLED", True):
        await service.hold_node_until_probe_passes(EXECUTOR_UUID, "gpu_flags", {})

    redis_service.delete.assert_not_awaited()
