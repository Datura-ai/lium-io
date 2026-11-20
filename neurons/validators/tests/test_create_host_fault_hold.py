from unittest.mock import AsyncMock, Mock

import docker.errors
import pytest
import requests
import urllib3

from services.rental_docker_sdk import RentalDockerOperationError, RentalDockerTransportDropped

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

    await service.hold_node_until_probe_passes(EXECUTOR_UUID, "volume_creation", {})

    redis_service.set.assert_awaited_once_with(
        f"rental_probe_failed:{EXECUTOR_UUID}",
        f"{STEP_CONTAINER_START}:volume_creation",
    )
    redis_service.delete.assert_awaited_once_with(f"rental_probe_ok:{EXECUTOR_UUID}")


@pytest.mark.asyncio
async def test_a_renter_side_failure_is_not_stamped():
    service, redis_service = _service()

    await service.hold_node_until_probe_passes(EXECUTOR_UUID, "docker_pull", {})

    redis_service.set.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_redis_error_while_stamping_does_not_raise():
    service, redis_service = _service()
    redis_service.set.side_effect = ConnectionError("redis down")

    await service.hold_node_until_probe_passes(EXECUTOR_UUID, "gpu_flags", {})

    redis_service.delete.assert_not_awaited()


def _dropped_mid_reply() -> RentalDockerOperationError:
    # the 8 Nov failures: the Docker reply over SSH cut off mid-body, wrapped by the SDK client
    try:
        try:
            raise requests.exceptions.ChunkedEncodingError(urllib3.exceptions.ProtocolError("Response ended prematurely"))
        except requests.exceptions.ChunkedEncodingError as exc:
            raise RentalDockerOperationError("Docker SDK inspect container failed: Response ended prematurely") from exc
    except RentalDockerOperationError as exc:
        return exc


@pytest.mark.asyncio
async def test_a_create_whose_docker_connection_dropped_stands_as_a_failed_probe():
    service, redis_service = _service()

    await service.hold_node_until_probe_passes(EXECUTOR_UUID, "add_public_keys", {}, error=_dropped_mid_reply())

    redis_service.set.assert_awaited_once_with(
        f"rental_probe_failed:{EXECUTOR_UUID}",
        f"{STEP_CONTAINER_START}:add_public_keys",
    )


@pytest.mark.asyncio
async def test_a_daemon_answer_at_an_image_step_is_not_stamped():
    service, redis_service = _service()

    await service.hold_node_until_probe_passes(
        EXECUTOR_UUID, "docker_run", {}, error=RentalDockerOperationError("Docker SDK start failed: exec format error")
    )

    redis_service.set.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_drop_before_the_validator_reached_the_host_is_not_stamped():
    service, redis_service = _service()

    await service.hold_node_until_probe_passes(EXECUTOR_UUID, "ssh_connect", {}, error=_dropped_mid_reply())

    redis_service.set.assert_not_awaited()


def _raised_while_handling(drop: BaseException, error: BaseException) -> BaseException:
    # a daemon answer raised in an except block that caught a drop: chained implicitly, never caused by it
    try:
        try:
            raise drop
        except type(drop):
            raise error
    except type(error) as exc:
        return exc


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(
            docker.errors.APIError("500 Server Error", response=Mock(status_code=500, reason="Internal Server Error")),
            id="daemon-500",
        ),
        pytest.param(docker.errors.ImageNotFound("No such image: renter/image:latest"), id="image-not-found"),
        pytest.param(
            RentalDockerOperationError("Docker SDK start container failed: OCI runtime create failed"),
            id="oci-runtime",
        ),
        # a non-zero exec exit carries the command's own output, which can say anything
        pytest.param(
            RentalDockerOperationError("Docker exec exited 1: Read timed out; Socket is closed; EOFError"),
            id="exec-nonzero-exit",
        ),
        pytest.param(
            _raised_while_handling(
                ConnectionResetError(),
                docker.errors.APIError("409 Client Error", response=Mock(status_code=409, reason="Conflict")),
            ),
            id="daemon-answer-after-a-drop",
        ),
    ],
)
@pytest.mark.asyncio
async def test_a_normal_daemon_or_exec_error_past_the_connect_is_not_stamped(error):
    service, redis_service = _service()

    await service.hold_node_until_probe_passes(EXECUTOR_UUID, "add_public_keys", {}, error=error)

    redis_service.set.assert_not_awaited()


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(_dropped_mid_reply(), id="chunked-encoding"),
        pytest.param(RentalDockerTransportDropped("channel closed"), id="ssh-channel-closed"),
        pytest.param(
            requests.exceptions.ConnectionError(
                urllib3.exceptions.ProtocolError("Connection aborted.", ConnectionResetError())
            ),
            id="connection-aborted",
        ),
    ],
)
@pytest.mark.asyncio
async def test_a_typed_drop_past_the_connect_is_stamped(error):
    service, redis_service = _service()

    await service.hold_node_until_probe_passes(EXECUTOR_UUID, "add_public_keys", {}, error=error)

    redis_service.set.assert_awaited_once()
