import logging
from unittest.mock import AsyncMock, Mock

import docker.errors
import pytest
import requests
import urllib3

from services.rental_docker_sdk import RentalDockerOperationError, RentalDockerTransportDropped

from payload_models.payloads import ContainerCreated, FailedContainerRequest, WorkloadKind
from services.docker_service import DockerService, is_host_fault_create_failure
from services.gpu_power_limit import POWER_STATE_CMD, GpuPowerRestoreRecord
from services.task.checks.rental_probe import STEP_CONTAINER_START
from test_deploy_optimizations import _patch_happy, _payload, _run, _ssh_client, _ssh_result

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
        # the host's nvidia-smi would not give a GPU its power back before the renter's container
        ("gpu_power_restore", True),
        # our own Redis could not confirm the restore state: not the node's fault
        ("gpu_power_restore_state", False),
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


def _customer_create(monkeypatch, *, redis_get, pl_exit: int):
    """A customer create on a host whose GPU-test reads 320 of 400 W; `nvidia-smi -pl` exits `pl_exit`."""
    service = DockerService(ssh_service=Mock(), redis_service=Mock(get=AsyncMock(side_effect=redis_get), mget=AsyncMock(side_effect=lambda keys: [None] * len(keys)), set=AsyncMock(), delete=AsyncMock(), write_atomically=AsyncMock(), get_rented_machine=AsyncMock(return_value=None), rented_machines_synced=AsyncMock(return_value=True)), attestation_service=Mock())
    ssh_client = _ssh_client(inspect_exit=0)
    _patch_happy(service, monkeypatch, ssh_client)
    default_run = ssh_client.run.side_effect

    def run(cmd, *args, **kwargs):
        if cmd == POWER_STATE_CMD:
            return _ssh_result(stdout="GPU-test, 320, 400, 100, 400\n")
        if " -pl " in cmd:
            return _ssh_result(exit_status=pl_exit)
        if "--query-gpu=power.limit,persistence_mode" in cmd:
            return _ssh_result(stdout=f"{320 if pl_exit else 400}, Enabled\n")
        return default_run(cmd, *args, **kwargs)

    ssh_client.run.side_effect = run
    return service


def _probe_failed_stamps(service: DockerService) -> list[tuple]:
    return [c.args for c in service.redis_service.set.await_args_list if str(c.args[0]).startswith("rental_probe_failed:")]


@pytest.mark.asyncio
async def test_a_customer_create_goes_on_when_our_redis_cannot_read_the_power_restore_records(monkeypatch, caplog):
    # prod: 19 'No connection available' reads in 14 days; the renter must not lose the rent over our Redis
    async def redis_get(key):
        if key.startswith("gpu_power_restore:"):
            raise ConnectionError("No connection available")
        return None

    service = _customer_create(monkeypatch, redis_get=redis_get, pl_exit=0)

    with caplog.at_level(logging.WARNING):
        result = await _run(service, _payload(workload_kind=WorkloadKind.CUSTOMER_RENTAL))

    assert isinstance(result, ContainerCreated)
    assert _probe_failed_stamps(service) == []
    assert any("records could not be read before a rental" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_a_customer_create_fails_as_the_nodes_when_the_host_refuses_a_recorded_restore(monkeypatch):
    record = GpuPowerRestoreRecord(gpu_uuid="GPU-test", watts=400, pod_id="filler", executor_id="ex", capped_at=0.0)

    async def redis_get(key):
        return record.model_dump_json() if key == "gpu_power_restore:GPU-test" else None

    service = _customer_create(monkeypatch, redis_get=redis_get, pl_exit=1)

    result = await _run(service, _payload(workload_kind=WorkloadKind.CUSTOMER_RENTAL))

    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == "gpu_power_restore"
    assert _probe_failed_stamps(service) == [
        (f"rental_probe_failed:{result.executor_id}", f"{STEP_CONTAINER_START}:gpu_power_restore")
    ]
