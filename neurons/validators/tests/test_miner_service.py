"""DAH-2272: early-caller port-check behavior in MinerService._handle_container.

No existing test drove MinerService._handle_container before this; the harness
below mocks every boundary MinerService._handle_container crosses (bittensor
wallet, REST submit, redis renting_in_progress, ssh decrypt) so the
`wait_for_port_check_containers` call at miner_service.py:~1749 actually
executes. DAH-2272 removed the wait/poll loop (the rental now force-removes any
lingering probe immediately), so these tests pin the surrounding control flow
rather than a timing event: on success the create proceeds; if the step ever
reports failure the container request fails cleanly without attempting a create.
"""
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from datura.requests.miner_requests import AcceptSSHKeyRequest, ExecutorSSHInfo
from payload_models.payloads import (
    ContainerCreateRequest,
    CustomOptions,
    FailedContainerRequest,
    PayloadPortMapping,
)
from services.miner_service import MinerService


def _make_executor_info(executor_id: str) -> ExecutorSSHInfo:
    return ExecutorSSHInfo(
        uuid=executor_id,
        address="127.0.0.1",
        port=8001,
        ssh_username="root",
        ssh_port=2200,
        python_path="/usr/bin/python3",
        root_dir="/root/app",
        port_range="9100-9130",
    )


def _make_create_payload(executor_id: str) -> ContainerCreateRequest:
    return ContainerCreateRequest(
        miner_hotkey="5TestMiner",
        executor_id=executor_id,
        miner_address="10.0.0.1",
        miner_port=8000,
        pod_id=str(uuid4()),
        docker_image="daturaai/pytorch:test",
        user_public_keys=["ssh-ed25519 test-key"],
        gpu_uuids=["GPU-test"],
        cpu_count=2,
        memory_gb=8,
        custom_options=CustomOptions(volumes=[], environment={}, startup_commands=None, shm_size="1g"),
        volume_limit_gb=100,
        storage_limit_gb=50,
        available_ports=[PayloadPortMapping(internal_port=22, external_port=30022)],
        pod_mapping=[],
        active_container_names=[],
        active_volume_names=[],
    )


@pytest.fixture
def miner_service(mocker):
    ssh_service = mocker.Mock()
    ssh_service.generate_ssh_key.return_value = (b"---PRIV---", b"ssh-ed25519 pub")
    ssh_service.decrypt_payload.return_value = "---DECRYPTED-PRIV---"

    redis_service = mocker.AsyncMock()
    redis_service.renting_in_progress = AsyncMock(return_value=False)
    redis_service.remove_rented_machine = AsyncMock()

    svc = MinerService(
        ssh_service=ssh_service,
        task_service=mocker.Mock(),
        redis_service=redis_service,
        attestation_service=mocker.Mock(),
    )
    return svc


def _wire_common_mocks(mocker, miner_service, executor_id: str):
    """Patch every boundary between MinerService._handle_container's entry and
    the wait_for_port_check_containers call, so only that call's outcome varies
    between the success and failure test cases."""
    my_key = Mock(ss58_address="validator-hotkey")
    my_key.sign.return_value = b"\x01\x02\x03"
    my_key.get_hotkey = Mock(return_value=my_key)

    # `settings` is a pydantic Settings *instance* — patch the method on the
    # class (get_bittensor_wallet is a plain method, not a pydantic field).
    mocker.patch(
        "core.config.Settings.get_bittensor_wallet",
        return_value=Mock(get_hotkey=Mock(return_value=my_key)),
    )

    accept_msg = AcceptSSHKeyRequest(executors=[_make_executor_info(executor_id)])
    mocker.patch.object(
        miner_service,
        "_make_rest_request",
        AsyncMock(return_value=(200, {"message_type": "AcceptSSHKeyRequest"})),
    )
    mocker.patch("services.miner_service._parse_miner_response", return_value=accept_msg)
    mocker.patch("services.miner_service.asyncssh.import_private_key", return_value=Mock())

    return my_key


@pytest.mark.asyncio
async def test_early_call_site_proceeds_on_success(mocker, miner_service):
    """When the port-check force-remove reports success, container creation proceeds."""
    executor_id = str(uuid4())
    payload = _make_create_payload(executor_id)
    _wire_common_mocks(mocker, miner_service, executor_id)

    wait_mock = mocker.patch(
        "services.miner_service.DockerService.wait_for_port_check_containers",
        AsyncMock(return_value=(True, "No port check containers found")),
    )
    create_mock = mocker.patch(
        "services.miner_service.DockerService.create_container",
        AsyncMock(return_value=Mock()),
    )

    await miner_service._handle_container(payload)

    wait_mock.assert_awaited_once()
    create_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_early_call_site_fails_container_when_step_reports_failure(mocker, miner_service):
    """Defensive: if the port-check step ever reports failure, the container
    request fails cleanly and no create is attempted. (Post-DAH-2272 the step
    always succeeds, so this pins the guard, not a live code path.)"""
    executor_id = str(uuid4())
    payload = _make_create_payload(executor_id)
    _wire_common_mocks(mocker, miner_service, executor_id)

    mocker.patch(
        "services.miner_service.DockerService.wait_for_port_check_containers",
        AsyncMock(return_value=(False, "Port check containers still present")),
    )
    create_mock = mocker.patch(
        "services.miner_service.DockerService.create_container",
        AsyncMock(return_value=Mock()),
    )

    result = await miner_service._handle_container(payload)

    assert isinstance(result, FailedContainerRequest)
    create_mock.assert_not_awaited()
