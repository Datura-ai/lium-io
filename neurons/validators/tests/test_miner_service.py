"""DAH-2272: MinerService._handle_container create-request path.

No existing test drove MinerService._handle_container before this; the harness
below mocks every boundary it crosses (bittensor wallet, REST submit, redis
renting_in_progress, ssh decrypt) so the create branch actually executes.

DAH-2272 removed the pre-flag port-check force-remove that used to live here —
probe removal now happens entirely inside create_container, AFTER the
pending-pod flag is set, so a probe the rental kills is covered by
PortConnectivityCheck's renting_in_progress tolerate. This test therefore pins
that a ContainerCreateRequest is delegated to create_container and that
miner_service no longer makes an early wait_for_port_check_containers call.
"""
from unittest.mock import AsyncMock, MagicMock, Mock
from uuid import uuid4

import pytest

from datura.requests.miner_requests import AcceptSSHKeyRequest, ExecutorSSHInfo
from payload_models.payloads import (
    ContainerCreateRequest,
    CustomOptions,
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
        backend_client=MagicMock(),
        file_encrypt_service=MagicMock(),
    )
    return svc


def _wire_common_mocks(mocker, miner_service, executor_id: str):
    """Patch every boundary between MinerService._handle_container's entry and
    the create_container call, so the create branch is reached deterministically."""
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
async def test_create_request_delegates_to_create_container(mocker, miner_service):
    """A ContainerCreateRequest is delegated to create_container, and
    miner_service itself no longer calls wait_for_port_check_containers
    (removal moved into create_container, after the pending-pod flag)."""
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

    create_mock.assert_awaited_once()
    # First positional arg to create_container is the create payload.
    assert create_mock.call_args.args[0] is payload
    # No pre-flag port-check removal in miner_service anymore.
    wait_mock.assert_not_awaited()
