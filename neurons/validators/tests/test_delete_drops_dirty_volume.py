"""DAH-5101 — a delete does not wait for the pod's unwritten volume data to reach a slow disk.

Kosta85's node (executor 8c7d58d6-2ac4-4d77-a9b0-d86f5559782b): a tenant writing faster than the
disk left gigabytes of unwritten data in its volume; dockerd unmounts the volume when the container
stops and the unmount flushes it, so the delete hung until the disk had absorbed it all. The rental
is over and the volume is removed anyway: its files are unlinked on the host before the stop, and
the kernel drops an unlinked file's unwritten pages at the last close instead of writing them.
"""

# ruff: noqa: F811 — the docker_service fixture is imported from test_docker_service
from __future__ import annotations

from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from datura.requests.miner_requests import ExecutorSSHInfo
from payload_models.payloads import ContainerDeleted, ContainerDeleteRequest, WorkloadKind

from services.docker_service import CONTAINER_STOP_GRACE_SECONDS, _wipe_volume_cmd
from tests.test_docker_service import (  # noqa: F401
    FAKE_SSH_HOST_KEY,
    DummySSHConnectionManager,
    _make_ssh_command_result,
    docker_service,
    mock_dependencies,
)

WIPE_CMD = _wipe_volume_cmd("volume_dirty")


def _delete_payload(local_volume: str | None = "volume_dirty") -> ContainerDeleteRequest:
    return ContainerDeleteRequest(
        miner_hotkey="miner",
        executor_id=str(uuid4()),
        pod_id=str(uuid4()),
        workload_kind=WorkloadKind.CUSTOMER_RENTAL,
        container_name="pod_dirty",
        local_volume=local_volume,
    )


def _executor_info(payload: ContainerDeleteRequest) -> ExecutorSSHInfo:
    return ExecutorSSHInfo(
        uuid=payload.executor_id,
        address="127.0.0.1",
        port=8080,
        ssh_username="root",
        ssh_port=2200,
        python_path="/usr/bin/python",
        root_dir="/root/app",
        ssh_host_key=FAKE_SSH_HOST_KEY,
    )


def _wire_delete(docker_service, monkeypatch, ssh_answers) -> list:
    """Returns one ordered log shared with the fake docker client, so the test sees the ssh
    commands and the SDK calls in the order they ran."""
    client = docker_service.rental_docker_client_factory.client
    order = client.container_call_order
    answers = iter(ssh_answers) if isinstance(ssh_answers, list) else None

    async def run(command, **kwargs):
        order.append(("ssh", command))
        if answers is None:
            raise ssh_answers
        return next(answers)

    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(side_effect=run)
    monkeypatch.setattr(
        "services.docker_service.asyncssh.connect",
        Mock(return_value=DummySSHConnectionManager(ssh_client)),
    )
    monkeypatch.setattr("services.docker_service.asyncssh.import_private_key", Mock())
    monkeypatch.setattr(docker_service, "_prepare_known_hosts_policy", AsyncMock(return_value=None))
    docker_service.ssh_service.decrypt_payload = Mock(return_value="private-key")
    docker_service.redis_service.remove_rented_machine = AsyncMock()
    return order


async def _delete(docker_service, payload):
    return await docker_service.delete_container(
        payload=payload,
        executor_info=_executor_info(payload),
        keypair=Mock(ss58_address="validator-hotkey"),
        private_key="encrypted",
    )


def test_wipe_command_unlinks_the_files_on_the_host_without_a_container():
    assert WIPE_CMD.startswith("v=volume_dirty; d=$(/usr/bin/docker volume inspect -f '{{.Driver}}' $v)")
    assert "p=$root/plugins/$id/propagated-mount/$(basename \"$m\")" in WIPE_CMD
    assert 'find "/proc/1/root$p" -mindepth 1 -maxdepth 1 -exec rm -rf {} +' in WIPE_CMD
    assert "docker run" not in WIPE_CMD and "docker exec" not in WIPE_CMD


@pytest.mark.asyncio
async def test_delete_wipes_the_volume_before_the_graceful_stop(docker_service, monkeypatch):
    order = _wire_delete(docker_service, monkeypatch, [_make_ssh_command_result(stdout="wiped /x\n")])
    payload = _delete_payload()

    result = await _delete(docker_service, payload)

    assert isinstance(result, ContainerDeleted)
    assert order[:3] == [("ssh", WIPE_CMD), ("stop", payload.container_name), ("remove", payload.container_name)]
    client = docker_service.rental_docker_client_factory.client
    assert client.stop_grace_seconds_calls == [CONTAINER_STOP_GRACE_SECONDS]


@pytest.mark.asyncio
async def test_delete_without_a_volume_does_not_wipe(docker_service, monkeypatch):
    # an unsafe volume name never reaches the wipe: delete_container refuses it first
    order = _wire_delete(docker_service, monkeypatch, [_make_ssh_command_result()])
    payload = _delete_payload(local_volume=None)

    result = await _delete(docker_service, payload)

    assert isinstance(result, ContainerDeleted)
    assert not any("propagated-mount" in command for kind, command in order if kind == "ssh")
    assert order[:2] == [("stop", payload.container_name), ("remove", payload.container_name)]


@pytest.mark.parametrize(
    "ssh_answers",
    [
        [_make_ssh_command_result(exit_status=4, stdout="no dir /x")],
        OSError("ssh transport closed"),
    ],
    ids=["wipe-fails", "ssh-raises"],
)
@pytest.mark.asyncio
async def test_delete_proceeds_as_before_when_the_wipe_fails(docker_service, monkeypatch, ssh_answers):
    order = _wire_delete(docker_service, monkeypatch, ssh_answers)
    payload = _delete_payload()

    result = await _delete(docker_service, payload)

    assert isinstance(result, ContainerDeleted)
    assert order[:3] == [("ssh", WIPE_CMD), ("stop", payload.container_name), ("remove", payload.container_name)]
