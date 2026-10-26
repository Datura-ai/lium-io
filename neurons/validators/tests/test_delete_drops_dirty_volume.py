"""DAH-5101 — a delete does not wait for the pod's unwritten volume data to reach a slow disk.

Kosta85's node (executor 8c7d58d6-2ac4-4d77-a9b0-d86f5559782b): a tenant writing faster than the
disk left gigabytes of dirty pages in the vloopback image; dockerd's unmount synced the whole
Docker filesystem and the delete hung until the disk had absorbed them. The rental is over and
the volume is removed anyway, so the validator drops that data before dockerd gets to the unmount.
"""

# ruff: noqa: F811 — the docker_service fixture is imported from test_docker_service
from __future__ import annotations

from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from datura.requests.miner_requests import ExecutorSSHInfo
from payload_models.payloads import ContainerDeleted, ContainerDeleteRequest, WorkloadKind

from core.docker_utils import DockerCommand
from services.docker_service import (
    CONTAINER_STOP_GRACE_SECONDS,
    FILLER_CONTAINER_STOP_GRACE_SECONDS,
    _BoundLog,
    _vloopback_drop_dirty_volume_cmd,
)
from tests.test_docker_service import (  # noqa: F401
    FAKE_SSH_HOST_KEY,
    DummySSHConnectionManager,
    _make_ssh_command_result,
    docker_service,
    mock_dependencies,
)


def _delete_payload(workload_kind: WorkloadKind = WorkloadKind.CUSTOMER_RENTAL) -> ContainerDeleteRequest:
    return ContainerDeleteRequest(
        miner_hotkey="miner",
        executor_id=str(uuid4()),
        pod_id=str(uuid4()),
        workload_kind=workload_kind,
        container_name="pod_dirty",
        local_volume="volume_dirty",
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


def _ssh_answers_for_vloopback_volume() -> list:
    return [
        _make_ssh_command_result(stdout="vloopback:latest\n"),  # docker volume inspect
        _make_ssh_command_result(stdout="plugin123\n"),  # docker plugin inspect
        _make_ssh_command_result(stdout="/var/lib/docker\n"),  # docker info
        _make_ssh_command_result(stdout="pid=4242 exited=yes\n"),  # stop_container_init
        _make_ssh_command_result(stdout="wiped=yes truncated=yes dev=/dev/loop3 img=/mnt/rootfs/var/lib/docker/loopback/volume_dirty\n"),
    ]


def _wire_delete(docker_service, monkeypatch, ssh_client: AsyncMock) -> None:
    monkeypatch.setattr(
        "services.docker_service.asyncssh.connect",
        Mock(return_value=DummySSHConnectionManager(ssh_client)),
    )
    monkeypatch.setattr("services.docker_service.asyncssh.import_private_key", Mock())
    monkeypatch.setattr(docker_service, "_prepare_known_hosts_policy", AsyncMock(return_value=None))
    docker_service.ssh_service.decrypt_payload = Mock(return_value="private-key")
    docker_service.redis_service.remove_rented_machine = AsyncMock()


def test_stop_command_signals_term_then_kill_and_waits_on_proc():
    cmd = DockerCommand.stop_container_init("pod_dirty", 30, 20)

    assert "docker inspect -f '{{.State.Pid}}' pod_dirty" in cmd
    assert "kill -TERM $pid" in cmd and "kill -9 $pid" in cmd
    assert "-lt 30" in cmd and "-lt 20" in cmd
    assert "/proc/$pid" in cmd
    assert "docker stop" not in cmd  # docker stop waits for the unmount, which waits for the disk


def test_drop_command_unlinks_the_volume_files_and_truncates_its_image():
    cmd = _vloopback_drop_dirty_volume_cmd("/var/lib/docker/plugins/plugin123", "volume_dirty")

    assert "--mount type=bind,src=/var/lib/docker/plugins/plugin123,dst=/mnt" in cmd
    assert "docker.io/library/alpine:3.19" in cmd
    assert "/mnt/propagated-mount/volume_dirty" in cmd
    assert "-mindepth 1 -maxdepth 1 -exec rm -rf {} +" in cmd  # the files, never the mountpoint
    assert "loop/backing_file" in cmd and "truncate -s 0" in cmd
    assert "losetup" not in cmd  # detaching flushes first; truncate is the only step that does not


def test_drop_command_quotes_the_volume_name():
    cmd = _vloopback_drop_dirty_volume_cmd("/var/lib/docker/plugins/p", "vol; rm -rf /")

    # shlex-quoted inside the helper script, which is shlex-quoted again for `sh -c`
    assert """'"'"'vol; rm -rf /'"'"'""" in cmd


@pytest.mark.asyncio
async def test_delete_drops_dirty_volume_instead_of_the_graceful_stop(docker_service, monkeypatch):
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(side_effect=_ssh_answers_for_vloopback_volume())
    _wire_delete(docker_service, monkeypatch, ssh_client)
    payload = _delete_payload()

    result = await docker_service.delete_container(
        payload=payload, executor_info=_executor_info(payload),
        keypair=Mock(ss58_address="validator-hotkey"), private_key="encrypted",
    )

    assert isinstance(result, ContainerDeleted)
    commands = [call.args[0] for call in ssh_client.run.await_args_list]
    assert commands[0] == "/usr/bin/docker volume inspect volume_dirty --format '{{.Driver}}'"
    assert commands[1] == "/usr/bin/docker plugin inspect vloopback:latest --format '{{.Id}}'"
    assert ".State.Pid" in commands[3] and f"-lt {CONTAINER_STOP_GRACE_SECONDS}" in commands[3]
    assert "truncate -s 0" in commands[4]
    assert "src=/var/lib/docker/plugins/plugin123,dst=/mnt" in commands[4]
    client = docker_service.rental_docker_client_factory.client
    assert client.container_call_order == [("remove", payload.container_name)]  # no docker stop
    assert client.stop_grace_seconds_calls == []


@pytest.mark.asyncio
async def test_delete_of_a_filler_uses_the_filler_grace(docker_service, monkeypatch):
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(side_effect=_ssh_answers_for_vloopback_volume())
    _wire_delete(docker_service, monkeypatch, ssh_client)
    payload = _delete_payload(WorkloadKind.FILLER)

    await docker_service.delete_container(
        payload=payload, executor_info=_executor_info(payload),
        keypair=Mock(ss58_address="validator-hotkey"), private_key="encrypted",
    )

    stop_cmd = ssh_client.run.await_args_list[3].args[0]
    assert f"-lt {FILLER_CONTAINER_STOP_GRACE_SECONDS}" in stop_cmd


@pytest.mark.asyncio
async def test_delete_keeps_the_graceful_stop_for_a_non_vloopback_volume(docker_service, monkeypatch):
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(return_value=_make_ssh_command_result(stdout="local\n"))
    _wire_delete(docker_service, monkeypatch, ssh_client)
    payload = _delete_payload()

    result = await docker_service.delete_container(
        payload=payload, executor_info=_executor_info(payload),
        keypair=Mock(ss58_address="validator-hotkey"), private_key="encrypted",
    )

    assert isinstance(result, ContainerDeleted)
    client = docker_service.rental_docker_client_factory.client
    assert client.container_call_order == [("stop", payload.container_name), ("remove", payload.container_name)]
    assert not any("truncate" in call.args[0] for call in ssh_client.run.await_args_list)


@pytest.mark.asyncio
async def test_delete_falls_back_to_the_graceful_stop_when_the_drop_step_raises(docker_service, monkeypatch):
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(side_effect=OSError("ssh transport closed"))
    _wire_delete(docker_service, monkeypatch, ssh_client)
    payload = _delete_payload()

    result = await docker_service.delete_container(
        payload=payload, executor_info=_executor_info(payload),
        keypair=Mock(ss58_address="validator-hotkey"), private_key="encrypted",
    )

    assert isinstance(result, ContainerDeleted)
    client = docker_service.rental_docker_client_factory.client
    assert client.container_call_order[0] == ("stop", payload.container_name)


@pytest.mark.asyncio
async def test_drop_step_logs_a_failed_helper_and_still_reports_the_processes_stopped(docker_service):
    answers = _ssh_answers_for_vloopback_volume()
    answers[4] = _make_ssh_command_result(exit_status=125, stderr="docker: Error response from daemon")
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(side_effect=answers)

    stopped = await docker_service._stop_and_drop_dirty_volume(ssh_client, _delete_payload(), _BoundLog({}))

    assert stopped is True  # the processes are gone either way; a second docker stop buys nothing
