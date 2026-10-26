"""DAH-5101 — a delete does not wait for the pod's unwritten volume data to reach a slow disk.

Kosta85's node (executor 8c7d58d6-2ac4-4d77-a9b0-d86f5559782b): a tenant writing faster than the
disk left gigabytes of unwritten data in its volume; dockerd unmounts the volume when the container
stops and the unmount flushes it, so the delete hung until the disk had absorbed it all. The rental
is over and the volume is removed anyway: its files are unlinked on the host before the stop, and
the kernel drops an unlinked file's unwritten pages at the last close instead of writing them.
"""

# ruff: noqa: F811 — the docker_service fixture is imported from test_docker_service
from __future__ import annotations

import asyncio
import os
import shlex
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from datura.requests.miner_requests import ExecutorSSHInfo
from payload_models.payloads import ContainerDeleted, ContainerDeleteRequest, WorkloadKind

import services.docker_service as docker_service_module
from services.docker_service import (
    _FILLER_VOLUME_WIPE_TIMEOUT_SEC,
    _VOLUME_WIPE_TIMEOUT_SEC,
    CONTAINER_STOP_GRACE_SECONDS,
    _wipe_volume_cmd,
)
from tests.test_docker_service import (  # noqa: F401
    FAKE_SSH_HOST_KEY,
    DummySSHConnectionManager,
    _make_ssh_command_result,
    docker_service,
    mock_dependencies,
    ssh_client_answering_through_run,
)

WIPE_CMD = _wipe_volume_cmd("volume_dirty")


def _bounded(command: str, seconds: int) -> str:
    return f"timeout -k 5 {seconds} sh -c {shlex.quote(command)}"


def _delete_payload(
    local_volume: str | None = "volume_dirty", workload_kind: WorkloadKind = WorkloadKind.CUSTOMER_RENTAL
) -> ContainerDeleteRequest:
    return ContainerDeleteRequest(
        miner_hotkey="miner",
        executor_id=str(uuid4()),
        pod_id=str(uuid4()),
        workload_kind=workload_kind,
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
        if callable(ssh_answers):
            return await ssh_answers(command)
        if answers is None:
            raise ssh_answers
        return next(answers)

    ssh_client = ssh_client_answering_through_run(AsyncMock(side_effect=run))
    monkeypatch.setattr(
        "services.docker_service.asyncssh.connect",
        Mock(return_value=DummySSHConnectionManager(ssh_client)),
    )
    monkeypatch.setattr("services.docker_service.asyncssh.import_private_key", Mock())
    monkeypatch.setattr(docker_service, "_prepare_known_hosts_policy", AsyncMock(return_value=None))
    docker_service.ssh_service.decrypt_payload = Mock(return_value="private-key")
    docker_service.redis_service.remove_rented_machine = AsyncMock()
    return order


async def _hang_the_wipe(command: str):
    if "propagated-mount" in command:
        await asyncio.Event().wait()
    return _make_ssh_command_result()


async def _delete(docker_service, payload):
    return await docker_service.delete_container(
        payload=payload,
        executor_info=_executor_info(payload),
        keypair=Mock(ss58_address="validator-hotkey"),
        private_key="encrypted",
    )


# ---------------------------------------------------------------------------
# the command, through a real shell and a docker stub
# ---------------------------------------------------------------------------

_DOCKER_ROOT = "/var/lib/docker"
_PLUGIN_ID = "0123456789abcdef"


def _run_wipe(tmp_path: Path, volume: str, driver: str, mountpoint: str, *, fail: str = "") -> subprocess.CompletedProcess:
    """Runs the wipe command under sh against a docker stub whose answers come from the arguments;
    `fail` names the stub call that exits nonzero (driver, mountpoint, info, plugin). Returns the process."""
    host_root = tmp_path / "host"
    docker = tmp_path / "docker"
    docker.write_text(
        "#!/bin/sh\n"
        f"fail={shlex.quote(fail)}\n"
        'case "$*" in\n'
        f'  *"{{{{.Driver}}}}"*) [ "$fail" = driver ] && exit 1; echo {shlex.quote(driver)};;\n'
        f'  *"{{{{.Mountpoint}}}}"*) [ "$fail" = mountpoint ] && exit 1; echo {shlex.quote(mountpoint)};;\n'
        f'  *"{{{{.DockerRootDir}}}}"*) [ "$fail" = info ] && exit 1; echo {_DOCKER_ROOT};;\n'
        f'  *"{{{{.Id}}}}"*) [ "$fail" = plugin ] && exit 1; echo {_PLUGIN_ID};;\n'
        "  *) exit 99;;\n"
        "esac\n"
    )
    docker.chmod(0o755)
    command = _wipe_volume_cmd(volume, docker=str(docker), host_root=str(host_root))
    return subprocess.run(["sh", "-c", command], capture_output=True, text=True)


def _host_files(tmp_path: Path, *dirs: str) -> dict[str, Path]:
    # every host-side dir the wipe might touch, each holding one file that must survive unless wiped
    host_root = tmp_path / "host"
    files = {}
    for rel in dirs:
        directory = host_root / rel.lstrip("/")
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "data.bin").write_bytes(b"x")
        files[rel] = directory / "data.bin"
    return files


_LOCAL_MOUNT = f"{_DOCKER_ROOT}/volumes/volume_dirty/_data"
_VLOOPBACK_HOST_DIR = f"{_DOCKER_ROOT}/plugins/{_PLUGIN_ID}/propagated-mount/volume_dirty"
_OTHER_DIRS = (f"{_DOCKER_ROOT}/volumes/other/_data", f"{_DOCKER_ROOT}/plugins/{_PLUGIN_ID}/propagated-mount/other", "/etc")


@pytest.mark.parametrize(
    "driver, mountpoint, wiped_dir",
    [("local", _LOCAL_MOUNT, _LOCAL_MOUNT), ("vloopback:v3", "/mnt/volume_dirty", _VLOOPBACK_HOST_DIR), ("vloopback", "/mnt/volume_dirty", _VLOOPBACK_HOST_DIR)],
    ids=["local", "vloopback-v3", "vloopback-legacy"],
)
def test_wipe_removes_only_the_volumes_files_under_its_expected_host_dir(tmp_path, driver, mountpoint, wiped_dir):
    files = _host_files(tmp_path, wiped_dir, *_OTHER_DIRS)

    completed = _run_wipe(tmp_path, "volume_dirty", driver, mountpoint)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == f"wiped {wiped_dir}"
    assert not files[wiped_dir].exists()
    assert all(files[other].exists() for other in _OTHER_DIRS)
    assert (tmp_path / "host" / wiped_dir.lstrip("/")).is_dir()


@pytest.mark.parametrize(
    "driver, mountpoint, fail, exit_status",
    [
        ("local", _LOCAL_MOUNT, "driver", 2),
        ("local", _LOCAL_MOUNT, "mountpoint", 2),
        ("local", "", "", 5),
        ("local", f"{_DOCKER_ROOT}/volumes/other/_data", "", 5),
        ("local", "/", "", 5),
        ("vloopback:v3", "/mnt/other", "", 5),
        ("vloopback:v3", "", "", 5),
        ("vloopback:v3", "/mnt/volume_dirty", "info", 3),
        ("vloopback:v3", "/mnt/volume_dirty", "plugin", 3),
        ("nfs", "/mnt/volume_dirty", "", 5),
        ("local", f"{_DOCKER_ROOT}/volumes/volume_dirty/_data", "none", 4),
    ],
    ids=[
        "driver-read-fails", "mountpoint-read-fails", "empty-mountpoint", "other-volumes-mountpoint",
        "root-mountpoint", "other-vloopback-mountpoint", "empty-vloopback-mountpoint",
        "docker-root-read-fails", "plugin-id-read-fails", "unexpected-driver", "host-dir-missing",
    ],
)
def test_wipe_removes_nothing_when_a_lookup_fails_or_answers_an_unexpected_path(tmp_path, driver, mountpoint, fail, exit_status):
    # the BLOCKER of the review: an empty or foreign path must never reach the rm
    dirs = _OTHER_DIRS + (_VLOOPBACK_HOST_DIR,) if exit_status != 4 else _OTHER_DIRS
    files = _host_files(tmp_path, *dirs)

    completed = _run_wipe(tmp_path, "volume_dirty", driver, mountpoint, fail=fail)

    assert completed.returncode == exit_status, completed.stdout + completed.stderr
    assert all(file.exists() for file in files.values())
    assert sorted(os.listdir(tmp_path / "host")) == sorted({rel.lstrip("/").split("/")[0] for rel in dirs})


def test_wipe_reports_a_failed_rm_instead_of_a_success(tmp_path):
    files = _host_files(tmp_path, _LOCAL_MOUNT)
    (files[_LOCAL_MOUNT].parent / "data.bin").chmod(0o444)
    files[_LOCAL_MOUNT].parent.chmod(0o555)
    try:
        completed = _run_wipe(tmp_path, "volume_dirty", "local", _LOCAL_MOUNT)
    finally:
        files[_LOCAL_MOUNT].parent.chmod(0o755)

    assert completed.returncode != 0
    assert "wiped" not in completed.stdout


def test_wipe_command_builds_the_path_from_the_name_and_runs_no_container():
    assert WIPE_CMD.startswith("v=volume_dirty; d=$(/usr/bin/docker volume inspect -f '{{.Driver}}' $v) || exit 2; ")
    assert "p=$root/plugins/$id/propagated-mount/$v" in WIPE_CMD
    assert "$(basename" not in WIPE_CMD
    assert 'find "/proc/1/root$p" -mindepth 1 -maxdepth 1 -exec rm -rf {} + && echo "wiped $p"' in WIPE_CMD
    assert "docker run" not in WIPE_CMD and "docker exec" not in WIPE_CMD


# ---------------------------------------------------------------------------
# the wipe in delete_container
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_wipes_the_volume_before_the_graceful_stop(docker_service, monkeypatch):
    order = _wire_delete(docker_service, monkeypatch, [_make_ssh_command_result(stdout="wiped /x\n")])
    payload = _delete_payload()

    result = await _delete(docker_service, payload)

    assert isinstance(result, ContainerDeleted)
    assert order[:3] == [
        ("ssh", _bounded(WIPE_CMD, _VOLUME_WIPE_TIMEOUT_SEC)),
        ("stop", payload.container_name),
        ("remove", payload.container_name),
    ]
    client = docker_service.rental_docker_client_factory.client
    assert client.stop_grace_seconds_calls == [CONTAINER_STOP_GRACE_SECONDS]


@pytest.mark.asyncio
async def test_filler_delete_gives_the_wipe_a_budget_that_fits_the_backends_stop_wait(docker_service, monkeypatch):
    order = _wire_delete(docker_service, monkeypatch, [_make_ssh_command_result(stdout="wiped /x\n")])
    payload = _delete_payload(workload_kind=WorkloadKind.FILLER)

    result = await _delete(docker_service, payload)

    assert isinstance(result, ContainerDeleted)
    assert order[0] == ("ssh", _bounded(WIPE_CMD, _FILLER_VOLUME_WIPE_TIMEOUT_SEC))
    assert _FILLER_VOLUME_WIPE_TIMEOUT_SEC + docker_service_module.FILLER_CONTAINER_STOP_GRACE_SECONDS < 30


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
        [_make_ssh_command_result(exit_status=124)],
        OSError("ssh transport closed"),
    ],
    ids=["wipe-fails", "host-timeout", "ssh-raises"],
)
@pytest.mark.asyncio
async def test_delete_proceeds_as_before_when_the_wipe_fails(docker_service, monkeypatch, ssh_answers):
    order = _wire_delete(docker_service, monkeypatch, ssh_answers)
    payload = _delete_payload()

    result = await _delete(docker_service, payload)

    assert isinstance(result, ContainerDeleted)
    assert order[:3] == [
        ("ssh", _bounded(WIPE_CMD, _VOLUME_WIPE_TIMEOUT_SEC)),
        ("stop", payload.container_name),
        ("remove", payload.container_name),
    ]


@pytest.mark.asyncio
async def test_delete_closes_the_wipes_channel_when_the_host_does_not_answer_in_time(docker_service, monkeypatch):
    # a wipe the host did not end in time must not hold the only session the stop's cleanup needs
    monkeypatch.setattr(docker_service_module, "_VOLUME_WIPE_TIMEOUT_SEC", 0.05)
    monkeypatch.setattr(docker_service_module, "_HOST_TIMEOUT_VALIDATOR_MARGIN_SECONDS", 0.05)
    order = _wire_delete(docker_service, monkeypatch, _hang_the_wipe)
    payload = _delete_payload()

    result = await asyncio.wait_for(_delete(docker_service, payload), timeout=5)

    assert isinstance(result, ContainerDeleted)
    assert [kind for kind, _ in order[:3]] == ["ssh", "stop", "remove"]
