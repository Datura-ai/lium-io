"""DAH-5101: a delete does not wait for the pod's unwritten volume data to reach a slow disk.
The volume's files are unlinked on the host before the stop, so the kernel drops their unwritten
pages at the last close instead of flushing them at the unmount (Kosta85's node, 8c7d58d6-2ac4-4d77-a9b0-d86f5559782b).
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
from core.config import settings
from core.docker_utils import DockerCommand
from services.docker_service import (
    _OWN_SSH_CONNECTION_MARGIN_SEC,
    _VOLUME_WIPE_TIMEOUT_SEC,
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

WIPE_CMD = _wipe_volume_cmd("volume_dirty", "pod_dirty")
TERM_WAIT_CMD = DockerCommand.terminate_and_wait("pod_dirty", 30)


def _bounded(command: str, seconds: float) -> str:
    return f"timeout -k 5 {seconds} sh -c {shlex.quote(command)}"


def _delete_payload(
    local_volume: str | None = "volume_dirty",
    workload_kind: WorkloadKind = WorkloadKind.CUSTOMER_RENTAL,
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
    commands and the SDK calls in the order they ran. `ssh_answers` is a callable (command ->
    result), a list consumed in order, or an exception every ssh command raises."""
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
    monkeypatch.setattr(settings, "POD_VOLUME_WIPE_BEFORE_STOP_ENABLED", True)
    return order


def _host_answers(wipe=None, wait=None):
    # one answer per host command, by the command's purpose; the default host wipes and the
    # container exits inside the grace
    async def answer(command: str):
        if "propagated-mount" in command:
            return wipe or _make_ssh_command_result(stdout="wiped /x\n")
        if "docker wait" in command:
            if isinstance(wait, BaseException):
                raise wait
            return wait or _make_ssh_command_result(stdout="0\n")
        return _make_ssh_command_result()

    return answer


async def _hang_the_wipe(command: str):
    if "propagated-mount" in command:
        await asyncio.Event().wait()
    return await _host_answers()(command)


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


def _run_wipe(
    tmp_path: Path,
    volume: str,
    driver: str,
    mountpoint: str,
    *,
    users: str = "pod_dirty",
    fail: str = "",
) -> subprocess.CompletedProcess:
    """Runs the wipe command under sh against a docker stub whose answers come from the arguments;
    `users` is what `docker ps` lists on the volume, `fail` names the stub call that exits nonzero
    (inspect, ps, info, plugin). Returns the process."""
    host_root = tmp_path / "host"
    docker = tmp_path / "docker"
    docker.write_text(
        "#!/bin/sh\n"
        f"fail={shlex.quote(fail)}\n"
        'case "$*" in\n'
        f'  *"{{{{.Driver}}}} {{{{.Mountpoint}}}}"*) [ "$fail" = inspect ] && exit 1; echo {shlex.quote(driver + " " + mountpoint)};;\n'
        f'  *"ps -a --filter volume="*) [ "$fail" = ps ] && exit 1; printf %s {shlex.quote(users)};;\n'
        f'  *"{{{{.DockerRootDir}}}}"*) [ "$fail" = info ] && exit 1; echo {_DOCKER_ROOT};;\n'
        f'  *"{{{{.Id}}}}"*) [ "$fail" = plugin ] && exit 1; echo {_PLUGIN_ID};;\n'
        "  *) exit 99;;\n"
        "esac\n"
    )
    docker.chmod(0o755)
    command = (
        _wipe_volume_cmd(volume, "pod_dirty")
        .replace("/usr/bin/docker", str(docker))
        .replace("/proc/1/root", str(host_root))
    )
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
_OTHER_DIRS = (
    f"{_DOCKER_ROOT}/volumes/other/_data",
    f"{_DOCKER_ROOT}/plugins/{_PLUGIN_ID}/propagated-mount/other",
    "/etc",
)


@pytest.mark.parametrize(
    "driver, mountpoint, wiped_dir",
    [
        ("local", _LOCAL_MOUNT, _LOCAL_MOUNT),
        ("vloopback:v3", "/mnt/volume_dirty", _VLOOPBACK_HOST_DIR),
        ("vloopback", "/mnt/volume_dirty", _VLOOPBACK_HOST_DIR),
    ],
)
@pytest.mark.parametrize("users", ["pod_dirty", ""], ids=["pod-on-volume", "pod-already-gone"])
def test_wipe_removes_only_the_volumes_files_under_its_expected_host_dir(
    tmp_path, driver, mountpoint, wiped_dir, users
):
    files = _host_files(tmp_path, wiped_dir, *_OTHER_DIRS)

    completed = _run_wipe(tmp_path, "volume_dirty", driver, mountpoint, users=users)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == f"wiped {wiped_dir}"
    assert "docker run" not in WIPE_CMD and "docker exec" not in WIPE_CMD
    assert not files[wiped_dir].exists()
    assert all(files[other].exists() for other in _OTHER_DIRS)
    assert (tmp_path / "host" / wiped_dir.lstrip("/")).is_dir()


@pytest.mark.parametrize(
    "driver, mountpoint, users, fail, exit_status",
    [
        ("local", _LOCAL_MOUNT, "pod_dirty", "inspect", 2),
        ("local", _LOCAL_MOUNT, "pod_dirty", "ps", 2),
        ("local", "", "pod_dirty", "", 5),
        ("local", f"{_DOCKER_ROOT}/volumes/other/_data", "pod_dirty", "", 5),
        ("local", "/", "pod_dirty", "", 5),
        ("vloopback:v3", "/mnt/other", "pod_dirty", "", 5),
        ("vloopback:v3", "", "pod_dirty", "", 5),
        ("vloopback:v3", "/mnt/volume_dirty", "pod_dirty", "info", 3),
        ("vloopback:v3", "/mnt/volume_dirty", "pod_dirty", "plugin", 3),
        ("nfs", "/mnt/volume_dirty", "pod_dirty", "", 5),
        ("local", _LOCAL_MOUNT, "pod_dirty", "none", 4),
        # a running backup's helper still reads the volume (review, 27.10)
        ("local", _LOCAL_MOUNT, "pod_dirty\nlium-storage-0123456789ab", "", 6),
        ("local", _LOCAL_MOUNT, "lium-storage-0123456789ab", "", 6),
        ("vloopback:v3", "/mnt/volume_dirty", "pod_dirty\nlium-storage-0123456789ab", "", 6),
    ],
)
def test_wipe_removes_nothing_when_a_lookup_fails_or_the_volume_is_shared_or_the_path_is_unexpected(
    tmp_path, driver, mountpoint, users, fail, exit_status
):
    # the BLOCKER of the review: an empty or foreign path must never reach the rm
    dirs = _OTHER_DIRS + (_VLOOPBACK_HOST_DIR, _LOCAL_MOUNT) if exit_status != 4 else _OTHER_DIRS
    files = _host_files(tmp_path, *dirs)

    completed = _run_wipe(tmp_path, "volume_dirty", driver, mountpoint, users=users, fail=fail)

    assert completed.returncode == exit_status, completed.stdout + completed.stderr
    assert all(file.exists() for file in files.values())
    assert sorted(os.listdir(tmp_path / "host")) == sorted(
        {rel.lstrip("/").split("/")[0] for rel in dirs}
    )


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


# ---------------------------------------------------------------------------
# the wipe in delete_container
# ---------------------------------------------------------------------------

_TERM_WAIT_SECONDS = 30 + _OWN_SSH_CONNECTION_MARGIN_SEC


@pytest.mark.asyncio
async def test_delete_wipes_then_terminates_and_removes_a_container_that_exits_in_the_grace(
    docker_service, monkeypatch
):
    order = _wire_delete(docker_service, monkeypatch, _host_answers())
    payload = _delete_payload()

    result = await _delete(docker_service, payload)

    assert isinstance(result, ContainerDeleted)
    assert order[:3] == [
        ("ssh", _bounded(WIPE_CMD, _VOLUME_WIPE_TIMEOUT_SEC)),
        ("ssh", _bounded(TERM_WAIT_CMD, _TERM_WAIT_SECONDS)),
        ("remove", payload.container_name),
    ]
    # the image's STOPSIGNAL, as `docker stop` sends it; SIGTERM when the image sets none
    assert "inspect --format '{{.Config.StopSignal}}'" in TERM_WAIT_CMD
    assert 'kill --signal "${s:-SIGTERM}"' in TERM_WAIT_CMD
    assert "timeout 30 /usr/bin/docker wait" in TERM_WAIT_CMD


@pytest.mark.asyncio
async def test_delete_wipes_again_before_the_kill_when_the_tenant_ignores_sigterm(
    docker_service, monkeypatch
):
    # the CRITICAL of the review: what the tenant writes after the first wipe must not be flushed
    # at the unmount either
    still_running = _make_ssh_command_result(exit_status=124)
    order = _wire_delete(docker_service, monkeypatch, _host_answers(wait=still_running))
    dial = AsyncMock(return_value={})
    monkeypatch.setattr(docker_service_module, "public_host_sock", dial)
    payload = _delete_payload()

    result = await _delete(docker_service, payload)

    assert isinstance(result, ContainerDeleted)
    assert order[:4] == [
        ("ssh", _bounded(WIPE_CMD, _VOLUME_WIPE_TIMEOUT_SEC)),
        ("ssh", _bounded(TERM_WAIT_CMD, _TERM_WAIT_SECONDS)),
        ("ssh", _bounded(WIPE_CMD, _VOLUME_WIPE_TIMEOUT_SEC)),
        ("remove", payload.container_name),
    ]
    # ssh_client plus one connection per wipe and one for the wait, each dialled afresh: a pinned
    # socket is consumed by the connection it was dialled for
    assert dial.await_count == 4


@pytest.mark.parametrize(
    "wait",
    [
        OSError("ssh transport closed"),
        _make_ssh_command_result(exit_status=1, stderr="docker: API error"),
    ],
    ids=["ssh-raised", "docker-failed"],
)
@pytest.mark.asyncio
async def test_delete_wipes_again_when_the_wait_itself_fails(docker_service, monkeypatch, wait):
    # a wait that did not answer, or a `docker wait` that failed, does not prove the container
    # exited, so the second wipe runs (harmless on a container already gone), and the forced removal
    # keeps its own untouched connection for the kill
    order = _wire_delete(docker_service, monkeypatch, _host_answers(wait=wait))
    payload = _delete_payload()

    result = await _delete(docker_service, payload)

    assert isinstance(result, ContainerDeleted)
    assert order[:4] == [
        ("ssh", _bounded(WIPE_CMD, _VOLUME_WIPE_TIMEOUT_SEC)),
        ("ssh", _bounded(TERM_WAIT_CMD, _TERM_WAIT_SECONDS)),
        ("ssh", _bounded(WIPE_CMD, _VOLUME_WIPE_TIMEOUT_SEC)),
        ("remove", payload.container_name),
    ]


@pytest.mark.asyncio
async def test_delete_stops_a_filler_as_before(docker_service, monkeypatch):
    # a filler's delete must answer inside the backend's 30 s, which the 15 s grace and the removal
    # already fill: no wipe, `docker stop` as today
    order = _wire_delete(docker_service, monkeypatch, [_make_ssh_command_result()])
    payload = _delete_payload(workload_kind=WorkloadKind.FILLER)

    result = await _delete(docker_service, payload)

    assert isinstance(result, ContainerDeleted)
    assert not any("propagated-mount" in command for kind, command in order if kind == "ssh")
    assert order[:2] == [("stop", payload.container_name), ("remove", payload.container_name)]


@pytest.mark.parametrize(
    "local_volume, flag", [(None, True), ("volume_dirty", False)], ids=["no-volume", "flag-off"]
)
@pytest.mark.asyncio
async def test_delete_stops_as_before_without_a_volume_or_with_the_flag_off(
    docker_service, monkeypatch, local_volume, flag
):
    # an unsafe volume name never reaches the wipe either: delete_container refuses it first
    order = _wire_delete(docker_service, monkeypatch, [_make_ssh_command_result()])
    monkeypatch.setattr(settings, "POD_VOLUME_WIPE_BEFORE_STOP_ENABLED", flag)
    payload = _delete_payload(local_volume=local_volume)

    result = await _delete(docker_service, payload)

    assert isinstance(result, ContainerDeleted)
    assert not any("propagated-mount" in command for kind, command in order if kind == "ssh")
    assert order[:2] == [("stop", payload.container_name), ("remove", payload.container_name)]


@pytest.mark.parametrize(
    "ssh_answers",
    [
        _host_answers(wipe=_make_ssh_command_result(exit_status=6, stdout="volume in use by x")),
        _host_answers(wipe=_make_ssh_command_result(exit_status=124)),
        OSError("ssh transport closed"),
        _hang_the_wipe,
    ],
    ids=["shared-volume", "host-timeout", "transport-error", "hang"],
)
@pytest.mark.asyncio
async def test_delete_proceeds_as_before_when_the_wipe_fails(
    docker_service, monkeypatch, ssh_answers
):
    # a wipe the host did not end in time is cut at its own deadline, on its own connection, so
    # the stop and the forced removal never wait for it or find the session taken
    monkeypatch.setattr(docker_service_module, "_VOLUME_WIPE_TIMEOUT_SEC", 0.05)
    monkeypatch.setattr(docker_service_module, "_OWN_SSH_CONNECTION_MARGIN_SEC", 0.05)
    monkeypatch.setattr(docker_service_module, "_HOST_TIMEOUT_VALIDATOR_MARGIN_SECONDS", 0.05)
    order = _wire_delete(docker_service, monkeypatch, ssh_answers)
    payload = _delete_payload()

    result = await asyncio.wait_for(_delete(docker_service, payload), timeout=5)

    assert isinstance(result, ContainerDeleted)
    assert order[0] == ("ssh", _bounded(WIPE_CMD, 0.05))
    # at most a wipe, the wait and the second wipe stand before the removal
    assert ("remove", payload.container_name) in order[:4]
