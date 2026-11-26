"""DAH-5101: a delete does not wait for the pod's unwritten volume data to reach a slow disk.
The volume's files are unlinked on the host before the stop, so the kernel drops their unwritten
pages at the last close instead of flushing them at the unmount (Kosta85's node, 8c7d58d6-2ac4-4d77-a9b0-d86f5559782b).
"""

# ruff: noqa: F811 — the docker_service fixture is imported from test_docker_service
from __future__ import annotations

import asyncio
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from datura.requests.miner_requests import ExecutorSSHInfo
from payload_models.payloads import (
    ContainerDeleted,
    ContainerDeleteRequest,
    FailedContainerErrorCodes,
    FailedContainerRequest,
    WorkloadKind,
)

import services.docker_service as docker_service_module
from services.storage_operations import REMOTE_OPERATION_DIRECTORY, StorageVolumeBeingWiped, storage_dispatch_holds_volume
from tests.test_storage_volume_lock import VolumeLockRedis
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
    lock_redis = VolumeLockRedis()
    for call in ("add_volume_hold", "remove_volume_hold"):
        setattr(docker_service.redis_service, call, getattr(lock_redis, call))
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
    operation_volumes: tuple[str, ...] = (),
    volume_markers: tuple[str, ...] = (),
) -> subprocess.CompletedProcess:
    """Runs the wipe command under sh against a docker stub whose answers come from the arguments;
    `users` is what `docker ps` lists on the volume, `fail` names the stub call that exits nonzero
    (inspect, ps, info, plugin). Returns the process."""
    host_root = tmp_path / "host"
    operations = tmp_path / "storage-operations"
    operations.mkdir(exist_ok=True)
    for i, name in enumerate(operation_volumes):
        spec = {"action": "backup", "workspace": {"volume_name": name}}
        (operations / f"op{i}.json").write_text(json.dumps(spec, separators=(",", ":"), sort_keys=True))
    for i, name in enumerate(volume_markers):
        # a running operation: the runner has stripped its spec to the reporter, the marker stays
        (operations / f"run{i}.json").write_text(json.dumps({"reporter": {"resource": "backup"}}))
        (operations / f"run{i}.volume").write_text(f"{name}\n")
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
        .replace(str(REMOTE_OPERATION_DIRECTORY), str(operations))
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


@pytest.mark.parametrize("driver, mountpoint", [("local", _LOCAL_MOUNT), ("vloopback:v3", "/mnt/volume_dirty")])
def test_wipe_leaves_a_volume_whose_backup_is_starting(tmp_path, driver, mountpoint):
    # DAH-7077: the backup's spec names the volume before its helper mounts it
    wiped_dir = _LOCAL_MOUNT if driver == "local" else _VLOOPBACK_HOST_DIR
    files = _host_files(tmp_path, wiped_dir)

    completed = _run_wipe(tmp_path, "volume_dirty", driver, mountpoint, operation_volumes=("other", "volume_dirty"))

    assert completed.returncode == 7
    assert completed.stdout.strip() == "volume has a storage operation"
    assert files[wiped_dir].exists()


def test_wipe_leaves_a_volume_whose_running_operation_stripped_its_spec(tmp_path):
    # Pixel's review: the runner rewrites the spec without the workspace once it has read it
    files = _host_files(tmp_path, _LOCAL_MOUNT)

    completed = _run_wipe(tmp_path, "volume_dirty", "local", _LOCAL_MOUNT, volume_markers=("volume_dirty",))

    assert completed.returncode == 7
    assert files[_LOCAL_MOUNT].exists()


@pytest.mark.parametrize("flag", ["--source-volume", "--target-volume"])
def test_wipe_leaves_a_volume_a_legacy_tar_job_runs_on(tmp_path, flag):
    files = _host_files(tmp_path, _LOCAL_MOUNT)
    job = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)", flag, "volume_dirty", "--backup-path", "/x"]
    )
    try:
        completed = _run_wipe(tmp_path, "volume_dirty", "local", _LOCAL_MOUNT)
    finally:
        job.kill()
        job.wait()

    assert completed.returncode == 7
    assert files[_LOCAL_MOUNT].exists()


def test_wipe_runs_when_only_another_volume_has_a_marker_or_legacy_job(tmp_path):
    files = _host_files(tmp_path, _LOCAL_MOUNT)
    job = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)", "--source-volume", "volume_dirty_2", "--backup-path", "/x"]
    )
    try:
        completed = _run_wipe(tmp_path, "volume_dirty", "local", _LOCAL_MOUNT, volume_markers=("volume_dirty_2",))
    finally:
        job.kill()
        job.wait()

    assert completed.returncode == 0, completed.stderr
    assert not files[_LOCAL_MOUNT].exists()


def test_wipe_runs_when_only_another_volume_has_a_storage_operation(tmp_path):
    files = _host_files(tmp_path, _LOCAL_MOUNT)

    completed = _run_wipe(tmp_path, "volume_dirty", "local", _LOCAL_MOUNT, operation_volumes=("volume_dirty_2",))

    assert completed.returncode == 0, completed.stderr
    assert not files[_LOCAL_MOUNT].exists()


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


@pytest.mark.asyncio
async def test_delete_removes_the_pods_kept_edit_containers_before_its_volume(docker_service, monkeypatch):
    # DAH-5773 review: a kept container still references the pod's volume, so `volume rm` failed as "in use"
    # and the renter's layer and the volume stayed on the host after the rental ended
    async def answer(command: str):
        if "docker ps -a" in command and "kept_unmounted_dirty_" in command:
            # another pod's kept container is never touched, even if a listing carried it
            return _make_ssh_command_result(stdout="kept_unmounted_dirty_1700 idkept\nkept_unmounted_dirtyx_1 idother\n")
        return await _host_answers()(command)

    order = _wire_delete(docker_service, monkeypatch, answer)
    client = docker_service.rental_docker_client_factory.client
    real_remove_volume = client.remove_volume

    async def remove_volume(*, volume_name, force=False):
        order.append(("remove_volume", volume_name))
        return await real_remove_volume(volume_name=volume_name, force=force)

    monkeypatch.setattr(client, "remove_volume", remove_volume)

    result = await _delete(docker_service, _delete_payload())

    assert isinstance(result, ContainerDeleted)
    rm_kept = [i for i, (kind, command) in enumerate(order) if kind == "ssh" and "docker rm -fv" in command]
    assert len(rm_kept) == 1
    assert "idkept" in order[rm_kept[0]][1] and "idother" not in order[rm_kept[0]][1]
    assert rm_kept[0] < order.index(("remove_volume", "volume_dirty"))


@pytest.mark.asyncio
async def test_delete_answers_in_progress_until_the_kept_edit_containers_are_removed(docker_service, monkeypatch):
    # DAH-5773 review: nothing else removes a kept container, so a failed removal must not close the pod
    # the first ask's rm fails before and after the direct kill
    rm_exit = iter([1, 1, 0])

    async def answer(command: str):
        if "docker ps -a" in command and "kept_unmounted_dirty_" in command:
            return _make_ssh_command_result(stdout="kept_unmounted_dirty_1700 idkept\n")
        if "docker rm -fv" in command:
            return _make_ssh_command_result(exit_status=next(rm_exit), stderr="device or resource busy")
        return await _host_answers()(command)

    order = _wire_delete(docker_service, monkeypatch, answer)
    client = docker_service.rental_docker_client_factory.client
    real_remove_volume = client.remove_volume

    async def remove_volume(*, volume_name, force=False):
        order.append(("remove_volume", volume_name))
        return await real_remove_volume(volume_name=volume_name, force=force)

    monkeypatch.setattr(client, "remove_volume", remove_volume)
    payload = _delete_payload()

    first = await _delete(docker_service, payload)

    assert isinstance(first, FailedContainerRequest)
    assert first.error_code == FailedContainerErrorCodes.DeletionInProgress
    assert ("remove_volume", "volume_dirty") not in order

    retried = await _delete(docker_service, payload)

    assert isinstance(retried, ContainerDeleted)
    assert ("remove_volume", "volume_dirty") in order


# DAH-7077: the volume lock between a storage dispatch and the wipe
@pytest.mark.asyncio
async def test_delete_skips_the_wipe_while_a_backup_dispatch_holds_the_volume(docker_service, monkeypatch):
    order = _wire_delete(docker_service, monkeypatch, _host_answers())
    lock_redis = docker_service.redis_service.add_volume_hold.__self__
    payload = _delete_payload()

    async with storage_dispatch_holds_volume(lock_redis, payload.local_volume, uuid4()):
        result = await _delete(docker_service, payload)

    assert isinstance(result, ContainerDeleted)
    assert not any(step == ("ssh", _bounded(WIPE_CMD, _VOLUME_WIPE_TIMEOUT_SEC)) for step in order)
    assert ("remove", payload.container_name) in order


@pytest.mark.parametrize(
    "ssh_answers, dispatch_refused",
    [
        (_host_answers(), True),
        (_host_answers(wipe=_make_ssh_command_result(exit_status=7, stdout="volume has a storage operation")), False),
    ],
    ids=["wiped", "wipe-skipped-on-the-node"],
)
@pytest.mark.asyncio
async def test_a_wipe_that_ran_keeps_the_volume_from_any_later_backup(
    docker_service, monkeypatch, ssh_answers, dispatch_refused
):
    _wire_delete(docker_service, monkeypatch, ssh_answers)
    lock_redis = docker_service.redis_service.add_volume_hold.__self__
    payload = _delete_payload()

    await _delete(docker_service, payload)

    if dispatch_refused:
        with pytest.raises(StorageVolumeBeingWiped):
            async with storage_dispatch_holds_volume(lock_redis, payload.local_volume, uuid4()):
                pass
    else:
        async with storage_dispatch_holds_volume(lock_redis, payload.local_volume, uuid4()):
            pass


@pytest.mark.asyncio
async def test_a_failed_lock_release_never_fails_the_delete(docker_service, monkeypatch):
    # Pixel's review: Redis failing on the release of a skipped wipe must leave the normal stop path
    order = _wire_delete(
        docker_service,
        monkeypatch,
        _host_answers(wipe=_make_ssh_command_result(exit_status=7, stdout="volume has a storage operation")),
    )

    async def broken_delete(*args, **kwargs):
        raise ConnectionError("redis down")

    monkeypatch.setattr(docker_service.redis_service, "remove_volume_hold", broken_delete)
    payload = _delete_payload()

    result = await _delete(docker_service, payload)

    assert isinstance(result, ContainerDeleted)
    assert ("remove", payload.container_name) in order


@pytest.mark.asyncio
async def test_delete_kills_a_kept_container_whose_exit_event_was_lost_and_removes_it(docker_service, monkeypatch):
    # DAH-7025 review: a delete that cut off the edit's direct kill finds the parked container wedged ("did not
    # receive an exit event"); rm -f alone fails on every re-ask, so the delete kills it directly and removes it
    killed = []

    async def answer(command: str):
        if "docker ps -a" in command and "kept_unmounted_dirty_" in command:
            return _make_ssh_command_result(stdout="kept_unmounted_dirty_1700 idkept\n")
        if "kill -9" in command and "idkept" in command:
            killed.append(command)
            return _make_ssh_command_result(stdout="killed pid=7 shim=8 init=gone\n")
        if "docker rm -fv" in command and not killed:
            return _make_ssh_command_result(
                exit_status=1, stderr="tried to kill container, but did not receive an exit event"
            )
        return await _host_answers()(command)

    _wire_delete(docker_service, monkeypatch, answer)

    result = await _delete(docker_service, _delete_payload())

    assert isinstance(result, ContainerDeleted)
    assert len(killed) == 1


@pytest.mark.asyncio
async def test_delete_answers_in_progress_when_the_rm_after_the_direct_kill_does_not_answer(docker_service, monkeypatch):
    # DAH-7025 review: the first rm fails, the direct kill completes, the retry rm times out: the kept container is
    # still there, so the delete must not close the pod
    killed = []

    async def answer(command: str):
        if "docker ps -a" in command and "kept_unmounted_dirty_" in command:
            return _make_ssh_command_result(stdout="kept_unmounted_dirty_1700 idkept\n")
        if "kill -9" in command and "idkept" in command:
            killed.append(command)
            return _make_ssh_command_result(stdout="killed pid=7 shim=8 init=gone\n")
        if "docker rm -fv" in command:
            if killed:
                raise TimeoutError
            return _make_ssh_command_result(exit_status=1, stderr="did not receive an exit event")
        return await _host_answers()(command)

    _wire_delete(docker_service, monkeypatch, answer)

    result = await _delete(docker_service, _delete_payload())

    assert isinstance(result, FailedContainerRequest)
    assert result.error_code == FailedContainerErrorCodes.DeletionInProgress
    assert len(killed) == 1
