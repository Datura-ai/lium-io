import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

MINER_JOBS_PATH = Path(__file__).resolve().parents[1] / "src" / "miner_jobs"
sys.path.insert(0, str(MINER_JOBS_PATH))

import backup_storage
import restore_storage
import workspace_mount as workspace_mount_module
from datura.requests.miner_requests import ExecutorSSHInfo
from payload_models.payloads import (
    BackupContainerRequest,
    CancelStorageOperationRequest,
    ExternalVolumeInfo,
    RestoreContainerRequest,
)
from services.docker_service import DockerService
from services.miner_service import MinerService
from workspace_mount import (
    VolumeAccess,
    detect_volume_access,
    normalize_workspace_path,
)

from services import docker_service as docker_service_module
from services import miner_service as miner_service_module


def _restore_args() -> SimpleNamespace:
    return SimpleNamespace(
        target_volume="target-volume",
        target_volume_path="/root",
        backup_volume_iam_user_access_key="access",
        backup_volume_iam_user_secret_key="secret",
        backup_volume_name="bucket",
        backup_source_path="backup.tar.gz",
        restore_path="/root/test",
        restore_log_id="restore-log",
        api_url="https://api.example",
        auth_token="token",
    )


def test_encrypted_restore_hands_the_target_dir_back_to_the_renter(monkeypatch):
    # mkdir runs as uid 0 and tar --strip-components=1 drops the archive's top
    # entry, so without an explicit chown the renter cannot write there
    run_commands: list[list[str]] = []

    monkeypatch.setattr(
        restore_storage,
        "run_command_args",
        lambda command, command_label=None: run_commands.append(command),
    )
    monkeypatch.setattr(
        workspace_mount_module.subprocess,
        "run",
        lambda command, **_: SimpleNamespace(returncode=0, stdout="renter\n", stderr=""),
    )
    access = VolumeAccess("vol", "/root", encrypted=True, container_name="rental-pod")

    restore_storage.ensure_restore_path(_restore_args(), access, "/root/restored")

    chown_commands = [command for command in run_commands if "chown" in command]
    assert chown_commands, f"no chown issued: {run_commands}"
    assert chown_commands[0][-2:] == ["renter", "/root/restored"]


def test_unreadable_image_user_fails_the_restore_instead_of_skipping_chown(monkeypatch):
    # returning None here would look exactly like a root image and silently skip
    # the hand-back, leaving the renter with an unwritable restore target
    monkeypatch.setattr(
        workspace_mount_module.subprocess,
        "run",
        lambda command, **_: SimpleNamespace(returncode=1, stdout="", stderr="no such container"),
    )
    access = VolumeAccess("vol", "/root", encrypted=True, container_name="rental-pod")

    with pytest.raises(RuntimeError, match="Could not read the image USER"):
        access.container_image_user()


def test_workspace_path_normalization_rejects_escape():
    assert normalize_workspace_path("/root/test", "/root") == "/workspace/test"
    assert normalize_workspace_path("test", "/root") == "/workspace/test"
    assert normalize_workspace_path("/workspace/test", "/root") == "/workspace/test"
    with pytest.raises(ValueError, match="escapes"):
        normalize_workspace_path("/root/../etc", "/root")


def _docker_run_side_effect(
    ps_output: str,
    inspect_outputs: dict[str, str],
    running: dict[str, bool] | None = None,
    ps_all_output: str | None = None,
    gocryptfs_conf_returncode: int = 1,
):
    running = running or {}
    if ps_all_output is None:
        ps_all_output = ps_output

    def fake_run(command, **kwargs):
        if command[:2] == ["/usr/bin/docker", "ps"]:
            if "-a" in command:
                return SimpleNamespace(returncode=0, stdout=ps_all_output, stderr="")
            return SimpleNamespace(returncode=0, stdout=ps_output, stderr="")
        if command[:3] == ["/usr/bin/docker", "inspect", "-f"]:
            if command[3] == "{{.State.Running}}":
                container_name = command[-1]
                is_running = running.get(container_name, True)
                return SimpleNamespace(returncode=0, stdout="true\n" if is_running else "false\n", stderr="")
            container_name = command[-1]
            return SimpleNamespace(
                returncode=0,
                stdout=inspect_outputs.get(container_name, ""),
                stderr="",
            )
        if (
            command[:3] == ["/usr/bin/docker", "run", "--rm"]
            and "--entrypoint" in command
            and command[command.index("--entrypoint") + 1] == "test"
            and command[-2:] == ["-f", "/probe/gocryptfs.conf"]
        ):
            return SimpleNamespace(returncode=gocryptfs_conf_returncode, stdout="", stderr="")
        raise AssertionError(f"unexpected docker command: {command}")

    return fake_run


def test_detect_volume_access_raises_when_stopped_container_has_cipher_mount(monkeypatch):
    monkeypatch.setattr(
        "workspace_mount.subprocess.run",
        _docker_run_side_effect(
            ps_output="",
            ps_all_output="stopped-pod\n",
            inspect_outputs={"stopped-pod": "source-volume\t/lium-cipher\n"},
        ),
    )

    with pytest.raises(RuntimeError, match="encrypted.*no running container.*/lium-cipher"):
        detect_volume_access("source-volume", "/root")


def test_detect_volume_access_raises_when_gocryptfs_conf_present(monkeypatch):
    monkeypatch.setattr(
        "workspace_mount.subprocess.run",
        _docker_run_side_effect(
            ps_output="",
            ps_all_output="",
            inspect_outputs={},
            gocryptfs_conf_returncode=0,
        ),
    )

    with pytest.raises(RuntimeError, match="gocryptfs.conf.*no running container.*/lium-cipher"):
        detect_volume_access("source-volume", "/root")


class _FakeRemoteFile:
    def __init__(self, files, path):
        self.files, self.path = files, path

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def write(self, data):
        self.files[self.path] = data


class _FakeSftpContext:
    def __init__(self):
        self.put = AsyncMock()
        self.files: dict[str, str] = {}
        self.opens: list[tuple] = []

    def open(self, path, mode, attrs=None):
        self.opens.append((path, mode, attrs))
        return _FakeRemoteFile(self.files, path)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None


class _FakeSshContext:
    def __init__(self, ssh_client):
        self.ssh_client = ssh_client

    async def __aenter__(self):
        return self.ssh_client

    async def __aexit__(self, exc_type, exc, tb):
        return None


def _executor_info() -> ExecutorSSHInfo:
    return ExecutorSSHInfo(
        uuid="executor",
        address="127.0.0.1",
        port=8080,
        ssh_username="root",
        ssh_port=22,
        port_mappings="[]",
        port_range="40000-40100",
        python_path="/usr/bin/python3",
        root_dir="/root",
    )


ACCESS_KEY = "access-key-value"
SECRET_KEY = "secret-key-value"
AUTH_TOKEN = "auth-token-value"
KNOWN_HOSTS = object()


def _docker_service_with_pinned_hosts() -> Mock:
    docker_service = Mock()
    docker_service._prepare_known_hosts_policy = AsyncMock(return_value=KNOWN_HOSTS)
    return docker_service


def _assert_secrets_only_in_the_0600_file(ssh_client, sftp_context) -> None:
    commands = " ".join(str(call.args[0]) for call in ssh_client.run.await_args_list)
    for value in (ACCESS_KEY, SECRET_KEY, AUTH_TOKEN):
        assert value not in commands
    ((path, mode, attrs),) = sftp_context.opens
    assert mode == "x" and attrs.permissions == 0o600
    assert f"--secrets-file {path}" in commands
    assert json.loads(sftp_context.files[path]) == {
        "auth_token": AUTH_TOKEN,
        "aws_access_key_id": ACCESS_KEY,
        "aws_secret_access_key": SECRET_KEY,
    }


def _backup_payload() -> BackupContainerRequest:
    return BackupContainerRequest(
        miner_hotkey="miner",
        executor_id="executor",
        pod_id="pod",
        source_volume="source-volume",
        backup_volume_info=ExternalVolumeInfo(
            name="bucket",
            plugin="s3fs",
            iam_user_access_key=ACCESS_KEY,
            iam_user_secret_key=SECRET_KEY,
        ),
        backup_path="/root/test",
        source_volume_path="/root",
        backup_target_path="target.tar.gz",
        auth_token=AUTH_TOKEN,
        backup_log_id="backup-log",
    )


def _restore_payload() -> RestoreContainerRequest:
    return RestoreContainerRequest(
        miner_hotkey="miner",
        executor_id="executor",
        pod_id="pod",
        target_volume="target-volume",
        backup_volume_info=ExternalVolumeInfo(
            name="bucket",
            plugin="s3fs",
            iam_user_access_key=ACCESS_KEY,
            iam_user_secret_key=SECRET_KEY,
        ),
        backup_source_path="source.tar.gz",
        target_volume_path="/root",
        auth_token=AUTH_TOKEN,
        restore_log_id="restore-log",
        restore_path="/root/test",
    )


@pytest.mark.asyncio
async def test_cancel_storage_operation_pins_the_host_key(monkeypatch):
    connect = Mock(return_value=_FakeSshContext(AsyncMock()))
    monkeypatch.setattr(miner_service_module.asyncssh, "connect", connect)
    monkeypatch.setattr(miner_service_module, "cancel_storage_operation", AsyncMock())
    payload = CancelStorageOperationRequest(
        miner_hotkey="miner",
        executor_id="executor",
        pod_id="pod",
        operation_id="11111111-1111-4111-8111-111111111111",
    )

    await MinerService.__new__(MinerService).handle_cancel_storage_operation_req(
        _executor_info(), payload, Mock(), _docker_service_with_pinned_hosts()
    )

    assert connect.call_args.kwargs["known_hosts"] is KNOWN_HOSTS


@pytest.mark.asyncio
async def test_legacy_bootstrap_restore_keeps_secrets_off_the_command_line():
    ssh_client = AsyncMock()
    sftp_context = _FakeSftpContext()
    ssh_client.start_sftp_client = Mock(return_value=sftp_context)
    payload = _restore_payload()
    restore = SimpleNamespace(
        restore_path="/root/test",
        legacy_object_key="source.tar.gz",
        auth_token=AUTH_TOKEN,
        restore_log_id="restore-log",
        backup_volume_info=payload.backup_volume_info,
    )

    await DockerService.__new__(DockerService)._run_legacy_bootstrap_restore(
        ssh_client=ssh_client,
        executor_info=_executor_info(),
        restore=restore,
        local_volume="volume",
        local_volume_path="/root",
    )

    _assert_secrets_only_in_the_0600_file(ssh_client, sftp_context)


def test_scripts_pass_s3_keys_to_docker_by_name_only():
    for script in (backup_storage, restore_storage):
        command = script.docker_base_command(SimpleNamespace())
        assert command[command.index("AWS_SECRET_ACCESS_KEY") - 1] == "-e"
        assert command[command.index("AWS_ACCESS_KEY_ID") - 1] == "-e"
        assert not any(item.startswith(("AWS_SECRET_ACCESS_KEY=", "AWS_ACCESS_KEY_ID=")) for item in command)


def test_load_secrets_file_exports_keys_and_deletes_the_file(tmp_path, monkeypatch):
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    path = tmp_path / "secrets.json"
    path.write_text(
        json.dumps({"auth_token": AUTH_TOKEN, "aws_access_key_id": ACCESS_KEY, "aws_secret_access_key": SECRET_KEY})
    )

    assert workspace_mount_module.load_secrets_file(str(path)) == AUTH_TOKEN

    assert not path.exists()
    assert os.environ["AWS_ACCESS_KEY_ID"] == ACCESS_KEY
    assert os.environ["AWS_SECRET_ACCESS_KEY"] == SECRET_KEY


@pytest.mark.asyncio
async def test_restic_backup_with_excludes_stops_on_an_executor_that_cannot_apply_them(monkeypatch):
    ssh_client = AsyncMock()
    monkeypatch.setattr(miner_service_module.asyncssh, "connect", Mock(return_value=_FakeSshContext(ssh_client)))
    monkeypatch.setattr(miner_service_module, "supports_workspace_field", AsyncMock(return_value=False))
    start = AsyncMock()
    monkeypatch.setattr(miner_service_module, "start_storage_operation", start)
    payload = _backup_payload()
    payload.backup_engine = "restic"
    payload.backup_log_id = "11111111-1111-4111-8111-111111111111"
    payload.exclude_patterns = [".ssh"]

    with pytest.raises(RuntimeError, match="exclude_patterns"):
        await MinerService.__new__(MinerService).handle_backup_container_req(
            _executor_info(), payload, Mock(), _docker_service_with_pinned_hosts()
        )
    start.assert_not_awaited()


def test_restic_backup_spec_carries_the_exclude_patterns():
    payload = _backup_payload()
    payload.backup_engine = "restic"
    payload.repository_password = "repository-password"
    payload.exclude_patterns = [".cache", ".ssh"]

    spec = MinerService._restic_backup_operation_spec(payload)

    assert spec["workspace"]["exclude_patterns"] == [".cache", ".ssh"]


def _walk_json(value, path=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _walk_json(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_json(item, f"{path}[{index}]")
    else:
        yield path, value


@pytest.mark.asyncio
async def test_encrypted_bootstrap_restore_spec_never_carries_the_passphrase(monkeypatch):
    # DAH-3274: the operation spec is SFTP'd into the executor container = a file on the
    # provider's disk. It must reach the data through the running pod, never unlock it.
    import json

    from services.volume_keys import derive_volume_passphrase

    master_secret = "test-master-secret-32-chars-long!!"
    monkeypatch.setattr(docker_service_module.settings, "VOLUME_MASTER_SECRET", master_secret)
    monkeypatch.setattr(docker_service_module, "supports_storage_operation", AsyncMock(return_value=True))
    start = AsyncMock(return_value={"spec": "/tmp/spec.json"})
    monkeypatch.setattr(docker_service_module, "start_storage_operation", start)
    monkeypatch.setattr(docker_service_module, "wait_for_storage_operation", AsyncMock())
    service = DockerService.__new__(DockerService)
    restore = SimpleNamespace(
        backup_engine="restic",
        restore_log_id="33333333-3333-4333-8333-333333333333",
        restore_path="",
        backup_volume_info=SimpleNamespace(
            name="bucket",
            iam_user_access_key="ak",
            iam_user_secret_key="sk",
            session_token=None,
        ),
        repository_password="repo-password",
        repository_pod_id="pod-1",
        snapshot_id="a" * 64,
        legacy_object_key=None,
        legacy_object_size_bytes=None,
        auth_token="token",
        failure_timeout_seconds=600,
    )

    await service._run_bootstrap_restore(
        ssh_client=AsyncMock(),
        executor_info=_executor_info(),
        payload=SimpleNamespace(pod_id="pod-1"),
        restore=restore,
        local_volume="volume_pod-1",
        local_volume_path="/root",
        encrypted=True,
        container_name="pod_pod-1",
    )

    spec = start.await_args.args[3]
    passphrase = derive_volume_passphrase(master_secret, "pod-1")
    assert passphrase not in json.dumps(spec)
    assert not [path for path, _ in _walk_json(spec) if "passphrase" in path.lower()]
    assert spec["workspace"]["mode"] == "encrypted_running"
    assert spec["workspace"]["container_name"] == "pod_pod-1"
    assert spec["workspace"]["requested_path"] == "/root"
    # create-time: the entrypoint may have touched the fresh mount already; the backup wins
    assert spec["workspace"]["bootstrap"] is True


