import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

MINER_JOBS_PATH = Path(__file__).resolve().parents[1] / "src" / "miner_jobs"
sys.path.insert(0, str(MINER_JOBS_PATH))

import backup_storage
from datura.requests.miner_requests import ExecutorSSHInfo
import restore_storage
import workspace_mount as workspace_mount_module
from workspace_mount import (
    VolumeAccess,
    detect_volume_access,
    normalize_volume_path,
    normalize_workspace_path,
    require_container_running,
)
from payload_models.payloads import BackupContainerRequest, ExternalVolumeInfo, RestoreContainerRequest
from services import miner_service as miner_service_module
from services.miner_service import MinerService


def _backup_args() -> SimpleNamespace:
    return SimpleNamespace(
        source_volume="source-volume",
        source_volume_path="/root",
        backup_volume_iam_user_access_key="access",
        backup_volume_iam_user_secret_key="secret",
        backup_volume_name="bucket",
        backup_target_path="target.tar.gz",
        backup_path="/root/test",
        backup_log_id="backup-log",
        api_url="https://api.example",
        auth_token="token",
    )


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


class _FakeStdout:
    def __iter__(self):
        return iter(["one\n", "two\n"])

    def close(self):
        return None


class _FakePopen:
    def __init__(self, command, *args, **kwargs):
        self.command = command
        self.stdout = _FakeStdout()
        self.returncode = 0

    def wait(self):
        return 0

    def communicate(self):
        return "", ""

    def kill(self):
        return None


def test_plain_backup_commands_use_workspace(monkeypatch):
    run_commands: list[list[str]] = []
    popen_commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        run_commands.append(command)
        return SimpleNamespace(returncode=0, stdout="1024 /workspace/test\n", stderr="")

    def fake_popen(command, *args, **kwargs):
        popen_commands.append(command)
        return _FakePopen(command, *args, **kwargs)

    monkeypatch.setattr(backup_storage.subprocess, "run", fake_run)
    monkeypatch.setattr(backup_storage.subprocess, "Popen", fake_popen)

    args = _backup_args()
    access = VolumeAccess(args.source_volume, args.source_volume_path)
    backup_path = access.normalized_path(args.backup_path)

    backup_storage.estimate_backup_sizes(args, access, backup_path)
    backup_storage.aws_cp(args, access, backup_path)

    flat_commands = [" ".join(command) for command in run_commands + popen_commands]
    tar_commands = [
        command for command in popen_commands if "--entrypoint" in command and command[command.index("--entrypoint") + 1] == "tar"
    ]
    assert backup_path == "/workspace/test"
    assert any("source-volume:/workspace" in command for command in flat_commands)
    assert any("--entrypoint find" in command and "/workspace/test -print" in command for command in flat_commands)
    assert any(command[command.index("-C") + 1] == "/workspace" and command[-1] == "test" for command in tar_commands)
    assert all("source-volume:/root" not in command for command in flat_commands)


def test_encrypted_backup_commands_exec_inside_rental(monkeypatch):
    run_commands: list[list[str]] = []
    popen_commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        run_commands.append(command)
        return SimpleNamespace(returncode=0, stdout="1024 /root/test\n", stderr="")

    def fake_popen(command, *args, **kwargs):
        popen_commands.append(command)
        return _FakePopen(command, *args, **kwargs)

    monkeypatch.setattr(backup_storage.subprocess, "run", fake_run)
    monkeypatch.setattr(backup_storage.subprocess, "Popen", fake_popen)

    args = _backup_args()
    access = VolumeAccess(
        args.source_volume,
        args.source_volume_path,
        encrypted=True,
        container_name="rental-pod",
    )
    backup_path = access.normalized_path(args.backup_path)

    backup_storage.estimate_backup_sizes(args, access, backup_path)
    backup_storage.aws_cp(args, access, backup_path)

    assert backup_path == "/root/test"
    assert any(command[:6] == ["/usr/bin/docker", "exec", "-u", "0", "rental-pod", "du"] for command in run_commands)
    assert any(command[:6] == ["/usr/bin/docker", "exec", "-u", "0", "rental-pod", "find"] for command in popen_commands)
    assert any(command[:6] == ["/usr/bin/docker", "exec", "-u", "0", "rental-pod", "tar"] for command in popen_commands)


def test_encrypted_restore_commands_exec_inside_rental(monkeypatch):
    run_commands: list[list[str]] = []
    popen_commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        run_commands.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def fake_popen(command, *args, **kwargs):
        popen_commands.append(command)
        return _FakePopen(command, *args, **kwargs)

    monkeypatch.setattr(restore_storage.subprocess, "run", fake_run)
    monkeypatch.setattr(restore_storage.subprocess, "Popen", fake_popen)

    args = _restore_args()
    access = VolumeAccess(
        args.target_volume,
        args.target_volume_path,
        encrypted=True,
        container_name="rental-pod",
    )
    restore_path = access.normalized_path(args.restore_path)

    restore_storage.ensure_restore_path(args, access, restore_path)
    restore_storage.aws_restore(args, access, restore_path)

    assert restore_path == "/root/test"
    assert any(command[:6] == ["/usr/bin/docker", "exec", "-u", "0", "rental-pod", "mkdir"] for command in run_commands)
    assert any(command[:7] == ["/usr/bin/docker", "exec", "-u", "0", "-i", "rental-pod", "tar"] for command in popen_commands)


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


def test_require_container_running_fails_when_stopped(monkeypatch):
    def fake_run(command, **kwargs):
        return SimpleNamespace(returncode=0, stdout="false\n", stderr="")

    monkeypatch.setattr("workspace_mount.subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="not running"):
        require_container_running("rental-pod")


def test_normalize_volume_path_keeps_root_paths():
    assert normalize_volume_path("/root/test", "/root") == "/root/test"
    assert normalize_volume_path("test", "/root") == "/root/test"


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


def test_detect_volume_access_plain_when_no_cipher_mount(monkeypatch):
    monkeypatch.setattr(
        "workspace_mount.subprocess.run",
        _docker_run_side_effect(
            ps_output="other-pod\n",
            inspect_outputs={"other-pod": "source-volume\t/root\n"},
            gocryptfs_conf_returncode=1,
        ),
    )

    access = detect_volume_access("source-volume", "/root")

    assert access.encrypted is False
    assert access.container_name is None


def test_detect_volume_access_encrypted_at_lium_cipher(monkeypatch):
    monkeypatch.setattr(
        "workspace_mount.subprocess.run",
        _docker_run_side_effect(
            ps_output="rental-pod\n",
            inspect_outputs={"rental-pod": "source-volume\t/lium-cipher\n"},
        ),
    )

    access = detect_volume_access("source-volume", "/root")

    assert access.encrypted is True
    assert access.container_name == "rental-pod"


def test_detect_volume_access_multiple_encrypted_raises(monkeypatch):
    monkeypatch.setattr(
        "workspace_mount.subprocess.run",
        _docker_run_side_effect(
            ps_output="pod-a\npod-b\n",
            inspect_outputs={
                "pod-a": "source-volume\t/lium-cipher\n",
                "pod-b": "source-volume\t/lium-cipher\n",
            },
        ),
    )

    with pytest.raises(RuntimeError, match="multiple running containers"):
        detect_volume_access("source-volume", "/root")


def test_detect_volume_access_encrypted_requires_running_container(monkeypatch):
    monkeypatch.setattr(
        "workspace_mount.subprocess.run",
        _docker_run_side_effect(
            ps_output="rental-pod\n",
            inspect_outputs={"rental-pod": "source-volume\t/lium-cipher\n"},
            running={"rental-pod": False},
        ),
    )

    with pytest.raises(RuntimeError, match="not running"):
        detect_volume_access("source-volume", "/root")


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


def test_detect_volume_access_plain_when_gocryptfs_conf_absent(monkeypatch):
    monkeypatch.setattr(
        "workspace_mount.subprocess.run",
        _docker_run_side_effect(
            ps_output="",
            ps_all_output="",
            inspect_outputs={},
            gocryptfs_conf_returncode=1,
        ),
    )

    access = detect_volume_access("source-volume", "/root")

    assert access.encrypted is False
    assert access.container_name is None


def test_detect_volume_access_raises_when_gocryptfs_conf_probe_fails(monkeypatch):
    monkeypatch.setattr(
        "workspace_mount.subprocess.run",
        _docker_run_side_effect(
            ps_output="",
            ps_all_output="",
            inspect_outputs={},
            gocryptfs_conf_returncode=125,
        ),
    )

    with pytest.raises(RuntimeError, match="Failed to probe gocryptfs.conf"):
        detect_volume_access("source-volume", "/root")


class _FakeSftpContext:
    def __init__(self):
        self.put = AsyncMock()

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


def _backup_payload() -> BackupContainerRequest:
    return BackupContainerRequest(
        miner_hotkey="miner",
        executor_id="executor",
        pod_id="pod",
        source_volume="source-volume",
        backup_volume_info=ExternalVolumeInfo(
            name="bucket",
            plugin="s3fs",
            iam_user_access_key="access",
            iam_user_secret_key="secret",
        ),
        backup_path="/root/test",
        source_volume_path="/root",
        backup_target_path="target.tar.gz",
        auth_token="token",
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
            iam_user_access_key="access",
            iam_user_secret_key="secret",
        ),
        backup_source_path="source.tar.gz",
        target_volume_path="/root",
        auth_token="token",
        restore_log_id="restore-log",
        restore_path="/root/test",
    )


@pytest.mark.asyncio
async def test_miner_service_backup_argv_excludes_encryption_flags(monkeypatch):
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(return_value=SimpleNamespace(exit_status=0, stdout="", stderr=""))
    sftp_context = _FakeSftpContext()
    ssh_client.start_sftp_client = Mock(return_value=sftp_context)
    monkeypatch.setattr(miner_service_module.asyncssh, "connect", Mock(return_value=_FakeSshContext(ssh_client)))

    service = MinerService.__new__(MinerService)
    await service.handle_backup_container_req(_executor_info(), _backup_payload(), Mock())

    nohup_call = ssh_client.run.await_args_list[-1]
    command = nohup_call.args[0]
    assert "--source-volume-encrypted" not in command
    assert "--container-name" not in command
    assert "--source-volume source-volume" in command


@pytest.mark.asyncio
async def test_miner_service_restore_argv_excludes_encryption_flags(monkeypatch):
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(return_value=SimpleNamespace(exit_status=0, stdout="", stderr=""))
    sftp_context = _FakeSftpContext()
    ssh_client.start_sftp_client = Mock(return_value=sftp_context)
    monkeypatch.setattr(miner_service_module.asyncssh, "connect", Mock(return_value=_FakeSshContext(ssh_client)))

    service = MinerService.__new__(MinerService)
    await service.handle_restore_container_req(_executor_info(), _restore_payload(), Mock())

    nohup_call = ssh_client.run.await_args_list[-1]
    command = nohup_call.args[0]
    assert "--target-volume-encrypted" not in command
    assert "--container-name" not in command
    assert "--target-volume target-volume" in command


def test_restic_backup_empty_path_targets_the_whole_volume():
    payload = _backup_payload()
    payload.backup_engine = "restic"
    payload.repository_password = "repository-password"
    payload.backup_path = ""

    spec = MinerService._restic_backup_operation_spec(payload)

    assert spec["workspace"]["requested_path"] == payload.source_volume_path


def test_restic_online_restore_empty_path_targets_the_whole_volume():
    payload = _restore_payload()
    payload.backup_engine = "restic"
    payload.repository_password = "repository-password"
    payload.snapshot_id = "a" * 64
    payload.restore_path = ""

    spec = MinerService._restic_restore_operation_spec(payload)

    assert spec["workspace"]["requested_path"] == payload.target_volume_path
