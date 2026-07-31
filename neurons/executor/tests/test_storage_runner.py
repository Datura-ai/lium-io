from __future__ import annotations

import io
import json
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from uuid import UUID

import pytest

from storage.models import OperationSpecError, OperationResultQuality, StorageOperationSpec
from storage.restic import JsonEventWriter, ResticStorageRunner
from storage.workspace import DockerVolumeWorkspace, LocalWorkspace, WorkspaceResolutionError, WorkspaceResolver


OPERATION_ID = UUID("11111111-1111-4111-8111-111111111111")
POD_ID = UUID("22222222-2222-4222-8222-222222222222")
SNAPSHOT_ID = "a" * 64
CONTAINER_ID = "b" * 64


def _operation_payload(
    *,
    action: str = "backup",
    mode: str = "plain_volume",
    requested_path: str = "/root/checkpoints",
) -> dict[str, object]:
    workspace: dict[str, object] = {
        "mode": mode,
        "volume_name": "customer-volume",
        "volume_path": "/root",
        "requested_path": requested_path,
    }
    if mode == "encrypted_running":
        workspace["container_name"] = "rental-pod"
    return {
        "operation_id": str(OPERATION_ID),
        "pod_id": str(POD_ID),
        "action": action,
        "snapshot_id": SNAPSHOT_ID if action == "restore" else None,
        "progress_interval_seconds": 0,
        "repository": {
            "bucket": "backup-bucket",
            "access_key_id": "access-key",
            "secret_access_key": "secret-key",
            "session_token": "session-token",
            "password": "repository-password",
        },
        "workspace": workspace,
    }


def test_operation_derives_repository_from_pod() -> None:
    operation = StorageOperationSpec.from_mapping(_operation_payload())

    assert operation.repository.url_for_pod(operation.pod_id) == (
        f"s3:s3.amazonaws.com/backup-bucket/restic/v1/pods/{POD_ID}"
    )


def test_restore_requires_snapshot_id() -> None:
    payload = _operation_payload(action="restore")
    payload["snapshot_id"] = None

    with pytest.raises(OperationSpecError, match="snapshot_id is required"):
        StorageOperationSpec.from_mapping(payload)


def test_requested_path_must_stay_inside_volume() -> None:
    operation = StorageOperationSpec.from_mapping(_operation_payload(requested_path="/etc"))

    with pytest.raises(WorkspaceResolutionError, match="outside volume path"):
        WorkspaceResolver(environ={"LIUM_STORAGE_HELPER_IMAGE": "executor:test"}).resolve(operation)


def test_plain_backup_uses_read_only_volume_and_keeps_secrets_out_of_arguments() -> None:
    operation = StorageOperationSpec.from_mapping(_operation_payload())
    workspace = WorkspaceResolver(environ={"LIUM_STORAGE_HELPER_IMAGE": "executor:test"}).resolve(operation)
    runner = ResticStorageRunner(operation, workspace)

    command, cwd = runner._execution_command(["backup", "--json", "."], working_directory=True)

    assert cwd is None
    assert "customer-volume:/workspace:ro" in command
    assert command[command.index("--workdir") + 1] == "/workspace/checkpoints"
    assert command[command.index("--log-driver") + 1] == "none"
    assert "AWS_ACCESS_KEY_ID" in command
    assert "AWS_SESSION_TOKEN" in command
    assert "access-key" not in command
    assert "secret-key" not in command
    assert "repository-password" not in command
    assert "session-token" not in command
    assert runner._environment["AWS_SESSION_TOKEN"] == "session-token"


def test_plain_restore_rejects_nonempty_target(monkeypatch: pytest.MonkeyPatch) -> None:
    operation = StorageOperationSpec.from_mapping(_operation_payload(action="restore"))

    monkeypatch.setattr(
        "storage.workspace.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=21, stdout="", stderr=""),
    )

    with pytest.raises(WorkspaceResolutionError, match="new or empty"):
        WorkspaceResolver(environ={"LIUM_STORAGE_HELPER_IMAGE": "executor:test"}).resolve(operation)


def test_encrypted_workspace_resolves_verified_plaintext_view(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    operation = StorageOperationSpec.from_mapping(_operation_payload(mode="encrypted_running"))
    process_root = tmp_path / "4321"
    plaintext = process_root / "root" / "root" / "checkpoints"
    plaintext.mkdir(parents=True)
    (process_root / "cgroup").write_text(f"0::/docker/{CONTAINER_ID}\n")
    (process_root / "mountinfo").write_text(
        "36 25 0:32 / /root rw,nosuid,nodev - fuse.gocryptfs gocryptfs rw,user_id=0\n"
    )
    inspection = json.dumps(
        [
            {
                "Id": CONTAINER_ID,
                "State": {"Running": True, "Pid": 4321},
                "Mounts": [{"Name": "customer-volume", "Destination": "/lium-cipher"}],
            }
        ]
    )

    monkeypatch.setattr(
        "storage.workspace.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=inspection, stderr=""),
    )

    workspace = WorkspaceResolver(docker_binary="docker", proc_root=tmp_path, environ={}).resolve(operation)

    assert workspace == LocalWorkspace(plaintext)


def test_encrypted_workspace_fails_closed_without_gocryptfs_mount(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    operation = StorageOperationSpec.from_mapping(_operation_payload(mode="encrypted_running"))
    process_root = tmp_path / "4321"
    (process_root / "root" / "root" / "checkpoints").mkdir(parents=True)
    (process_root / "cgroup").write_text(f"0::/docker/{CONTAINER_ID}\n")
    (process_root / "mountinfo").write_text("36 25 0:32 / /root rw - ext4 /dev/sda rw\n")
    inspection = json.dumps(
        [
            {
                "Id": CONTAINER_ID,
                "State": {"Running": True, "Pid": 4321},
                "Mounts": [{"Name": "customer-volume", "Destination": "/lium-cipher"}],
            }
        ]
    )

    monkeypatch.setattr(
        "storage.workspace.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=inspection, stderr=""),
    )

    with pytest.raises(WorkspaceResolutionError, match="not a live fuse.gocryptfs mount"):
        WorkspaceResolver(docker_binary="docker", proc_root=tmp_path, environ={}).resolve(operation)


def test_encrypted_workspace_fails_closed_when_pid_cgroup_does_not_match(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    operation = StorageOperationSpec.from_mapping(_operation_payload(mode="encrypted_running"))
    process_root = tmp_path / "4321"
    (process_root / "root" / "root" / "checkpoints").mkdir(parents=True)
    (process_root / "cgroup").write_text("0::/docker/not-the-rental\n")
    (process_root / "mountinfo").write_text(
        "36 25 0:32 / /root rw,nosuid,nodev - fuse.gocryptfs gocryptfs rw,user_id=0\n"
    )
    inspection = json.dumps(
        [
            {
                "Id": CONTAINER_ID,
                "State": {"Running": True, "Pid": 4321},
                "Mounts": [{"Name": "customer-volume", "Destination": "/lium-cipher"}],
            }
        ]
    )
    monkeypatch.setattr(
        "storage.workspace.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=inspection, stderr=""),
    )

    with pytest.raises(WorkspaceResolutionError, match="PID cgroup does not match"):
        WorkspaceResolver(docker_binary="docker", proc_root=tmp_path, environ={}).resolve(operation)


def test_encrypted_workspace_fails_closed_when_container_restarts_during_resolution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    operation = StorageOperationSpec.from_mapping(_operation_payload(mode="encrypted_running"))
    process_root = tmp_path / "4321"
    plaintext = process_root / "root" / "root" / "checkpoints"
    plaintext.mkdir(parents=True)
    (process_root / "cgroup").write_text(f"0::/docker/{CONTAINER_ID}\n")
    (process_root / "mountinfo").write_text(
        "36 25 0:32 / /root rw,nosuid,nodev - fuse.gocryptfs gocryptfs rw,user_id=0\n"
    )
    inspections = iter(
        [
            json.dumps(
                [
                    {
                        "Id": CONTAINER_ID,
                        "State": {"Running": True, "Pid": 4321},
                        "Mounts": [{"Name": "customer-volume", "Destination": "/lium-cipher"}],
                    }
                ]
            ),
            json.dumps(
                [
                    {
                        "Id": CONTAINER_ID,
                        "State": {"Running": True, "Pid": 9999},
                        "Mounts": [{"Name": "customer-volume", "Destination": "/lium-cipher"}],
                    }
                ]
            ),
        ]
    )
    monkeypatch.setattr(
        "storage.workspace.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=next(inspections), stderr=""),
    )

    with pytest.raises(WorkspaceResolutionError, match="container changed"):
        WorkspaceResolver(docker_binary="docker", proc_root=tmp_path, environ={}).resolve(operation)


class _FakePopen:
    def __init__(self, command: list[str], *args: object, **kwargs: object) -> None:
        self.command = command
        self.cwd = kwargs.get("cwd")
        self.stdout = iter(
            [
                '{"message_type":"status","percent_done":0.5}\n',
                f'{{"message_type":"summary","snapshot_id":"{SNAPSHOT_ID}"}}\n',
            ]
        )

    def wait(self) -> int:
        return 3

    def kill(self) -> None:
        return None


def test_exit_code_three_with_snapshot_is_completed_partial(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    operation = StorageOperationSpec.from_mapping(_operation_payload())
    output = io.StringIO()
    events = JsonEventWriter(str(OPERATION_ID), 0, output=output)
    workspace = LocalWorkspace(tmp_path)
    runner = ResticStorageRunner(operation, workspace, event_writer=events)

    monkeypatch.setattr(
        "storage.restic.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="[]", stderr=""),
    )
    monkeypatch.setattr("storage.restic.subprocess.Popen", _FakePopen)

    result = runner.run()

    assert result.status == "COMPLETED"
    assert result.result_quality is OperationResultQuality.PARTIAL
    assert result.snapshot_id == SNAPSHOT_ID
    assert '"result_quality":"PARTIAL"' in output.getvalue()


def test_docker_restore_writes_to_requested_directory() -> None:
    operation = StorageOperationSpec.from_mapping(_operation_payload(action="restore"))
    workspace = DockerVolumeWorkspace(
        image="executor:test",
        volume_name="customer-volume",
        path=PurePosixPath("/workspace/restored"),
        read_only=False,
    )
    runner = ResticStorageRunner(operation, workspace)

    command, _ = runner._execution_command(
        ["restore", "--json", f"{SNAPSHOT_ID}:/", "--target", "/workspace/restored"],
        working_directory=False,
    )

    assert "customer-volume:/workspace:rw" in command
    assert "--workdir" not in command
    assert f"{SNAPSHOT_ID}:/" in command
