from __future__ import annotations

import bz2
import hashlib
import io
import json
import os
import shutil
import signal
import subprocess
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import UUID

import pytest
import requests
import storage_runner
from storage.models import (
    OperationResultQuality,
    OperationSpecError,
    ReporterResource,
    ReporterSpec,
    StorageAction,
    StorageOperationSpec,
)
from storage.reporting import (
    PENDING_LOGS_MAX_BYTES,
    ReportingLeaseExpired,
    StorageEventReporter,
    _newest_within_bytes,
)
from storage.restic import (
    BACKUP_SOURCE_SCRIPT,
    JsonEventWriter,
    ResticOperationError,
    ResticStorageRunner,
    RestoreStats,
    StorageOperationCancelled,
)
from storage.workspace import (
    DockerUserNamespaceWorkspace,
    DockerVolumeWorkspace,
    LocalWorkspace,
    WorkspaceResolutionError,
    WorkspaceResolver,
)

OPERATION_ID = UUID("11111111-1111-4111-8111-111111111111")
POD_ID = UUID("22222222-2222-4222-8222-222222222222")
SNAPSHOT_ID = "a" * 64
CONTAINER_ID = "b" * 64


def _operation_payload(
    *,
    action: str = "backup",
    mode: str = "plain_volume",
    requested_path: str = "/root/checkpoints",
    bootstrap: bool | None = None,
) -> dict[str, object]:
    workspace: dict[str, object] = {
        "mode": mode,
        "volume_name": "customer-volume",
        "volume_path": "/root",
        "requested_path": requested_path,
    }
    if mode == "encrypted_running":
        workspace["container_name"] = "rental-pod"
    if bootstrap is not None:
        workspace["bootstrap"] = bootstrap
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


def test_legacy_s3_connection_override_is_ignored() -> None:
    payload = _operation_payload()
    repository = payload["repository"]
    assert isinstance(repository, dict)
    repository["s3_connections"] = 64

    operation = StorageOperationSpec.from_mapping(payload)
    runner = ResticStorageRunner(operation, LocalWorkspace(Path("/workspace")))

    assert "s3.connections=64" not in runner._restic_command("snapshots")


def test_restore_requires_snapshot_id() -> None:
    payload = _operation_payload(action="restore")
    payload["snapshot_id"] = None

    with pytest.raises(OperationSpecError, match="snapshot_id is required"):
        StorageOperationSpec.from_mapping(payload)


def test_restore_rejects_snapshot_id_options() -> None:
    payload = _operation_payload(action="restore")
    payload["snapshot_id"] = "--password-command=malicious-command"

    with pytest.raises(OperationSpecError, match="hexadecimal restic snapshot ID"):
        StorageOperationSpec.from_mapping(payload)


def test_restore_missing_repository_never_initializes_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    operation = StorageOperationSpec.from_mapping(_operation_payload(action="restore"))
    commands: list[list[str]] = []

    def run(command: list[str], **kwargs: object) -> SimpleNamespace:
        commands.append(command)
        return SimpleNamespace(returncode=10, stdout="", stderr="repository does not exist")

    monkeypatch.setattr("storage.restic.subprocess.run", run)

    with pytest.raises(ResticOperationError, match="repository does not exist") as raised:
        ResticStorageRunner(operation, LocalWorkspace(tmp_path)).run()

    assert raised.value.error_code == "RESTIC_REPOSITORY_MISSING"
    assert commands
    assert commands[0][-4:] == ["snapshots", "--json", "--", SNAPSHOT_ID]
    assert all("init" not in command for command in commands)


class _RetryClock:
    def __init__(self) -> None:
        self.now: float = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def _repository_command_result(
    exit_code: int,
    *,
    standard_output: str = "",
    standard_error: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["restic"],
        returncode=exit_code,
        stdout=standard_output,
        stderr=standard_error,
    )


class _RepositoryCommandSequence:
    def __init__(self, responses: list[subprocess.CompletedProcess[str]]) -> None:
        self.commands: list[list[str]] = []
        self._responses: Iterator[subprocess.CompletedProcess[str]] = iter(responses)

    def __call__(
        self,
        command: list[str],
        **_: object,
    ) -> subprocess.CompletedProcess[str]:
        self.commands.append(command)
        return next(self._responses)


def test_backup_repository_probe_retries_transient_access_denied(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    operation = StorageOperationSpec.from_mapping(_operation_payload())
    clock = _RetryClock()
    event_output = io.StringIO()
    event_writer = JsonEventWriter(
        str(OPERATION_ID),
        0,
        heartbeat_interval_seconds=1.0,
        output=event_output,
        clock=clock,
    )
    runner = ResticStorageRunner(operation, LocalWorkspace(tmp_path), event_writer=event_writer)
    repository_command_sequence = _RepositoryCommandSequence(
        [
            _repository_command_result(
                1,
                standard_error="Stat(<config/>) failed: Stat: Access Denied",
            ),
            _repository_command_result(10, standard_error="repository does not exist"),
            _repository_command_result(0, standard_output="{}"),
        ]
    )

    monkeypatch.setattr("storage.restic.subprocess.run", repository_command_sequence)
    monkeypatch.setattr("storage.restic.time.monotonic", clock)
    monkeypatch.setattr("storage.restic.time.sleep", clock.sleep)
    runner._ensure_repository_for_backup()

    assert [command[2] for command in repository_command_sequence.commands] == [
        "snapshots",
        "snapshots",
        "init",
    ]
    assert clock.now == 2.0
    assert '"event":"heartbeat"' in event_output.getvalue()


def test_backup_repository_initialization_retries_transient_access_denied(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    operation = StorageOperationSpec.from_mapping(_operation_payload())
    runner = ResticStorageRunner(operation, LocalWorkspace(tmp_path))
    clock = _RetryClock()
    repository_command_sequence = _RepositoryCommandSequence(
        [
            _repository_command_result(10, standard_error="repository does not exist"),
            _repository_command_result(1, standard_error="S3 error: AccessDenied"),
            _repository_command_result(1, standard_error="repository already initialized"),
            _repository_command_result(0, standard_output="[]"),
        ]
    )

    monkeypatch.setattr("storage.restic.subprocess.run", repository_command_sequence)
    monkeypatch.setattr("storage.restic.time.monotonic", clock)
    monkeypatch.setattr("storage.restic.time.sleep", clock.sleep)
    runner._ensure_repository_for_backup()

    assert [command[2] for command in repository_command_sequence.commands] == [
        "snapshots",
        "init",
        "init",
        "snapshots",
    ]
    assert clock.now == 2.0


def test_backup_repository_retry_stops_on_cancellation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    operation = StorageOperationSpec.from_mapping(_operation_payload())
    clock = _RetryClock()
    event_writer = JsonEventWriter(
        str(OPERATION_ID),
        0,
        clock=clock,
        cancellation_probe=lambda: clock.now >= 1.0,
    )
    runner = ResticStorageRunner(operation, LocalWorkspace(tmp_path), event_writer=event_writer)

    monkeypatch.setattr(
        "storage.restic.subprocess.run",
        lambda *args, **kwargs: _repository_command_result(
            1,
            standard_error="Stat: Access Denied",
        ),
    )
    monkeypatch.setattr("storage.restic.time.monotonic", clock)
    monkeypatch.setattr("storage.restic.time.sleep", clock.sleep)
    with pytest.raises(StorageOperationCancelled, match="cancellation requested"):
        runner._ensure_repository_for_backup()

    assert clock.now == 1.0


def test_backup_repository_retry_exhaustion_preserves_redacted_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    operation = StorageOperationSpec.from_mapping(_operation_payload())
    runner = ResticStorageRunner(operation, LocalWorkspace(tmp_path))
    clock = _RetryClock()
    run_repository_command = MagicMock(
        return_value=_repository_command_result(
            1,
            standard_error=(
                "Stat(<config/>) failed: Stat: Access Denied for secret-key"
            ),
        )
    )

    monkeypatch.setattr("storage.restic.subprocess.run", run_repository_command)
    monkeypatch.setattr("storage.restic.time.monotonic", clock)
    monkeypatch.setattr("storage.restic.time.sleep", clock.sleep)
    with pytest.raises(ResticOperationError, match="Backup storage was not ready") as raised:
        runner._ensure_repository_for_backup()

    assert run_repository_command.call_count == 10
    assert clock.now == 180.0
    expected_error = "Stat(<config/>) failed: Stat: Access Denied for [REDACTED]"
    assert expected_error in str(raised.value)
    assert expected_error in capsys.readouterr().err
    assert "secret-key" not in str(raised.value)


@pytest.mark.parametrize(
    "detail",
    (
        "Fatal: wrong password or no key found",
        "dial tcp: lookup invalid.example: no such host",
        "The AWS Access Key Id you provided does not exist in our records.",
        "503 Service Unavailable",
    ),
)
def test_backup_repository_does_not_retry_other_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    detail: str,
) -> None:
    operation = StorageOperationSpec.from_mapping(_operation_payload())
    runner = ResticStorageRunner(operation, LocalWorkspace(tmp_path))
    run_repository_command = MagicMock(
        return_value=_repository_command_result(1, standard_error=detail)
    )
    sleep_mock = MagicMock()

    monkeypatch.setattr("storage.restic.subprocess.run", run_repository_command)
    monkeypatch.setattr("storage.restic.time.sleep", sleep_mock)

    with pytest.raises(ResticOperationError, match="repository probe failed"):
        runner._ensure_repository_for_backup()

    run_repository_command.assert_called_once()
    sleep_mock.assert_not_called()


def test_requested_path_must_stay_inside_volume() -> None:
    operation = StorageOperationSpec.from_mapping(_operation_payload(requested_path="/etc"))

    with pytest.raises(WorkspaceResolutionError, match="outside volume path"):
        WorkspaceResolver(environ={"LIUM_STORAGE_HELPER_IMAGE": "executor:test"}).resolve(operation)


def test_helper_image_resolves_container_hostname_when_ssh_environment_omits_hostname(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("storage.workspace.socket.gethostname", lambda: "a" * 12)
    monkeypatch.setattr(
        "storage.workspace.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=f"sha256:{'b' * 64}\n",
            stderr="",
        ),
    )

    image = WorkspaceResolver(docker_binary="docker", environ={})._current_executor_image()

    assert image == f"sha256:{'b' * 64}"


def test_plain_backup_uses_read_only_volume_and_keeps_secrets_out_of_arguments() -> None:
    operation = StorageOperationSpec.from_mapping(_operation_payload())
    workspace = WorkspaceResolver(environ={"LIUM_STORAGE_HELPER_IMAGE": "executor:test"}).resolve(
        operation
    )
    runner = ResticStorageRunner(operation, workspace)

    command, cwd = runner._execution_command(["backup", "--json", "."], working_directory=True)

    assert cwd is None
    assert "customer-volume:/workspace:ro" in command
    script = command.index(BACKUP_SOURCE_SCRIPT)
    assert command[script + 2 : script + 4] == ["/workspace/checkpoints", runner._restic_binary]
    assert command[command.index("--log-driver") + 1] == "none"
    assert command[command.index("--tmpfs") + 1] == "/tmp:rw,nosuid,nodev,size=536870912"
    assert command[command.index("--memory") + 1] == "4g"
    assert "GOMAXPROCS=8" in command
    assert command[command.index("--oom-score-adj") + 1] == "1000"
    assert "GOGC=50" in command
    assert command[command.index("--label") + 1] == f"lium.backup-helper={OPERATION_ID}"
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

    commands: list[list[str]] = []

    def run_docker(command: list[str], **kwargs: object) -> SimpleNamespace:
        commands.append(command)
        stdout = inspection if command[1] == "inspect" else ""
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr("storage.workspace.subprocess.run", run_docker)

    workspace = WorkspaceResolver(
        docker_binary="docker",
        proc_root=tmp_path,
        environ={"LIUM_STORAGE_HELPER_IMAGE": "executor:test"},
    ).resolve(operation)

    assert workspace == DockerUserNamespaceWorkspace(
        image="executor:test",
        container_name="rental-pod",
        container_id=CONTAINER_ID,
        pid=4321,
        path=PurePosixPath(str(plaintext)),
        read_only=True,
    )
    preflight = next(command for command in commands if command[1] == "run")
    assert preflight[preflight.index("--pid") + 1] == "host"
    assert "--privileged" in preflight
    assert preflight[preflight.index("--entrypoint") + 1] == "/usr/bin/nsenter"
    assert "-U" in preflight
    assert "-m" not in preflight


def _running_encrypted_pod(tmp_path: Path, plaintext_relative: str = "root/root") -> str:
    process_root = tmp_path / "4321"
    (process_root / plaintext_relative).mkdir(parents=True)
    (process_root / "cgroup").write_text(f"0::/docker/{CONTAINER_ID}\n")
    (process_root / "mountinfo").write_text(
        "36 25 0:32 / /root rw,nosuid,nodev - fuse.gocryptfs gocryptfs rw,user_id=0\n"
    )
    return json.dumps(
        [
            {
                "Id": CONTAINER_ID,
                "State": {"Running": True, "Pid": 4321},
                "Mounts": [{"Name": "customer-volume", "Destination": "/lium-cipher"}],
            }
        ]
    )


@pytest.mark.parametrize("bootstrap", (None, False))
def test_online_encrypted_restore_still_refuses_a_nonempty_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    bootstrap: bool | None,
) -> None:
    operation = StorageOperationSpec.from_mapping(
        _operation_payload(action="restore", mode="encrypted_running", requested_path="/root", bootstrap=bootstrap)
    )
    inspection = _running_encrypted_pod(tmp_path)

    def run_docker(command: list[str], **kwargs: object) -> SimpleNamespace:
        if command[1] == "inspect":
            return SimpleNamespace(returncode=0, stdout=inspection, stderr="")
        return SimpleNamespace(returncode=21, stdout="", stderr="")   # the target has entries

    monkeypatch.setattr("storage.workspace.subprocess.run", run_docker)

    with pytest.raises(WorkspaceResolutionError, match="new or empty"):
        WorkspaceResolver(
            docker_binary="docker", proc_root=tmp_path, environ={"LIUM_STORAGE_HELPER_IMAGE": "executor:test"}
        ).resolve(operation)


def test_bootstrap_encrypted_restore_skips_the_emptiness_preflight(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # DAH-3274: at create time the pod has been running since `docker run`, so `.jupyter` or
    # `.bashrc` may already sit in the fresh mount. Nothing there is the customer's; the
    # preflight only checks the target is a directory and the restore writes over it.
    operation = StorageOperationSpec.from_mapping(
        _operation_payload(action="restore", mode="encrypted_running", requested_path="/root", bootstrap=True)
    )
    inspection = _running_encrypted_pod(tmp_path)
    scripts: list[str] = []

    def run_docker(command: list[str], **kwargs: object) -> SimpleNamespace:
        if command[1] == "inspect":
            return SimpleNamespace(returncode=0, stdout=inspection, stderr="")
        scripts.append(command[command.index("-c") + 1])
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("storage.workspace.subprocess.run", run_docker)

    workspace = WorkspaceResolver(
        docker_binary="docker", proc_root=tmp_path, environ={"LIUM_STORAGE_HELPER_IMAGE": "executor:test"}
    ).resolve(operation)

    assert workspace.read_only is False
    (preflight,) = scripts
    assert "exit 20" in preflight          # a file where the directory should be still fails
    assert "find" not in preflight         # emptiness is not required
    assert "-mindepth" not in preflight


def test_bootstrap_flag_is_refused_outside_encrypted_running() -> None:
    with pytest.raises(
        OperationSpecError, match="bootstrap is only meaningful for encrypted_running"
    ):
        StorageOperationSpec.from_mapping(_operation_payload(action="restore", bootstrap=True))
    # a backup carrying the flag would parse and silently ignore it (review, taiberium)
    with pytest.raises(OperationSpecError, match="bootstrap is only meaningful for restore"):
        StorageOperationSpec.from_mapping(
            _operation_payload(action="backup", mode="encrypted_running", bootstrap=True)
        )
    with pytest.raises(OperationSpecError, match="must be a boolean"):
        StorageOperationSpec.from_mapping(
            _operation_payload(action="restore", mode="encrypted_running", bootstrap="yes")  # type: ignore[arg-type]
        )


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
        WorkspaceResolver(
            docker_binary="docker",
            proc_root=tmp_path,
            environ={"LIUM_STORAGE_HELPER_IMAGE": "executor:test"},
        ).resolve(operation)


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
        WorkspaceResolver(
            docker_binary="docker",
            proc_root=tmp_path,
            environ={"LIUM_STORAGE_HELPER_IMAGE": "executor:test"},
        ).resolve(operation)


def test_encrypted_workspace_without_host_pid_namespace_names_the_host_fault(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    operation = StorageOperationSpec.from_mapping(_operation_payload(mode="encrypted_running"))
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

    with pytest.raises(WorkspaceResolutionError, match=r"host PID namespace \(pid: host\)"):
        WorkspaceResolver(
            docker_binary="docker",
            proc_root=tmp_path,
            environ={"LIUM_STORAGE_HELPER_IMAGE": "executor:test"},
        ).resolve(operation)


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

    def run_docker(command: list[str], **kwargs: object) -> SimpleNamespace:
        if command[1] == "inspect":
            return SimpleNamespace(returncode=0, stdout=next(inspections), stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("storage.workspace.subprocess.run", run_docker)

    with pytest.raises(WorkspaceResolutionError, match="container changed"):
        WorkspaceResolver(
            docker_binary="docker",
            proc_root=tmp_path,
            environ={"LIUM_STORAGE_HELPER_IMAGE": "executor:test"},
        ).resolve(operation)


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
    (tmp_path / "file").write_text("x")
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


class _KilledPopen(_FakePopen):
    def wait(self) -> int:
        return 137


@pytest.mark.parametrize(
    ("oom_killed", "message"),
    [("true", "^backup ran out of memory$"), ("false", r"^backup helper was killed \(often out of memory\)$")],
)
def test_killed_helper_reports_oom_only_when_docker_says_so(
    monkeypatch: pytest.MonkeyPatch, oom_killed: str, message: str
) -> None:
    operation = StorageOperationSpec.from_mapping(_operation_payload())
    workspace = WorkspaceResolver(environ={"LIUM_STORAGE_HELPER_IMAGE": "executor:test"}).resolve(operation)
    runner = ResticStorageRunner(operation, workspace)
    calls: list[list[str]] = []

    def fake_run(command: list[str], *args: object, **kwargs: object) -> SimpleNamespace:
        calls.append(command)
        stdout = f"{oom_killed}\n" if "inspect" in command else "[]"
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr("storage.restic.subprocess.run", fake_run)
    monkeypatch.setattr("storage.restic.subprocess.Popen", _KilledPopen)

    with pytest.raises(ResticOperationError, match=message):
        runner.run()
    assert calls[-1][1:3] == ["rm", "--force"]


class _TerminatedRunnerPopen(_FakePopen):
    pid = 0

    def wait(self, timeout: float | None = None) -> int:
        os.kill(os.getpid(), signal.SIGTERM)
        return 0

    def poll(self) -> int:
        return 0


def test_sigterm_to_runner_removes_the_helper_container(monkeypatch: pytest.MonkeyPatch) -> None:
    operation = StorageOperationSpec.from_mapping(_operation_payload())
    workspace = WorkspaceResolver(environ={"LIUM_STORAGE_HELPER_IMAGE": "executor:test"}).resolve(operation)
    runner = ResticStorageRunner(operation, workspace)
    calls: list[list[str]] = []

    def fake_run(command: list[str], *args: object, **kwargs: object) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="[]", stderr="")

    monkeypatch.setattr("storage.restic.subprocess.run", fake_run)
    monkeypatch.setattr("storage.restic.subprocess.Popen", _TerminatedRunnerPopen)
    previous = signal.signal(signal.SIGTERM, storage_runner._cancel_on_sigterm)
    try:
        with pytest.raises(StorageOperationCancelled, match="SIGTERM"):
            runner.run()
    finally:
        signal.signal(signal.SIGTERM, previous)

    assert calls[-1][1:3] == ["rm", "--force"]


def test_stale_helper_cleanup_removes_only_long_exited_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    operation = StorageOperationSpec.from_mapping(_operation_payload())
    workspace = WorkspaceResolver(environ={"LIUM_STORAGE_HELPER_IMAGE": "executor:test"}).resolve(operation)
    runner = ResticStorageRunner(operation, workspace)
    old = (datetime.now(timezone.utc) - timedelta(minutes=11)).strftime("%Y-%m-%dT%H:%M:%S.123456789Z")
    fresh = (datetime.now(timezone.utc) - timedelta(minutes=2)).strftime("%Y-%m-%dT%H:%M:%S.123456789Z")
    calls: list[list[str]] = []

    def fake_run(command: list[str], *args: object, **kwargs: object) -> SimpleNamespace:
        calls.append(command)
        stdout = ""
        if command[1] == "ps":
            stdout = "old\nfresh\nrestarted\n"
        elif command[1] == "inspect":
            stdout = f"old exited {old}\nfresh exited {fresh}\nrestarted running {old}\n"
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr("storage.restic.subprocess.run", fake_run)

    runner._remove_stopped_helpers()

    assert "status=exited" in calls[0] and "label=lium.backup-helper" in calls[0]
    assert [c[1:] for c in calls if c[1] == "rm"] == [["rm", "--force", "old"]]


def test_docker_restore_uses_native_json_restore_to_requested_directory() -> None:
    operation = StorageOperationSpec.from_mapping(_operation_payload(action="restore"))
    workspace = DockerVolumeWorkspace(
        image="executor:test",
        volume_name="customer-volume",
        path=PurePosixPath("/workspace/restored"),
        read_only=False,
    )
    runner = ResticStorageRunner(operation, workspace)

    command, _ = runner._restore_execution_command(SNAPSHOT_ID)

    assert "customer-volume:/workspace:rw" in command
    assert "--workdir" not in command
    assert command[-2:] == ["--", SNAPSHOT_ID]
    assert "/workspace/restored" in command
    assert "restore" in command
    assert "--json" in command
    assert "--sparse" in command
    assert "dump" not in command


def test_encrypted_backup_enters_only_the_rental_user_namespace() -> None:
    operation = StorageOperationSpec.from_mapping(_operation_payload(mode="encrypted_running"))
    workspace = DockerUserNamespaceWorkspace(
        image="executor:test",
        container_name="rental-pod",
        container_id=CONTAINER_ID,
        pid=4321,
        path=PurePosixPath("/proc/4321/root/root/checkpoints"),
        read_only=True,
    )
    runner = ResticStorageRunner(operation, workspace)

    command, cwd = runner._execution_command(["backup", "--json", "."], working_directory=True)

    assert cwd is None
    assert command[command.index("--pid") + 1] == "host"
    assert "--privileged" in command
    assert command[command.index("--security-opt") + 1] == "label=disable"
    assert command[command.index("--entrypoint") + 1] == "/usr/bin/nsenter"
    assert command[command.index("-t") + 1] == "4321"
    assert "-U" in command
    assert "-m" not in command
    assert "/proc/4321/root/root/checkpoints" in command
    assert not any(argument.startswith("s3.connections=") for argument in command)


def test_encrypted_restore_preserves_user_xattrs() -> None:
    operation = StorageOperationSpec.from_mapping(
        _operation_payload(action="restore", mode="encrypted_running")
    )
    workspace = DockerUserNamespaceWorkspace(
        image="executor:test",
        container_name="rental-pod",
        container_id=CONTAINER_ID,
        pid=4321,
        path=PurePosixPath("/proc/4321/root/root/restored"),
        read_only=False,
    )
    runner = ResticStorageRunner(operation, workspace)

    command, cwd = runner._restore_execution_command(SNAPSHOT_ID)

    assert cwd is None
    assert command[command.index("--entrypoint") + 1] == "/usr/bin/nsenter"
    assert command[command.index("-t") + 1] == "4321"
    assert "/proc/4321/root/root/restored" in command
    assert "restore" in command
    assert "--json" in command
    excluded_xattrs = [
        command[index + 1]
        for index, argument in enumerate(command)
        if argument == "--exclude-xattr"
    ]
    assert excluded_xattrs == ["security.*", "trusted.*"]
    assert "user.*" not in excluded_xattrs


def test_spec_refuses_a_volume_passphrase() -> None:
    # DAH-3274: the spec file lands on the provider's disk; a validator that still puts the
    # gocryptfs passphrase in it is refused before anything is built from it.
    payload = _operation_payload(action="restore", mode="encrypted_running")
    payload["workspace"]["volume_passphrase"] = "gocryptfs-secret"

    with pytest.raises(OperationSpecError, match="must not carry volume_passphrase"):
        StorageOperationSpec.from_mapping(payload)


def test_encrypted_bootstrap_mode_no_longer_exists() -> None:
    payload = _operation_payload(action="restore", mode="encrypted_bootstrap")

    with pytest.raises(OperationSpecError, match="workspace.mode must be one of"):
        StorageOperationSpec.from_mapping(payload)


@pytest.mark.parametrize(
    "workspace",
    [
        DockerVolumeWorkspace(
            image="executor:test",
            volume_name="customer-volume",
            path=PurePosixPath("/workspace"),
            read_only=False,
        ),
        DockerUserNamespaceWorkspace(
            image="executor:test",
            container_name="rental-pod",
            container_id=CONTAINER_ID,
            pid=4321,
            path=PurePosixPath("/proc/4321/root/root"),
            read_only=False,
        ),
    ],
    ids=["plain_volume", "encrypted_running"],
)
def test_no_rendered_helper_spec_carries_a_volume_passphrase(
    workspace: DockerVolumeWorkspace | DockerUserNamespaceWorkspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The rendered `docker run` argv plus the environment handed to it IS the helper's
    # container spec — what `docker inspect` and config.v2.json show the provider. Grep it.
    monkeypatch.setenv("LIUM_VOLUME_PASSPHRASE", "leaked-from-the-executor-environment")
    mode = "encrypted_running" if isinstance(workspace, DockerUserNamespaceWorkspace) else "plain_volume"
    runner = ResticStorageRunner(
        StorageOperationSpec.from_mapping(_operation_payload(action="restore", mode=mode)),
        workspace,
    )

    rendered: list[str] = []
    forwarded_env: set[str] = set()
    for command, _ in (
        runner._restore_execution_command(SNAPSHOT_ID),
        runner._legacy_restore_execution_command("legacy/object.tgz"),
        runner._execution_command(["backup", "--json", "."], working_directory=True),
    ):
        rendered.extend(command)
        forwarded_env.update(
            command[index + 1] for index, argument in enumerate(command) if argument == "-e"
        )

    # only `-e NAME` entries cross from the executor's environment into the container
    assert "LIUM_VOLUME_PASSPHRASE" not in forwarded_env, forwarded_env
    assert not any("passphrase" in item.lower() for item in rendered), rendered
    assert not any("gocryptfs" in item for item in rendered), rendered
    assert "leaked-from-the-executor-environment" not in " ".join(rendered)


def test_local_cancellation_marker_is_observed() -> None:
    events = JsonEventWriter(
        str(OPERATION_ID),
        0,
        cancellation_probe=lambda: True,
    )

    assert events.cancellation_requested is True


class _FakeRestorePopen:
    def __init__(self, command: list[str], *args: object, **kwargs: object) -> None:
        self.stdout = iter(["tar: LIUM_RESTORE_CHECKPOINT_1024\n"])

    def wait(self) -> int:
        return 0

    def kill(self) -> None:
        return None


class _SingleEventPopen:
    def __init__(self, command: list[str], *args: object, **kwargs: object) -> None:
        self.stdout = iter(['{"message_type":"status","percent_done":0.1}\n'])


def test_reporting_lease_loss_terminates_the_owned_process(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    operation = StorageOperationSpec.from_mapping(_operation_payload())
    events = SimpleNamespace(
        heartbeat_interval_seconds=1,
        restic_event=MagicMock(side_effect=ReportingLeaseExpired("lease expired")),
    )
    runner = ResticStorageRunner(operation, LocalWorkspace(tmp_path), event_writer=events)
    terminate = MagicMock()
    monkeypatch.setattr("storage.restic.subprocess.Popen", _SingleEventPopen)
    monkeypatch.setattr(runner, "_terminate_process", terminate)

    with pytest.raises(ReportingLeaseExpired, match="lease expired"):
        runner._stream_command(["backup-test"], None)

    terminate.assert_called_once()


def test_restore_checkpoint_emits_bounded_progress(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    operation = StorageOperationSpec.from_mapping(_operation_payload(action="restore"))
    output = io.StringIO()
    events = JsonEventWriter(str(OPERATION_ID), 0, output=output)
    runner = ResticStorageRunner(operation, LocalWorkspace(tmp_path), event_writer=events)
    monkeypatch.setattr("storage.restic.subprocess.Popen", _FakeRestorePopen)

    exit_code, _ = runner._stream_command(
        ["restore-test"],
        None,
        restore_stats=RestoreStats(total_size=20 * 1024 * 1024, total_file_count=3),
    )

    assert exit_code == 0
    event = json.loads(output.getvalue())
    assert event["payload"]["percent_done"] == 0.5
    assert event["payload"]["bytes_restored"] == 10 * 1024 * 1024


class _ReporterResponse:
    status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, bool]:
        return {"cancel_requested": True}


class _ReporterSession:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    def put(self, url: str, **kwargs: object) -> _ReporterResponse:
        self.requests.append({"url": url, **kwargs})
        return _ReporterResponse()


def test_reporter_preserves_zero_counters_and_receives_cancellation() -> None:
    session = _ReporterSession()
    reporter = StorageEventReporter(
        OPERATION_ID,
        StorageAction.BACKUP,
        ReporterSpec(
            api_url="https://api.example",
            auth_token="token",
            resource=ReporterResource.BACKUP,
        ),
        session=session,
    )

    reporter.send(
        {
            "event": "restic",
            "payload": {
                "message_type": "status",
                "percent_done": 0,
                "total_files": 0,
                "files_done": 0,
                "total_bytes": 0,
                "bytes_done": 0,
            },
        }
    )

    assert reporter.cancel_requested is True
    request = session.requests[0]
    assert request["url"] == f"https://api.example/backup-logs/{OPERATION_ID}/progress"
    payload = request["json"]
    assert isinstance(payload, dict)
    assert payload["total_files"] == 0
    assert payload["processed_files"] == 0
    assert payload["total_bytes"] == 0
    assert payload["processed_bytes"] == 0


def test_restore_reporter_surfaces_scan_then_restore_stages() -> None:
    session = _ReporterSession()
    clock = _ManualClock()
    reporter = StorageEventReporter(
        OPERATION_ID,
        StorageAction.RESTORE,
        ReporterSpec(
            api_url="https://api.example",
            auth_token="token",
            resource=ReporterResource.RESTORE,
        ),
        session=session,
        clock=clock,
    )

    reporter.send(
        {"event": "restic", "payload": {"message_type": "status", "total_files": 123}}
    )
    clock.now += 10
    reporter.send(
        {
            "event": "restic",
            "payload": {"message_type": "status", "total_files": 123, "files_restored": 1},
        }
    )

    first_payload = session.requests[0]["json"]
    second_payload = session.requests[1]["json"]
    assert isinstance(first_payload, dict)
    assert isinstance(second_payload, dict)
    assert first_payload["stage"] == "PREPARING"
    assert first_payload["total_files"] == 123
    assert second_payload["stage"] == "RESTORING"


def test_reporter_includes_specific_restic_error_in_failed_result() -> None:
    session = _ReporterSession()
    reporter = StorageEventReporter(
        OPERATION_ID,
        StorageAction.BACKUP,
        ReporterSpec(
            api_url="https://api.example",
            auth_token="token",
            resource=ReporterResource.BACKUP,
        ),
        session=session,
    )

    reporter.send({"event": "diagnostic", "message": "docker emitted an unrelated warning"})
    reporter.send(
        {
            "event": "restic",
            "payload": {
                "message_type": "error",
                "error": {"message": "failed to save model.bin: no space left on device"},
            },
        }
    )
    reporter.send({"event": "result", "status": "FAILED", "exit_code": 1})

    terminal_payload = session.requests[-1]["json"]
    assert isinstance(terminal_payload, dict)
    assert terminal_payload["error_message"] == (
        "failed to save model.bin: no space left on device"
    )


class _FailingReporterSession:
    def put(self, *args: object, **kwargs: object) -> None:
        raise requests.ConnectionError("backend unavailable")


class _RevokedReporterSession:
    def put(self, *args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(status_code=404)


def test_missing_restore_log_revokes_runner_without_waiting_for_lease() -> None:
    reporter = StorageEventReporter(
        OPERATION_ID,
        StorageAction.RESTORE,
        ReporterSpec(
            api_url="https://api.example",
            auth_token="token",
            resource=ReporterResource.RESTORE,
        ),
        session=_RevokedReporterSession(),
    )

    with pytest.raises(ReportingLeaseExpired, match="no longer active"):
        reporter.send({"event": "heartbeat"})


class _ManualClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def test_terminal_reporting_retries_until_the_reporting_lease_expires() -> None:
    clock = _ManualClock()
    reporter = StorageEventReporter(
        OPERATION_ID,
        StorageAction.BACKUP,
        ReporterSpec(
            api_url="https://api.example",
            auth_token="token",
            resource=ReporterResource.BACKUP,
            failure_timeout_seconds=3,
        ),
        session=_FailingReporterSession(),
        clock=clock,
        sleeper=clock.sleep,
    )

    with pytest.raises(ReportingLeaseExpired, match="reporting was unavailable"):
        reporter.send({"event": "result", "status": "COMPLETED", "result_quality": "FULL"})

    assert clock.now == 3


class _WafBlockedSession:
    def __init__(self) -> None:
        self.status_code = 403
        self.put_times: list[float] = []
        self.clock: _ManualClock | None = None

    def put(self, *args: object, **kwargs: object) -> SimpleNamespace:
        assert self.clock is not None
        self.put_times.append(self.clock.now)
        status_code = self.status_code

        def raise_for_status() -> None:
            if status_code >= 400:
                raise requests.HTTPError(f"{status_code} blocked")

        return SimpleNamespace(
            status_code=status_code, raise_for_status=raise_for_status, json=lambda: {}
        )


def test_progress_reporting_is_throttled_and_backs_off_on_403() -> None:
    clock = _ManualClock()
    session = _WafBlockedSession()
    session.clock = clock
    session.status_code = 200
    reporter = StorageEventReporter(
        OPERATION_ID,
        StorageAction.RESTORE,
        ReporterSpec(
            api_url="https://api.example",
            auth_token="token",
            resource=ReporterResource.RESTORE,
        ),
        session=session,
        clock=clock,
    )

    # Four events a second for 30 s while the backend accepts: one PUT per 10 s.
    for _ in range(120):
        reporter.send({"event": "heartbeat"})
        clock.now += 0.25
    assert session.put_times == [0.0, 10.0, 20.0]

    # The WAF now answers 403: retries spread out (10, 20, 40, 60 s ceilings, with jitter)
    # and never raise while the reporting lease is still valid.
    session.status_code = 403
    session.put_times.clear()
    for _ in range(480):
        reporter.send({"event": "heartbeat"})
        clock.now += 0.25
    gaps = [b - a for a, b in zip(session.put_times, session.put_times[1:])]
    assert len(session.put_times) <= 8
    assert all(gap >= 5 for gap in gaps)
    assert all(gap <= 60.25 for gap in gaps)
    assert any(gap > 20 for gap in gaps)

    # The final result is always sent, even inside the backoff window.
    session.status_code = 200
    session.put_times.clear()
    reporter.send({"event": "result", "status": "COMPLETED", "result_quality": "FULL"})
    assert session.put_times == [clock.now]


def test_reporter_full_completion_finishes_known_counters() -> None:
    session = _ReporterSession()
    reporter = StorageEventReporter(
        OPERATION_ID,
        StorageAction.BACKUP,
        ReporterSpec(
            api_url="https://api.example",
            auth_token="token",
            resource=ReporterResource.BACKUP,
        ),
        session=session,
    )

    reporter.send(
        {
            "event": "restic",
            "payload": {
                "message_type": "status",
                "total_files": 7,
                "files_done": 2,
                "total_bytes": 100,
                "bytes_done": 25,
            },
        }
    )
    reporter.send(
        {"event": "result", "status": "COMPLETED", "result_quality": "FULL", "exit_code": 0}
    )

    terminal_payload = session.requests[-1]["json"]
    assert isinstance(terminal_payload, dict)
    assert terminal_payload["processed_files"] == 7
    assert terminal_payload["processed_bytes"] == 100


def test_reporter_uses_backup_summary_counters() -> None:
    session = _ReporterSession()
    reporter = StorageEventReporter(
        OPERATION_ID,
        StorageAction.BACKUP,
        ReporterSpec(
            api_url="https://api.example",
            auth_token="token",
            resource=ReporterResource.BACKUP,
        ),
        session=session,
    )

    reporter.send(
        {
            "event": "restic",
            "payload": {
                "message_type": "status",
                "total_files": 726,
                "files_done": 16,
                "total_bytes": 17_424,
                "bytes_done": 408,
            },
        }
    )
    reporter.send(
        {
            "event": "restic",
            "payload": {
                "message_type": "summary",
                "total_files_processed": 2_002,
                "total_bytes_processed": 268_483_487,
            },
        }
    )
    reporter.send(
        {"event": "result", "status": "COMPLETED", "result_quality": "FULL", "exit_code": 0}
    )

    terminal_payload = session.requests[-1]["json"]
    assert isinstance(terminal_payload, dict)
    assert terminal_payload["total_files"] == 2_002
    assert terminal_payload["processed_files"] == 2_002
    assert terminal_payload["total_bytes"] == 268_483_487
    assert terminal_payload["processed_bytes"] == 268_483_487


def test_encrypted_backup_script_stops_when_the_backup_path_is_missing(tmp_path: Path) -> None:
    marker = tmp_path / "ran"

    completed = subprocess.run(
        ["/bin/sh", "-c", BACKUP_SOURCE_SCRIPT, "sh", str(tmp_path / "missing"), "touch", str(marker)],
        capture_output=True,
    )

    assert completed.returncode == 66
    assert not marker.exists()
    # the reporter keeps the first diagnostic, so sh's own cd error must not reach it ahead of ours
    assert completed.stderr == b""


def test_backup_source_script_stops_when_the_backup_path_is_empty(tmp_path: Path) -> None:
    marker = tmp_path / "ran"
    (tmp_path / "backup").mkdir()

    completed = subprocess.run(
        ["/bin/sh", "-c", BACKUP_SOURCE_SCRIPT, "sh", str(tmp_path / "backup"), "touch", str(marker)],
        capture_output=True,
    )

    assert completed.returncode == 66
    assert not marker.exists()


def test_backup_source_script_fails_when_the_backup_path_is_a_file(tmp_path: Path) -> None:
    marker = tmp_path / "ran"
    (tmp_path / "backup").write_text("x")

    completed = subprocess.run(
        ["/bin/sh", "-c", BACKUP_SOURCE_SCRIPT, "sh", str(tmp_path / "backup"), "touch", str(marker)],
        capture_output=True,
    )

    assert completed.returncode == 1
    assert not marker.exists()


def test_local_backup_of_a_file_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "backup"
    path.write_text("x")
    runner = ResticStorageRunner(StorageOperationSpec.from_mapping(_operation_payload()), LocalWorkspace(path))
    monkeypatch.setattr("storage.restic.subprocess.Popen", lambda *a, **k: pytest.fail("restic ran"))

    with pytest.raises(ResticOperationError, match="not a directory"):
        runner._backup()


class _MissingSourcePopen(_FakePopen):
    def __init__(self, command: list[str], *args: object, **kwargs: object) -> None:
        super().__init__(command, *args, **kwargs)
        self.stdout = iter(["sh: cd: can't cd to /workspace/missing: No such file or directory\n"])

    def wait(self) -> int:
        return 66


def test_backup_of_a_missing_path_is_an_empty_no_op(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    operation = StorageOperationSpec.from_mapping(_operation_payload())
    (tmp_path / "file").write_text("x")  # past the local pre-check: restic itself reports the missing source
    runner = ResticStorageRunner(operation, LocalWorkspace(tmp_path))
    monkeypatch.setattr(
        "storage.restic.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="[]", stderr=""),
    )
    monkeypatch.setattr("storage.restic.subprocess.Popen", _MissingSourcePopen)

    result = runner._backup()

    assert (result.status, result.result_quality, result.snapshot_id) == ("COMPLETED", OperationResultQuality.EMPTY, None)


_EMPTY_JSON = '{"message_type":"exit_error","code":1,"message":"Fatal: unable to save snapshot: snapshot is empty"}\n'
_EMPTY_TEXT = "Fatal: unable to save snapshot: snapshot is empty\n"
_EMPTY_IN_FILENAME = '{"message_type":"status","current_files":["/data/snapshot is empty.txt"]}\n'


def _run_backup_with_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, lines: list[str], exclude_patterns: list[str]
) -> object:
    class _Popen(_FakePopen):
        def __init__(self, command: list[str], *args: object, **kwargs: object) -> None:
            super().__init__(command, *args, **kwargs)
            self.stdout = iter(lines)

        def wait(self) -> int:
            return 1

    (tmp_path / "file").write_text("x")
    payload = _operation_payload()
    payload["workspace"]["exclude_patterns"] = exclude_patterns
    runner = ResticStorageRunner(StorageOperationSpec.from_mapping(payload), LocalWorkspace(tmp_path))
    monkeypatch.setattr(
        "storage.restic.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="[]", stderr=""),
    )
    monkeypatch.setattr("storage.restic.subprocess.Popen", _Popen)
    return runner._backup()


@pytest.mark.parametrize("line", [_EMPTY_JSON, _EMPTY_TEXT], ids=["json", "text"])
def test_backup_with_every_file_excluded_is_an_empty_no_op(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, line: str
) -> None:
    result = _run_backup_with_output(monkeypatch, tmp_path, [line], [".cache"])

    assert (result.status, result.result_quality, result.snapshot_id) == ("COMPLETED", OperationResultQuality.EMPTY, None)


@pytest.mark.parametrize(
    ("lines", "exclude_patterns"),
    [([_EMPTY_JSON], []), ([_EMPTY_IN_FILENAME], [".cache"])],
    ids=["no-excludes", "filename-contains-text"],
)
def test_backup_empty_snapshot_without_excludes_or_real_error_still_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, lines: list[str], exclude_patterns: list[str]
) -> None:
    with pytest.raises(ResticOperationError):
        _run_backup_with_output(monkeypatch, tmp_path, lines, exclude_patterns)


@pytest.mark.parametrize("make_dir", [False, True], ids=["missing", "empty"])
def test_local_backup_of_a_missing_or_empty_directory_skips_restic(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, make_dir: bool
) -> None:
    path = tmp_path / "backup"
    if make_dir:
        path.mkdir()
    runner = ResticStorageRunner(StorageOperationSpec.from_mapping(_operation_payload()), LocalWorkspace(path))
    monkeypatch.setattr("storage.restic.subprocess.Popen", lambda *a, **k: pytest.fail("restic ran"))

    result = runner._backup()

    assert (result.status, result.result_quality, result.snapshot_id) == ("COMPLETED", OperationResultQuality.EMPTY, None)


def test_backup_passes_the_exclude_patterns_to_restic(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    payload = _operation_payload()
    payload["workspace"]["exclude_patterns"] = [".cache", ".ssh"]
    (tmp_path / "file").write_text("x")
    runner = ResticStorageRunner(StorageOperationSpec.from_mapping(payload), LocalWorkspace(tmp_path))
    commands = []
    monkeypatch.setattr(runner, "_stream", lambda command, **kwargs: commands.append(command) or (0, {"snapshot_id": SNAPSHOT_ID}))

    runner._backup()

    assert commands[0][-3:] == ["--exclude=.cache", "--exclude=.ssh", "."]


# As neurons/executor/Dockerfile pins it.
RESTIC_URL = "https://github.com/restic/restic/releases/download/v0.19.1/restic_0.19.1_linux_amd64.bz2"
RESTIC_SHA256 = "f415415624dcc452f2a02b8c33641791a8c6d6d3b65bbb3543fcf9a25151585c"


@pytest.fixture(scope="session")
def restic_binary(tmp_path_factory: pytest.TempPathFactory) -> str:
    """restic on PATH, else the pinned release, checksum-verified (CI has no restic)."""
    if found := shutil.which("restic"):
        return found
    archive = requests.get(RESTIC_URL, timeout=60).content
    assert hashlib.sha256(archive).hexdigest() == RESTIC_SHA256
    binary = tmp_path_factory.mktemp("restic") / "restic"
    binary.write_bytes(bz2.decompress(archive))
    binary.chmod(0o755)
    return str(binary)


def test_real_restic_skips_nested_excludes_and_keeps_neighbours(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, restic_binary: str
) -> None:
    """The runner's own backup arguments, run by the real restic binary against a local repository."""
    workspace = tmp_path / "workspace"
    for name in ("proj/.config/gcloud/credentials.db", "proj/.config/gcloud_notes", "proj/.config/other/x", ".env",
                 ".env.local", "proj/model.bin"):
        (workspace / name).parent.mkdir(parents=True, exist_ok=True)
        (workspace / name).write_text("x")
    payload = _operation_payload()
    payload["workspace"]["exclude_patterns"] = [".config/gcloud", ".env"]
    runner = ResticStorageRunner(StorageOperationSpec.from_mapping(payload), LocalWorkspace(workspace))
    commands = []
    monkeypatch.setattr(runner, "_stream", lambda command, **kwargs: commands.append(command) or (0, {"snapshot_id": SNAPSHOT_ID}))
    runner._backup()
    env = {**os.environ, "RESTIC_REPOSITORY": str(tmp_path / "repo"), "RESTIC_PASSWORD": "test"}

    restic = restic_binary
    subprocess.run([restic, "init"], env=env, check=True, capture_output=True)
    subprocess.run([restic, *commands[0]], cwd=workspace, env=env, check=True, capture_output=True)
    listing = subprocess.run([restic, "ls", "latest", "--json"], env=env, check=True, capture_output=True, text=True)
    paths = {entry["path"] for entry in map(json.loads, listing.stdout.splitlines()) if entry.get("path")}

    assert {"/proj/.config/gcloud_notes", "/proj/.config/other/x", "/.env.local", "/proj/model.bin"} <= paths
    assert not {path for path in paths if "/.config/gcloud/" in path or path.endswith("/.config/gcloud")}
    assert "/.env" not in paths


@pytest.mark.parametrize("patterns", [".cache", [""], [1]])
def test_operation_spec_rejects_malformed_exclude_patterns(patterns: object) -> None:
    payload = _operation_payload()
    payload["workspace"]["exclude_patterns"] = patterns

    with pytest.raises(OperationSpecError):
        StorageOperationSpec.from_mapping(payload)


def test_queued_diagnostic_logs_stay_under_the_waf_body_limit() -> None:
    lines = [f"{index}" + "ж" * 1999 for index in range(20)]

    kept = _newest_within_bytes(lines)

    assert kept == lines[-len(kept) :]
    assert len(json.dumps(kept)) <= PENDING_LOGS_MAX_BYTES + len(kept) * 2
    assert len(json.dumps({"logs": kept})) < 64_000


@pytest.mark.parametrize(("kind", "passes"), [("absent", True), ("dir", True), ("file", False)])
def test_encrypted_backup_preflight_lets_an_absent_path_through(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, kind: str, passes: bool
) -> None:
    target = tmp_path / "backup"
    if kind == "dir":
        target.mkdir()
    elif kind == "file":
        target.write_text("x")

    real_run = subprocess.run

    def run_locally(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        script = command[command.index("-c") + 1]
        return real_run(["/bin/sh", "-c", script, "sh", str(target)], capture_output=True, text=True)

    monkeypatch.setattr("storage.workspace.subprocess.run", run_locally)
    validate = lambda: WorkspaceResolver(docker_binary="docker")._validate_encrypted_path(  # noqa: E731
        image="executor:test",
        pid=1,
        visible_path=PurePosixPath(str(target)),
        requested_path=PurePosixPath("/root/backup"),
        action=StorageAction.BACKUP,
    )

    if passes:
        validate()
    else:
        with pytest.raises(WorkspaceResolutionError):
            validate()


EXPIRED_LINE = "Save(<data/1234>) returned error, retrying after 1s: ExpiredToken: The provided token has expired.\n"


class _RenewalSession:
    """DAH-4945: the backend's progress and credentials routes"""

    def __init__(self, status_code: int = 200) -> None:
        self.status_code = status_code
        self.posts: list[dict[str, object]] = []

    def put(self, url: str, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(status_code=200, raise_for_status=lambda: None, json=lambda: {})

    def post(self, url: str, **kwargs: object) -> SimpleNamespace:
        self.posts.append({"url": url, **kwargs})
        n = len(self.posts)
        body = {"access_key_id": f"key-{n}", "secret_access_key": f"secret-{n}", "session_token": f"token-{n}"}
        return SimpleNamespace(status_code=self.status_code, raise_for_status=lambda: None, json=lambda: body)


def _renewing_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    session: _RenewalSession,
    expired_runs: int,
    first_line: str | None = None,
) -> tuple[ResticStorageRunner, list[dict[str, str]]]:
    spec = ReporterSpec(api_url="https://api.test", auth_token="op-token", resource=ReporterResource.BACKUP)
    reporter = StorageEventReporter(
        OPERATION_ID, StorageAction.BACKUP, spec, session=session, sleeper=lambda _: None
    )
    events = JsonEventWriter(str(OPERATION_ID), 0, output=io.StringIO(), reporter=reporter)
    (tmp_path / "file").write_text("x")
    runner = ResticStorageRunner(
        StorageOperationSpec.from_mapping(_operation_payload()), LocalWorkspace(tmp_path), event_writer=events
    )
    environments: list[dict[str, str]] = []

    class _Popen(_FakePopen):
        def __init__(self, command: list[str], *args: object, **kwargs: object) -> None:
            super().__init__(command, *args, **kwargs)
            environments.append(dict(kwargs["env"]))
            if len(environments) <= expired_runs:
                self.stdout = iter(['{"message_type":"status","percent_done":0.5}\n', EXPIRED_LINE])
            elif first_line and len(environments) == 1:
                summary = json.dumps({"message_type": "summary", "snapshot_id": SNAPSHOT_ID})
                self.stdout = iter([*first_line.splitlines(keepends=True), summary + "\n"])

    monkeypatch.setattr("storage.restic.subprocess.Popen", _Popen)
    monkeypatch.setattr(runner, "_terminate_process", MagicMock())
    monkeypatch.setattr(
        "storage.restic.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="[]", stderr=""),
    )
    return runner, environments


@pytest.mark.parametrize(
    ("line", "renewals"),
    [
        (EXPIRED_LINE, 1),
        ("client.PutObject: The provided token has expired.\n", 1),
        ('{"message_type":"error","error":{"message":"The provided token has expired."}}\n', 1),
        ('{"message_type":"exit_error","message":"The provided token has expired."}\n', 1),
        ('{"message_type":"status","current_files":["/workspace/ExpiredToken.java"]}\n', 0),
        (
            '{"message_type":"error","error":{"message":"permission denied"},'
            '"item":"/workspace/ExpiredToken.java"}\n',
            0,
        ),
        (
            '{"message_type":"error","error":{"message":"open /workspace/ExpiredToken.java: permission denied"},'
            '"during":"archival","item":"/workspace/ExpiredToken.java"}\n',
            0,
        ),
        # restic keeps relative source paths: removing the absolute item leaves "open ExpiredToken.java: ..."
        (
            json.dumps(
                {
                    "message_type": "error",
                    "during": "archival",
                    "item": "/workspace/ExpiredToken.java",
                    "error": {"message": "/workspace/ExpiredToken.java: open ExpiredToken.java: permission denied"},
                }
            )
            + "\n",
            0,
        ),
        # expiry while restic removes its lock after the snapshot was saved
        (json.dumps({"message_type": "summary", "snapshot_id": SNAPSHOT_ID}) + "\n" + EXPIRED_LINE, 0),
    ],
)
def test_backup_renews_expired_credentials_and_reruns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, line: str, renewals: int
) -> None:
    session = _RenewalSession()
    runner, environments = _renewing_runner(
        monkeypatch, tmp_path, session, expired_runs=0, first_line=line
    )

    result = runner.run()

    assert (result.status, result.snapshot_id) == ("COMPLETED", SNAPSHOT_ID)
    assert (len(environments), len(session.posts)) == (renewals + 1, renewals)
    if renewals:
        assert [env["AWS_SESSION_TOKEN"] for env in environments] == ["session-token", "token-1"]
        assert environments[1]["AWS_ACCESS_KEY_ID"] == "key-1"
        (post,) = session.posts
        assert post["url"] == f"https://api.test/backup-logs/{OPERATION_ID}/credentials"
        assert post["headers"] == {"Authorization": "Bearer op-token"}


def test_backup_fails_when_the_backend_refuses_renewal(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner, environments = _renewing_runner(
        monkeypatch, tmp_path, _RenewalSession(410), expired_runs=1
    )

    with pytest.raises(ResticOperationError, match="could not be renewed"):
        runner.run()
    assert len(environments) == 1


def test_backup_renews_at_most_three_times(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    session = _RenewalSession()
    runner, environments = _renewing_runner(monkeypatch, tmp_path, session, expired_runs=99)

    with pytest.raises(ResticOperationError, match="credentials expired"):
        runner.run()
    assert (len(environments), len(session.posts)) == (4, 3)
