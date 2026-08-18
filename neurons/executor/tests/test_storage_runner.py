from __future__ import annotations

import io
import json
import subprocess
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import UUID

import pytest
import requests
from storage.models import (
    OperationResultQuality,
    OperationSpecError,
    ReporterResource,
    ReporterSpec,
    StorageAction,
    StorageOperationSpec,
)
from storage.reporting import ReportingLeaseExpired, StorageEventReporter
from storage.restic import (
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
    monkeypatch.setattr("storage.restic.REPOSITORY_RETRY_DELAYS_SECONDS", (2.0,))

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
    monkeypatch.setattr("storage.restic.REPOSITORY_RETRY_DELAYS_SECONDS", (2.0,))

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
    monkeypatch.setattr("storage.restic.REPOSITORY_RETRY_DELAYS_SECONDS", (2.0,))

    with pytest.raises(StorageOperationCancelled, match="cancellation requested"):
        runner._ensure_repository_for_backup()

    assert clock.now == 1.0


def test_backup_repository_retry_exhaustion_returns_customer_safe_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    operation = StorageOperationSpec.from_mapping(_operation_payload())
    runner = ResticStorageRunner(operation, LocalWorkspace(tmp_path))
    clock = _RetryClock()
    run_repository_command = MagicMock(
        return_value=_repository_command_result(1, standard_error="S3 StatusCode: 503")
    )

    monkeypatch.setattr("storage.restic.subprocess.run", run_repository_command)
    monkeypatch.setattr("storage.restic.time.monotonic", clock)
    monkeypatch.setattr("storage.restic.time.sleep", clock.sleep)
    monkeypatch.setattr("storage.restic.REPOSITORY_RETRY_DELAYS_SECONDS", (1.0, 2.0))

    with pytest.raises(ResticOperationError, match="backup storage is not ready"):
        runner._ensure_repository_for_backup()

    assert run_repository_command.call_count == 3
    assert clock.now == 3.0
    assert "StatusCode: 503" in capsys.readouterr().err


@pytest.mark.parametrize(
    "detail",
    (
        "Fatal: wrong password or no key found",
        "dial tcp: lookup invalid.example: no such host",
    ),
)
def test_backup_repository_does_not_retry_permanent_failure(
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
    assert command[command.index("--workdir") + 1] == "/workspace/checkpoints"
    assert command[command.index("--log-driver") + 1] == "none"
    assert command[command.index("--tmpfs") + 1] == "/tmp:rw,nosuid,nodev,size=536870912"
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
    assert SNAPSHOT_ID in command
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
    reporter = StorageEventReporter(
        OPERATION_ID,
        StorageAction.RESTORE,
        ReporterSpec(
            api_url="https://api.example",
            auth_token="token",
            resource=ReporterResource.RESTORE,
        ),
        session=session,
    )

    reporter.send(
        {"event": "restic", "payload": {"message_type": "status", "total_files": 123}}
    )
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
