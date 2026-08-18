from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import IO

from storage.models import (
    OperationResultQuality,
    StorageAction,
    StorageEngine,
    StorageOperationSpec,
)
from storage.reporting import StorageEventReporter
from storage.workspace import (
    DockerEncryptedVolumeWorkspace,
    DockerUserNamespaceWorkspace,
    LocalWorkspace,
    ResolvedWorkspace,
)

RESTORE_CHECKPOINT_PREFIX = "LIUM_RESTORE_CHECKPOINT_"
TAR_RECORD_SIZE_BYTES = 20 * 512
TAR_CHECKPOINT_RECORDS = 1024
MEBIBYTE = 1024 * 1024
RESTIC_TMPFS_BYTES = 512 * MEBIBYTE
# Newly issued backup credentials can take a short time to become usable in AWS.
# This budget covers repository probing, initialization, and confirmation together.
REPOSITORY_READINESS_TIMEOUT_SECONDS: float = 180.0
# Retry quickly at first, then cap the delay so readiness is still checked regularly.
REPOSITORY_RETRY_INITIAL_DELAY_SECONDS: float = 2.0
REPOSITORY_RETRY_MAX_DELAY_SECONDS: float = 30.0
# Short waits keep cancellation and operation heartbeats responsive during backoff.
REPOSITORY_RETRY_POLL_SECONDS: float = 1.0
# Restic treats AccessDenied as permanent, but a newly attached IAM policy can
# return it briefly while AWS propagates the policy to S3.
IAM_POLICY_PROPAGATION_ERROR_MARKER = "accessdenied"
ENCRYPTED_BACKUP_SCRIPT = 'cd "$1"; shift; exec "$@"'
ENCRYPTED_BOOTSTRAP_SCRIPT = r"""
set -eu
passfile=/tmp/.lium-volume-passphrase
umask 077
printf '%s' "$LIUM_VOLUME_PASSPHRASE" > "$passfile"
if [ ! -f /lium-cipher/gocryptfs.conf ]; then
  /usr/local/bin/gocryptfs -init /lium-cipher -passfile "$passfile"
fi
/usr/local/bin/gocryptfs /lium-cipher /workspace -passfile "$passfile" -o allow_other -nonempty
cleanup() {
  /bin/fusermount3 -u /workspace >/dev/null 2>&1 || true
  rm -f "$passfile"
}
trap cleanup EXIT INT TERM
"$@"
"""


class ResticOperationError(RuntimeError):
    def __init__(self, message: str, *, error_code: str | None = None) -> None:
        super().__init__(message)
        self.error_code = error_code


class StorageOperationCancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class ResticResult:
    status: str
    result_quality: OperationResultQuality | None
    snapshot_id: str | None
    exit_code: int
    error_code: str | None = None


@dataclass(frozen=True)
class RestoreStats:
    total_size: int
    total_file_count: int


class JsonEventWriter:
    def __init__(
        self,
        operation_id: str,
        progress_interval_seconds: float,
        heartbeat_interval_seconds: float = 30.0,
        output: IO[str] = sys.stdout,
        clock: Callable[[], float] = time.monotonic,
        reporter: StorageEventReporter | None = None,
        cancellation_probe: Callable[[], bool] | None = None,
    ) -> None:
        self._operation_id = operation_id
        self._progress_interval_seconds = progress_interval_seconds
        self._heartbeat_interval_seconds = heartbeat_interval_seconds
        self._output = output
        self._clock = clock
        self._last_progress_at: float | None = None
        self._last_heartbeat_at = self._clock()
        self._reporter = reporter
        self._cancellation_probe = cancellation_probe

    def restic_event(self, payload: Mapping[str, object]) -> None:
        message_type = payload.get("message_type")
        if message_type == "status" and not self._progress_due():
            return
        self._write(
            {"event": "restic", "operation_id": self._operation_id, "payload": dict(payload)}
        )

    def diagnostic(self, message: str) -> None:
        self._write({"event": "diagnostic", "operation_id": self._operation_id, "message": message})

    def stage(self, stage: str) -> None:
        self._write({"event": "stage", "operation_id": self._operation_id, "stage": stage})

    def heartbeat_if_due(self) -> None:
        now = self._clock()
        if now - self._last_heartbeat_at < self._heartbeat_interval_seconds:
            return
        self._last_heartbeat_at = now
        self._write({"event": "heartbeat", "operation_id": self._operation_id})

    @property
    def heartbeat_interval_seconds(self) -> float:
        return self._heartbeat_interval_seconds

    @property
    def cancellation_requested(self) -> bool:
        return bool(
            (self._reporter and self._reporter.cancel_requested)
            or (self._cancellation_probe and self._cancellation_probe())
        )

    def result(self, result: ResticResult) -> None:
        self._write(
            {
                "event": "result",
                "operation_id": self._operation_id,
                "status": result.status,
                "result_quality": result.result_quality.value if result.result_quality else None,
                "snapshot_id": result.snapshot_id,
                "exit_code": result.exit_code,
                "error_code": result.error_code,
            }
        )

    def _progress_due(self) -> bool:
        now = self._clock()
        if (
            self._last_progress_at is None
            or now - self._last_progress_at >= self._progress_interval_seconds
        ):
            self._last_progress_at = now
            return True
        return False

    def _write(self, payload: Mapping[str, object]) -> None:
        self._output.write(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")
        self._output.flush()
        if self._reporter:
            self._reporter.send(payload)


class ResticStorageRunner:
    def __init__(
        self,
        operation: StorageOperationSpec,
        workspace: ResolvedWorkspace,
        restic_binary: str = "/usr/local/bin/restic",
        docker_binary: str = "/usr/bin/docker",
        event_writer: JsonEventWriter | None = None,
    ) -> None:
        self._operation = operation
        self._workspace = workspace
        self._restic_binary = restic_binary
        self._docker_binary = docker_binary
        self._events = event_writer or JsonEventWriter(
            str(operation.operation_id),
            operation.progress_interval_seconds,
            operation.heartbeat_interval_seconds,
        )
        self._environment = self._build_environment()

    def run(self) -> ResticResult:
        if self._operation.engine is StorageEngine.RESTIC:
            if self._operation.action is StorageAction.BACKUP:
                self._ensure_repository_for_backup()
            else:
                self._require_repository_for_restore()
        if self._operation.action is StorageAction.BACKUP:
            result = self._backup()
        else:
            result = self._restore()
        self._events.result(result)
        return result

    def _ensure_repository_for_backup(self) -> None:
        repository_readiness_deadline = (
            time.monotonic() + REPOSITORY_READINESS_TIMEOUT_SECONDS
        )
        repository_probe = self._run_repository_command_with_retry(
            ["snapshots", "--json"],
            repository_command_name="probe",
            retry_deadline=repository_readiness_deadline,
            accepted_exit_codes=(0, 10),
        )
        if repository_probe.returncode == 0:
            return
        if repository_probe.returncode != 10:
            failure_detail = self._redacted_repository_failure_detail(repository_probe)
            raise ResticOperationError(
                "restic repository probe failed with exit "
                f"{repository_probe.returncode}: {failure_detail}"
            )

        repository_initialization = self._run_repository_command_with_retry(
            ["init", "--json"],
            repository_command_name="initialization",
            retry_deadline=repository_readiness_deadline,
        )
        if repository_initialization.returncode == 0:
            return

        # Initialization may have succeeded even if its response was lost. Probe
        # again before retrying so we never initialize an existing repository.
        repository_confirmation = self._run_repository_command_with_retry(
            ["snapshots", "--json"],
            repository_command_name="post-initialization probe",
            retry_deadline=repository_readiness_deadline,
            accepted_exit_codes=(0, 10),
        )
        if repository_confirmation.returncode != 0:
            failure_detail = self._redacted_repository_failure_detail(repository_initialization)
            raise ResticOperationError(
                f"restic repository initialization failed: {failure_detail}"
            )

    def _run_repository_command_with_retry(
        self,
        restic_arguments: list[str],
        *,
        repository_command_name: str,
        retry_deadline: float,
        accepted_exit_codes: tuple[int, ...] = (0,),
    ) -> subprocess.CompletedProcess[str]:
        retry_delay_seconds = REPOSITORY_RETRY_INITIAL_DELAY_SECONDS
        while True:
            self._raise_if_cancellation_requested()
            repository_command_result = subprocess.run(
                self._restic_command(restic_arguments),
                env=self._environment,
                capture_output=True,
                text=True,
            )
            if repository_command_result.returncode in accepted_exit_codes:
                return repository_command_result

            failure_detail = self._redacted_repository_failure_detail(repository_command_result)
            if not _is_retryable_repository_failure(failure_detail):
                return repository_command_result
            remaining_retry_seconds = retry_deadline - time.monotonic()
            if remaining_retry_seconds <= 0:
                self._log_transient_repository_failure(
                    repository_command_name,
                    failure_detail,
                    retry_delay_seconds=None,
                )
                raise ResticOperationError(
                    "Backup storage was not ready after retrying: "
                    f"{failure_detail}"
                ) from None
            wait_seconds = min(retry_delay_seconds, remaining_retry_seconds)
            self._log_transient_repository_failure(
                repository_command_name,
                failure_detail,
                retry_delay_seconds=wait_seconds,
            )
            self._wait_before_repository_retry(wait_seconds)
            retry_delay_seconds = min(
                retry_delay_seconds * 2,
                REPOSITORY_RETRY_MAX_DELAY_SECONDS,
            )

    def _redacted_repository_failure_detail(
        self,
        repository_command_result: subprocess.CompletedProcess[str],
    ) -> str:
        command_output = repository_command_result.stderr or repository_command_result.stdout
        return _redact(command_output, self._secret_values())

    def _wait_before_repository_retry(self, delay_seconds: float) -> None:
        deadline = time.monotonic() + delay_seconds
        while True:
            self._raise_if_cancellation_requested()
            self._events.heartbeat_if_due()
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                return
            time.sleep(min(REPOSITORY_RETRY_POLL_SECONDS, remaining_seconds))

    def _raise_if_cancellation_requested(self) -> None:
        if self._events.cancellation_requested:
            raise StorageOperationCancelled("storage operation cancellation requested")

    def _log_transient_repository_failure(
        self,
        repository_command_name: str,
        failure_detail: str,
        *,
        retry_delay_seconds: float | None,
    ) -> None:
        retry_status = (
            f"; retrying in {retry_delay_seconds:g}s"
            if retry_delay_seconds is not None
            else "; retries exhausted"
        )
        print(
            "Transient repository "
            f"{repository_command_name} failure{retry_status}: {failure_detail}",
            file=sys.stderr,
            flush=True,
        )

    def _require_repository_for_restore(self) -> None:
        # Restore is read-only: a missing source must fail instead of creating an
        # empty repository at the requested prefix.
        snapshot_id = self._operation.snapshot_id
        if not snapshot_id:
            raise ResticOperationError("restore requires a snapshot ID")
        probe = subprocess.run(
            self._restic_command(["snapshots", "--json", snapshot_id]),
            env=self._environment,
            capture_output=True,
            text=True,
        )
        if probe.returncode == 10:
            raise ResticOperationError(
                "backup repository does not exist",
                error_code="RESTIC_REPOSITORY_MISSING",
            )
        if probe.returncode != 0:
            detail = _redact(probe.stderr or probe.stdout, self._secret_values())
            raise ResticOperationError(
                f"backup repository probe failed with exit {probe.returncode}: {detail}"
            )
        try:
            snapshots = json.loads(probe.stdout)
        except json.JSONDecodeError as error:
            raise ResticOperationError("backup repository probe returned invalid JSON") from error
        if not isinstance(snapshots, list) or not any(
            isinstance(snapshot, dict)
            and isinstance(snapshot.get("id"), str)
            and snapshot["id"].startswith(snapshot_id)
            for snapshot in snapshots
        ):
            raise ResticOperationError(
                f"backup snapshot {snapshot_id} does not exist",
                error_code="RESTIC_SNAPSHOT_MISSING",
            )

    def _backup(self) -> ResticResult:
        if self._operation.engine is not StorageEngine.RESTIC:
            raise ResticOperationError("new backups require the restic engine")
        command = [
            "backup",
            "--json",
            "--host",
            f"lium-pod-{self._operation.pod_id}",
            "--tag",
            f"lium-operation:{self._operation.operation_id}",
            ".",
        ]
        exit_code, summary = self._stream(command, working_directory=True)
        snapshot_id = _snapshot_id(summary)
        if exit_code == 0 and snapshot_id:
            return ResticResult("COMPLETED", OperationResultQuality.FULL, snapshot_id, exit_code)
        if exit_code == 3 and snapshot_id:
            return ResticResult("COMPLETED", OperationResultQuality.PARTIAL, snapshot_id, exit_code)
        raise ResticOperationError(
            f"restic backup failed with exit {exit_code}"
            + (" without a snapshot ID" if not snapshot_id else "")
        )

    def _restore(self) -> ResticResult:
        if self._operation.engine is StorageEngine.TAR_AWS_CLI:
            return self._restore_legacy_archive()
        snapshot_id = self._operation.snapshot_id
        if not snapshot_id:
            raise ResticOperationError("restore requires a snapshot ID")
        command, cwd = self._restore_execution_command(snapshot_id)
        exit_code, _ = self._stream_command(command, cwd)
        if exit_code != 0:
            raise ResticOperationError(f"restic restore failed with exit {exit_code}")
        self._events.stage("FINALIZING")
        return ResticResult("COMPLETED", OperationResultQuality.FULL, snapshot_id, exit_code)

    def _restore_legacy_archive(self) -> ResticResult:
        object_key = self._operation.legacy_object_key
        if not object_key:
            raise ResticOperationError("legacy restore requires an object key")
        restore_stats = RestoreStats(
            total_size=self._operation.legacy_object_size_bytes or 0,
            total_file_count=0,
        )
        command, cwd = self._legacy_restore_execution_command(object_key)
        exit_code, _ = self._stream_command(command, cwd, restore_stats=restore_stats)
        if exit_code != 0:
            raise ResticOperationError(f"legacy restore failed with exit {exit_code}")
        self._events.restic_event(
            {
                "message_type": "summary",
                "percent_done": 1,
                "total_files": 0,
                "files_restored": 0,
                "total_bytes": restore_stats.total_size,
                "bytes_restored": restore_stats.total_size,
            }
        )
        self._events.stage("FINALIZING")
        return ResticResult("COMPLETED", OperationResultQuality.FULL, None, exit_code)

    def _stream(
        self,
        restic_arguments: list[str],
        working_directory: bool,
    ) -> tuple[int, Mapping[str, object] | None]:
        command, cwd = self._execution_command(restic_arguments, working_directory)
        return self._stream_command(command, cwd)

    def _stream_command(
        self,
        command: list[str],
        cwd: str | None,
        *,
        restore_stats: RestoreStats | None = None,
    ) -> tuple[int, Mapping[str, object] | None]:
        started_at = time.monotonic()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=self._environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        summary: Mapping[str, object] | None = None
        if process.stdout is None:
            process.kill()
            raise ResticOperationError("restic output stream was not created")

        output_lines: queue.Queue[str | None] = queue.Queue()

        def read_output() -> None:
            try:
                for raw_line in process.stdout:
                    output_lines.put(raw_line)
            finally:
                output_lines.put(None)

        output_reader = threading.Thread(target=read_output, name="restic-output", daemon=True)
        output_reader.start()

        try:
            while True:
                try:
                    raw_line = output_lines.get(
                        timeout=min(self._events.heartbeat_interval_seconds, 1.0)
                    )
                except queue.Empty:
                    self._events.heartbeat_if_due()
                    self._cancel_if_requested(process)
                    continue
                if raw_line is None:
                    break
                line = raw_line.strip()
                if not line:
                    continue
                checkpoint = _restore_checkpoint(line)
                if checkpoint is not None and restore_stats is not None:
                    self._events.restic_event(
                        _restore_progress(checkpoint, restore_stats, time.monotonic() - started_at)
                    )
                    self._events.heartbeat_if_due()
                    self._cancel_if_requested(process)
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    self._events.diagnostic(_redact(line, self._secret_values()))
                    continue
                if not isinstance(payload, dict):
                    self._events.diagnostic("restic emitted a non-object JSON message")
                    continue
                self._events.restic_event(payload)
                self._events.heartbeat_if_due()
                self._cancel_if_requested(process)
                if payload.get("message_type") == "summary":
                    summary = payload

            exit_code = process.wait()
            return exit_code, summary
        except StorageOperationCancelled:
            raise
        except Exception:
            self._terminate_process(process)
            raise
        finally:
            output_reader.join(timeout=1)

    def _cancel_if_requested(self, process: subprocess.Popen[str]) -> None:
        if not self._events.cancellation_requested:
            return
        self._terminate_process(process)
        raise StorageOperationCancelled("storage operation cancellation requested")

    def _terminate_process(self, process: subprocess.Popen[str]) -> None:
        self._stop_helper_container()
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)

    def _stop_helper_container(self) -> None:
        subprocess.run(
            [self._docker_binary, "stop", "--time", "5", self._helper_container_name()],
            capture_output=True,
            text=True,
        )

    def _execution_command(
        self,
        restic_arguments: list[str],
        working_directory: bool,
    ) -> tuple[list[str], str | None]:
        restic_command = self._restic_command(restic_arguments)
        if isinstance(self._workspace, LocalWorkspace):
            cwd = str(self._workspace.path) if working_directory else None
            return restic_command, cwd

        if isinstance(self._workspace, DockerUserNamespaceWorkspace):
            if working_directory:
                namespace_command = [
                    "/bin/sh",
                    "-c",
                    ENCRYPTED_BACKUP_SCRIPT,
                    "sh",
                    str(self._workspace.path),
                    *restic_command,
                ]
            else:
                namespace_command = restic_command
            return self._encrypted_helper_command(namespace_command), None

        if isinstance(self._workspace, DockerEncryptedVolumeWorkspace):
            return self._encrypted_volume_helper_command(restic_command, working_directory), None

        volume_mode = "ro" if self._workspace.read_only else "rw"
        command = [
            *self._docker_helper_base(),
            "--security-opt",
            "no-new-privileges",
            "-v",
            f"{self._workspace.volume_name}:/workspace:{volume_mode}",
        ]
        if working_directory:
            command.extend(["--workdir", str(self._workspace.path)])
        command.extend(
            [
                "--entrypoint",
                self._restic_binary,
                self._workspace.image,
                *restic_command[1:],
            ]
        )
        return command, None

    def _restore_execution_command(self, snapshot_id: str) -> tuple[list[str], str | None]:
        arguments = [
            "restore",
            "--json",
            "--target",
            self._workspace_path(),
            "--sparse",
        ]
        if isinstance(self._workspace, DockerUserNamespaceWorkspace):
            # User xattrs belong to customer data; only host-managed namespaces
            # that cannot be written from the rental user namespace are skipped.
            arguments.extend(["--exclude-xattr", "security.*", "--exclude-xattr", "trusted.*"])
        arguments.append(snapshot_id)
        return self._execution_command(arguments, working_directory=False)

    def _legacy_restore_execution_command(self, object_key: str) -> tuple[list[str], str | None]:
        pipeline_script = (
            'mkdir -p -- "$4"; '
            '"$1" s3 cp "s3://$2/$3" - --no-progress | '
            'tar --extract --gzip --file - --directory "$4" --strip-components=1 '
            "--preserve-permissions --same-owner --numeric-owner --xattrs --acls "
            f"--blocking-factor=20 --checkpoint={TAR_CHECKPOINT_RECORDS} "
            f"--checkpoint-action=echo={RESTORE_CHECKPOINT_PREFIX}%u"
        )
        pipeline_command = [
            "/bin/bash",
            "-o",
            "pipefail",
            "-c",
            pipeline_script,
            "bash",
            "/usr/bin/aws",
            self._operation.repository.bucket,
            object_key,
            self._workspace_path(),
        ]
        if isinstance(self._workspace, LocalWorkspace):
            return pipeline_command, None
        if isinstance(self._workspace, DockerUserNamespaceWorkspace):
            return self._encrypted_helper_command(pipeline_command), None
        if isinstance(self._workspace, DockerEncryptedVolumeWorkspace):
            return self._encrypted_volume_helper_command(pipeline_command, False), None
        return [
            *self._docker_helper_base(),
            "--security-opt",
            "no-new-privileges",
            "-v",
            f"{self._workspace.volume_name}:/workspace:rw",
            "--entrypoint",
            "/bin/bash",
            self._workspace.image,
            *pipeline_command[1:],
        ], None

    def _encrypted_helper_command(self, namespace_command: list[str]) -> list[str]:
        if not isinstance(self._workspace, DockerUserNamespaceWorkspace):
            raise ResticOperationError("encrypted helper requested for a non-encrypted workspace")
        return [
            *self._docker_helper_base(),
            "--pid",
            "host",
            "--privileged",
            "--security-opt",
            "label=disable",
            "--entrypoint",
            "/usr/bin/nsenter",
            self._workspace.image,
            "-t",
            str(self._workspace.pid),
            "-U",
            "--",
            *namespace_command,
        ]

    def _encrypted_volume_helper_command(
        self,
        command: list[str],
        working_directory: bool,
    ) -> list[str]:
        if not isinstance(self._workspace, DockerEncryptedVolumeWorkspace):
            raise ResticOperationError(
                "encrypted volume helper requested for a different workspace"
            )
        wrapped_command = command
        if working_directory:
            wrapped_command = [
                "/bin/sh",
                "-c",
                ENCRYPTED_BACKUP_SCRIPT,
                "sh",
                str(self._workspace.path),
                *command,
            ]
        return [
            *self._docker_helper_base(),
            "--privileged",
            "--security-opt",
            "label=disable",
            "--device",
            "/dev/fuse:/dev/fuse",
            "--tmpfs",
            "/workspace:rw,nosuid,nodev,size=67108864",
            "-v",
            f"{self._workspace.volume_name}:/lium-cipher:rw",
            "--entrypoint",
            "/bin/bash",
            self._workspace.image,
            "-c",
            ENCRYPTED_BOOTSTRAP_SCRIPT,
            "bash",
            *wrapped_command,
        ]

    def _docker_helper_base(self) -> list[str]:
        return [
            self._docker_binary,
            "run",
            "--rm",
            "--name",
            self._helper_container_name(),
            "--log-driver",
            "none",
            "--read-only",
            "--tmpfs",
            f"/tmp:rw,nosuid,nodev,size={RESTIC_TMPFS_BYTES}",
            "-e",
            "AWS_ACCESS_KEY_ID",
            "-e",
            "AWS_SECRET_ACCESS_KEY",
            "-e",
            "AWS_SESSION_TOKEN",
            "-e",
            "AWS_DEFAULT_REGION",
            "-e",
            "RESTIC_REPOSITORY",
            "-e",
            "RESTIC_PASSWORD",
            "-e",
            "LIUM_VOLUME_PASSPHRASE",
        ]

    def _helper_container_name(self) -> str:
        return f"lium-storage-{str(self._operation.operation_id)[:12]}"

    def _restic_command(self, arguments: list[str]) -> list[str]:
        # Restic's native concurrency avoided the intermittent S3 transfer tails
        # observed when Lium forced 64 connections.
        return [
            self._restic_binary,
            "--no-cache",
            *arguments,
        ]

    def _workspace_path(self) -> str:
        return str(self._workspace.path)

    def _build_environment(self) -> dict[str, str]:
        repository = self._operation.repository
        environment = os.environ.copy()
        environment.update(
            {
                "AWS_ACCESS_KEY_ID": repository.access_key_id,
                "AWS_SECRET_ACCESS_KEY": repository.secret_access_key,
                "AWS_DEFAULT_REGION": repository.region,
                "RESTIC_REPOSITORY": repository.url_for_pod(self._operation.repository_pod_id),
            }
        )
        if repository.password:
            environment["RESTIC_PASSWORD"] = repository.password
        else:
            environment.pop("RESTIC_PASSWORD", None)
        if isinstance(self._workspace, DockerEncryptedVolumeWorkspace):
            environment["LIUM_VOLUME_PASSPHRASE"] = self._workspace.volume_passphrase
        else:
            environment.pop("LIUM_VOLUME_PASSPHRASE", None)
        if repository.session_token:
            environment["AWS_SESSION_TOKEN"] = repository.session_token
        else:
            environment.pop("AWS_SESSION_TOKEN", None)
        return environment

    def _secret_values(self) -> tuple[str, ...]:
        repository = self._operation.repository
        return (
            repository.access_key_id,
            repository.secret_access_key,
            repository.session_token or "",
            repository.password or "",
            self._workspace.volume_passphrase
            if isinstance(self._workspace, DockerEncryptedVolumeWorkspace)
            else "",
        )


def _is_retryable_repository_failure(failure_detail: str) -> bool:
    normalized_detail = failure_detail.casefold().replace(" ", "")
    return IAM_POLICY_PROPAGATION_ERROR_MARKER in normalized_detail


def _snapshot_id(summary: Mapping[str, object] | None) -> str | None:
    if not summary:
        return None
    value = summary.get("snapshot_id")
    return value if isinstance(value, str) and value else None


def _restore_checkpoint(line: str) -> int | None:
    marker_index = line.find(RESTORE_CHECKPOINT_PREFIX)
    if marker_index < 0:
        return None
    value = line[marker_index + len(RESTORE_CHECKPOINT_PREFIX) :].strip()
    try:
        checkpoint = int(value)
    except ValueError:
        return None
    return checkpoint if checkpoint >= 0 else None


def _restore_progress(
    checkpoint: int,
    stats: RestoreStats,
    seconds_elapsed: float,
) -> dict[str, object]:
    archive_bytes = checkpoint * TAR_RECORD_SIZE_BYTES
    if stats.total_size == 0:
        percent_done = 0.99
        bytes_restored = 0
    else:
        percent_done = min(archive_bytes / stats.total_size, 0.99)
        bytes_restored = min(archive_bytes, stats.total_size)
    return {
        "message_type": "status",
        "seconds_elapsed": int(seconds_elapsed),
        "percent_done": percent_done,
        "total_files": stats.total_file_count,
        "total_bytes": stats.total_size,
        "bytes_restored": bytes_restored,
    }


def _redact(value: str, secrets: tuple[str, ...]) -> str:
    for secret in secrets:
        if secret:
            value = value.replace(secret, "[REDACTED]")
    return _bounded_message(value)


def _bounded_message(value: str, limit: int = 2000) -> str:
    compact = " ".join(value.split())
    if len(compact) <= limit:
        return compact
    return compact[:limit] + "…"
