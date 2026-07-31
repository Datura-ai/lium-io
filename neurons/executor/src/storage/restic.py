from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import IO, Callable, Mapping

from storage.models import OperationResultQuality, StorageAction, StorageOperationSpec
from storage.workspace import (
    DockerUserNamespaceWorkspace,
    DockerVolumeWorkspace,
    LocalWorkspace,
    ResolvedWorkspace,
)


RESTORE_CHECKPOINT_PREFIX = "LIUM_RESTORE_CHECKPOINT_"
TAR_RECORD_SIZE_BYTES = 20 * 512
TAR_CHECKPOINT_RECORDS = 1024
ENCRYPTED_BACKUP_SCRIPT = 'cd "$1"; shift; exec "$@"'


def _restore_pipeline_script(*, extended_metadata: bool) -> str:
    metadata_options = '--acls --xattrs --xattrs-include="*" ' if extended_metadata else ""
    return (
        'mkdir -p -- "$4"; '
        '"$1" --no-cache -o "$2" dump --archive tar "$3" / | '
        'tar --extract --file - --directory "$4" --preserve-permissions '
        f'--same-owner --numeric-owner {metadata_options}'
        f'--blocking-factor=20 --checkpoint={TAR_CHECKPOINT_RECORDS} '
        f'--checkpoint-action=echo={RESTORE_CHECKPOINT_PREFIX}%u'
    )


class ResticOperationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ResticResult:
    status: str
    result_quality: OperationResultQuality | None
    snapshot_id: str | None
    exit_code: int


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
    ) -> None:
        self._operation_id = operation_id
        self._progress_interval_seconds = progress_interval_seconds
        self._heartbeat_interval_seconds = heartbeat_interval_seconds
        self._output = output
        self._clock = clock
        self._last_progress_at: float | None = None
        self._last_heartbeat_at = self._clock()

    def restic_event(self, payload: Mapping[str, object]) -> None:
        message_type = payload.get("message_type")
        if message_type == "status" and not self._progress_due():
            return
        self._write({"event": "restic", "operation_id": self._operation_id, "payload": dict(payload)})

    def diagnostic(self, message: str) -> None:
        self._write({"event": "diagnostic", "operation_id": self._operation_id, "message": message})

    def heartbeat_if_due(self) -> None:
        now = self._clock()
        if now - self._last_heartbeat_at < self._heartbeat_interval_seconds:
            return
        self._last_heartbeat_at = now
        self._write({"event": "heartbeat", "operation_id": self._operation_id})

    @property
    def heartbeat_interval_seconds(self) -> float:
        return self._heartbeat_interval_seconds

    def result(self, result: ResticResult) -> None:
        self._write(
            {
                "event": "result",
                "operation_id": self._operation_id,
                "status": result.status,
                "result_quality": result.result_quality.value if result.result_quality else None,
                "snapshot_id": result.snapshot_id,
                "exit_code": result.exit_code,
            }
        )

    def _progress_due(self) -> bool:
        now = self._clock()
        if self._last_progress_at is None or now - self._last_progress_at >= self._progress_interval_seconds:
            self._last_progress_at = now
            return True
        return False

    def _write(self, payload: Mapping[str, object]) -> None:
        self._output.write(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")
        self._output.flush()


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
        self._ensure_repository()
        if self._operation.action is StorageAction.BACKUP:
            result = self._backup()
        else:
            result = self._restore()
        self._events.result(result)
        return result

    def _ensure_repository(self) -> None:
        probe = subprocess.run(
            self._restic_command(["snapshots", "--json"]),
            env=self._environment,
            capture_output=True,
            text=True,
        )
        if probe.returncode == 0:
            return
        if probe.returncode != 10:
            detail = _redact(probe.stderr or probe.stdout, self._secret_values())
            raise ResticOperationError(f"restic repository probe failed with exit {probe.returncode}: {detail}")

        initialized = subprocess.run(
            self._restic_command(["init", "--json"]),
            env=self._environment,
            capture_output=True,
            text=True,
        )
        if initialized.returncode == 0:
            return

        retry_probe = subprocess.run(
            self._restic_command(["snapshots", "--json"]),
            env=self._environment,
            capture_output=True,
            text=True,
        )
        if retry_probe.returncode != 0:
            detail = _redact(initialized.stderr or initialized.stdout, self._secret_values())
            raise ResticOperationError(f"restic repository initialization failed: {detail}")

    def _backup(self) -> ResticResult:
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
        snapshot_id = self._operation.snapshot_id
        if not snapshot_id:
            raise ResticOperationError("restore requires a snapshot ID")
        restore_stats = self._restore_stats(snapshot_id)
        command, cwd = self._restore_execution_command(snapshot_id)
        exit_code, _ = self._stream_command(command, cwd, restore_stats=restore_stats)
        if exit_code != 0:
            raise ResticOperationError(f"restic restore failed with exit {exit_code}")
        self._events.restic_event(
            {
                "message_type": "summary",
                "percent_done": 1,
                "total_files": restore_stats.total_file_count,
                "files_restored": restore_stats.total_file_count,
                "total_bytes": restore_stats.total_size,
                "bytes_restored": restore_stats.total_size,
            }
        )
        return ResticResult("COMPLETED", OperationResultQuality.FULL, snapshot_id, exit_code)

    def _restore_stats(self, snapshot_id: str) -> RestoreStats:
        result = subprocess.run(
            self._restic_command(["stats", "--json", "--mode", "restore-size", snapshot_id]),
            env=self._environment,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            detail = _redact(result.stderr or result.stdout, self._secret_values())
            raise ResticOperationError(
                f"restic restore size lookup failed with exit {result.returncode}: {detail}"
            )
        try:
            payload = json.loads(result.stdout)
            total_size = payload["total_size"]
            total_file_count = payload["total_file_count"]
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise ResticOperationError("restic restore size lookup returned invalid JSON") from error
        if not isinstance(total_size, int) or total_size < 0:
            raise ResticOperationError("restic restore size lookup returned an invalid total_size")
        if not isinstance(total_file_count, int) or total_file_count < 0:
            raise ResticOperationError("restic restore size lookup returned an invalid total_file_count")
        return RestoreStats(total_size=total_size, total_file_count=total_file_count)

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

        while True:
            try:
                raw_line = output_lines.get(timeout=self._events.heartbeat_interval_seconds)
            except queue.Empty:
                self._events.heartbeat_if_due()
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
            if payload.get("message_type") == "summary":
                summary = payload

        exit_code = process.wait()
        output_reader.join(timeout=1)
        return exit_code, summary

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
        extended_metadata = not isinstance(self._workspace, DockerUserNamespaceWorkspace)
        pipeline_command = [
            "/bin/bash",
            "-o",
            "pipefail",
            "-c",
            _restore_pipeline_script(extended_metadata=extended_metadata),
            "bash",
            self._restic_binary,
            self._s3_connection_option(),
            snapshot_id,
            self._workspace_path(),
        ]
        if isinstance(self._workspace, LocalWorkspace):
            return pipeline_command, None
        if isinstance(self._workspace, DockerUserNamespaceWorkspace):
            return self._encrypted_helper_command(pipeline_command), None
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

    def _docker_helper_base(self) -> list[str]:
        return [
            self._docker_binary,
            "run",
            "--rm",
            "--name",
            f"lium-storage-{str(self._operation.operation_id)[:12]}",
            "--log-driver",
            "none",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=268435456",
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
            "RESTIC_HOST",
        ]

    def _restic_command(self, arguments: list[str]) -> list[str]:
        return [
            self._restic_binary,
            "--no-cache",
            "-o",
            self._s3_connection_option(),
            *arguments,
        ]

    def _s3_connection_option(self) -> str:
        return f"s3.connections={self._operation.repository.s3_connections}"

    def _workspace_path(self) -> str:
        if isinstance(self._workspace, LocalWorkspace):
            return str(self._workspace.path)
        return str(self._workspace.path)

    def _build_environment(self) -> dict[str, str]:
        repository = self._operation.repository
        environment = os.environ.copy()
        environment.update(
            {
                "AWS_ACCESS_KEY_ID": repository.access_key_id,
                "AWS_SECRET_ACCESS_KEY": repository.secret_access_key,
                "AWS_DEFAULT_REGION": repository.region,
                "RESTIC_REPOSITORY": repository.url_for_pod(self._operation.pod_id),
                "RESTIC_PASSWORD": repository.password,
                "RESTIC_HOST": f"lium-pod-{self._operation.pod_id}",
            }
        )
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
            repository.password,
        )


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
