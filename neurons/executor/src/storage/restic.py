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
from storage.workspace import DockerVolumeWorkspace, LocalWorkspace, ResolvedWorkspace


class ResticOperationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ResticResult:
    status: str
    result_quality: OperationResultQuality | None
    snapshot_id: str | None
    exit_code: int


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
            [self._restic_binary, "--no-cache", "snapshots", "--json"],
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
            [self._restic_binary, "init", "--json"],
            env=self._environment,
            capture_output=True,
            text=True,
        )
        if initialized.returncode == 0:
            return

        retry_probe = subprocess.run(
            [self._restic_binary, "--no-cache", "snapshots", "--json"],
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
        command = ["restore", "--json", f"{snapshot_id}:/", "--target", self._workspace_path()]
        exit_code, _ = self._stream(command, working_directory=False)
        if exit_code != 0:
            raise ResticOperationError(f"restic restore failed with exit {exit_code}")
        return ResticResult("COMPLETED", OperationResultQuality.FULL, snapshot_id, exit_code)

    def _stream(
        self,
        restic_arguments: list[str],
        working_directory: bool,
    ) -> tuple[int, Mapping[str, object] | None]:
        command, cwd = self._execution_command(restic_arguments, working_directory)
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
        if isinstance(self._workspace, LocalWorkspace):
            cwd = str(self._workspace.path) if working_directory else None
            return [self._restic_binary, "--no-cache", *restic_arguments], cwd

        volume_mode = "ro" if self._workspace.read_only else "rw"
        container_name = f"lium-storage-{str(self._operation.operation_id)[:12]}"
        command = [
            self._docker_binary,
            "run",
            "--rm",
            "--name",
            container_name,
            "--log-driver",
            "none",
            "--read-only",
            "--security-opt",
            "no-new-privileges",
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
                "--no-cache",
                *restic_arguments,
            ]
        )
        return command, None

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
