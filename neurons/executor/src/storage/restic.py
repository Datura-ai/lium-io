from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import IO

from storage.models import (
    OperationResultQuality,
    StorageAction,
    StorageEngine,
    StorageOperationSpec,
)
from storage.reporting import StorageEventReporter
from storage.workspace import (
    DockerUserNamespaceWorkspace,
    LocalWorkspace,
    ResolvedWorkspace,
)

RESTORE_CHECKPOINT_PREFIX = "LIUM_RESTORE_CHECKPOINT_"
TAR_RECORD_SIZE_BYTES = 20 * 512
TAR_CHECKPOINT_RECORDS = 1024
MEBIBYTE = 1024 * 1024
RESTIC_TMPFS_BYTES = 512 * MEBIBYTE
# DAH-2002: bound the helper so a backup cannot OOM the rental. Measured restic 0.19.1 peak RSS: 0.41 GiB at
# 2M files, ~125 B per extra file; bytes barely matter (20 GiB file: 0.17 GiB). 4g covers ~30M files.
# restic sizes its worker pools from GOMAXPROCS (~7.5 MiB per CPU); unpinned, 192+ CPU hosts hit 4g in seconds.
RESTIC_HELPER_GOMAXPROCS = "8"
RESTIC_HELPER_MEMORY = "4g"
# Highest OOM score: the kernel kills the helper before the rental.
RESTIC_HELPER_OOM_SCORE_ADJ = "1000"
RESTIC_HELPER_GOGC = "50"
SIGKILL_EXIT = 137
HELPER_LABEL = "lium.backup-helper"
STALE_HELPER_AGE = timedelta(minutes=10)
# Without `|| exit`, a failed cd leaves sh in the helper's own directory and restic backs that up instead.
# sh's own cd error is dropped: StorageEventReporter keeps the first diagnostic, which must be ours.
# Runs the backup inside a helper (docker volume or encrypted namespace). A missing or empty path exits 66 before
# restic starts, so nothing is uploaded (DAH-2002); a path that is not a directory fails.
BACKUP_SOURCE_SCRIPT = '[ -e "$1" ] || exit 66; cd "$1" 2>/dev/null || exit 1; [ -n "$(ls -A)" ] || exit 66; shift; exec "$@"'
BACKUP_SOURCE_MISSING_EXIT = 66

# restic 0.19.1 prints this as the exit_error JSON "message" with --json, else as a plain stderr line
SNAPSHOT_EMPTY_MESSAGE = "Fatal: unable to save snapshot: snapshot is empty"

class ResticOperationError(RuntimeError):
    def __init__(self, message: str, *, error_code: str | None = None) -> None:
        super().__init__(message)
        self.error_code = error_code


class StorageOperationCancelled(RuntimeError):
    pass


class CredentialsExpired(ResticOperationError):
    pass


# DAH-4945: the backend renews scoped credentials up to 4 sessions' worth after the backup started.
CREDENTIAL_RENEWALS = 3


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
    def reporter(self) -> StorageEventReporter | None:
        return self._reporter

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
        self._snapshot_empty = False
        self._helper_oom_killed = False

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
        # A newly attached IAM policy can briefly return Access Denied while AWS
        # propagates it. Share one three-minute readiness window across setup.
        repository_readiness_deadline = time.monotonic() + 180.0
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
        # Start with a short delay so newly propagated permissions recover quickly.
        retry_delay_seconds = 2.0
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
            # Avoid hammering S3 while still checking readiness at least every 30 seconds.
            retry_delay_seconds = min(retry_delay_seconds * 2, 30.0)

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
            # Wake every second so cancellation and heartbeats remain responsive.
            time.sleep(min(1.0, remaining_seconds))

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
            self._restic_command(["snapshots", "--json", "--", snapshot_id]),
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
            *(f"--exclude={pattern}" for pattern in self._operation.workspace.exclude_patterns),
            ".",
        ]
        # DAH-2002/DAH-4693: a missing, empty or fully excluded path is a no-op, not a failed run
        nothing = ResticResult("COMPLETED", OperationResultQuality.EMPTY, None, 0)
        if isinstance(self._workspace, LocalWorkspace):
            path = self._workspace.path
            if path.exists() and not path.is_dir():
                raise ResticOperationError("the backup path is not a directory")
            if not path.exists() or not any(path.iterdir()):
                return nothing
        renewals = 0
        while True:
            try:
                exit_code, summary = self._stream(command, working_directory=True)
                break
            except CredentialsExpired:
                if renewals >= CREDENTIAL_RENEWALS:
                    raise
                renewals += 1
                # restic resumes from the blobs its repository already indexed
                self._renew_credentials()
        snapshot_id = _snapshot_id(summary)
        # DAH-4693: an empty snapshot is a no-op only when excludes removed every file
        snapshot_empty = self._snapshot_empty and bool(self._operation.workspace.exclude_patterns)
        if (exit_code == BACKUP_SOURCE_MISSING_EXIT or snapshot_empty) and not snapshot_id:
            return nothing
        if exit_code == 0 and snapshot_id:
            return ResticResult("COMPLETED", OperationResultQuality.FULL, snapshot_id, exit_code)
        if exit_code == 3 and snapshot_id:
            return ResticResult("COMPLETED", OperationResultQuality.PARTIAL, snapshot_id, exit_code)
        if exit_code == SIGKILL_EXIT:
            raise ResticOperationError(
                "backup ran out of memory"
                if self._helper_oom_killed
                else "backup helper was killed (often out of memory)"
            )
        raise ResticOperationError(
            f"restic backup failed with exit {exit_code}"
            + (" without a snapshot ID" if not snapshot_id else "")
        )

    def _renew_credentials(self) -> None:
        reporter = self._events.reporter
        credentials = reporter.renew_credentials() if reporter else None
        if not credentials:
            raise ResticOperationError("storage credentials expired and could not be renewed")
        self._operation = replace(
            self._operation, repository=replace(self._operation.repository, **credentials)
        )
        self._environment = self._build_environment()
        # the stopped helper's --rm removal can lag; the rerun reuses its name
        subprocess.run(
            [self._docker_binary, "rm", "-f", self._helper_container_name()],
            capture_output=True,
            text=True,
        )
        self._events.diagnostic("storage credentials expired; renewed them, restarting the backup")

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
        # The helper runs without --rm so its OOMKilled state can be read after exit, then it is removed.
        helper = not isinstance(self._workspace, LocalWorkspace)
        if helper:
            self._remove_stopped_helpers()
            self._remove_helper_container()
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
                # DAH-4693: restic refuses to save an empty snapshot (plain-text form)
                self._snapshot_empty = self._snapshot_empty or line == SNAPSHOT_EMPTY_MESSAGE
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
                    if not _snapshot_id(summary):
                        self._raise_if_credentials_expired(line)
                    self._events.diagnostic(_redact(line, self._secret_values()))
                    continue
                if not isinstance(payload, dict):
                    self._events.diagnostic("restic emitted a non-object JSON message")
                    continue
                if payload.get("message_type") == "exit_error" and payload.get("message") == SNAPSHOT_EMPTY_MESSAGE:
                    self._snapshot_empty = True
                # an expiry after the summary (restic's lock cleanup) must not rerun a finished backup
                if not _snapshot_id(summary):
                    self._raise_if_credentials_expired(_error_text(payload))
                self._events.restic_event(payload)
                self._events.heartbeat_if_due()
                self._cancel_if_requested(process)
                if payload.get("message_type") == "summary":
                    summary = payload

            exit_code = process.wait()
            return exit_code, summary
        except StorageOperationCancelled:
            if process.poll() is None:  # SIGTERM to the runner, not a cancel that already stopped it
                self._terminate_process(process)
            raise
        except Exception:
            self._terminate_process(process)
            raise
        finally:
            output_reader.join(timeout=1)
            if helper:
                self._helper_oom_killed = self._remove_helper_container()

    def _raise_if_credentials_expired(self, error_text: str) -> None:
        # restic 0.19 retries an expired session for 15 minutes before failing; stop it now
        if self._operation.repository.session_token and _credentials_expired(error_text):
            raise CredentialsExpired("storage credentials expired")

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

    def _remove_stopped_helpers(self) -> None:
        # A runner killed before cleanup leaves its exited helper (and its credentials) behind. Only long-exited
        # helpers are removed: a created or running one may belong to another operation that is starting now.
        listed = subprocess.run(
            [self._docker_binary, "ps", "--all", "--quiet", "--filter", f"label={HELPER_LABEL}",
             "--filter", "status=exited"],
            capture_output=True,
            text=True,
        )
        container_ids = listed.stdout.split() if listed.returncode == 0 else []
        if not container_ids:
            return
        inspected = subprocess.run(
            [self._docker_binary, "inspect", "--format", "{{.Id}} {{.State.Status}} {{.State.FinishedAt}}",
             *container_ids],
            capture_output=True,
            text=True,
        )
        cutoff = datetime.now(timezone.utc) - STALE_HELPER_AGE
        for line in inspected.stdout.splitlines():
            fields = line.split()
            if len(fields) != 3 or fields[1] != "exited":
                continue
            try:  # FinishedAt has nanoseconds; seconds are enough for a 10-minute cutoff
                finished_at = datetime.strptime(fields[2][:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            if finished_at < cutoff:
                subprocess.run([self._docker_binary, "rm", "--force", fields[0]], capture_output=True, text=True)

    def _remove_helper_container(self) -> bool:
        """Remove the helper container; return whether docker recorded it as OOM-killed."""
        name = self._helper_container_name()
        inspection = subprocess.run(
            [self._docker_binary, "inspect", "--format", "{{.State.OOMKilled}}", name],
            capture_output=True,
            text=True,
        )
        subprocess.run([self._docker_binary, "rm", "--force", name], capture_output=True, text=True)
        return inspection.returncode == 0 and inspection.stdout.strip() == "true"

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
                    BACKUP_SOURCE_SCRIPT,
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
            command.extend(
                [
                    "--entrypoint",
                    "/bin/sh",
                    self._workspace.image,
                    "-c",
                    BACKUP_SOURCE_SCRIPT,
                    "sh",
                    str(self._workspace.path),
                    self._restic_binary,
                    *restic_command[1:],
                ]
            )
            return command, None
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
        arguments.extend(["--", snapshot_id])
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
            "--name",
            self._helper_container_name(),
            "--label",
            f"{HELPER_LABEL}={self._operation.operation_id}",
            "--log-driver",
            "none",
            "--memory",
            RESTIC_HELPER_MEMORY,
            "--oom-score-adj",
            RESTIC_HELPER_OOM_SCORE_ADJ,
            "-e",
            f"GOGC={RESTIC_HELPER_GOGC}",
            "-e",
            f"GOMAXPROCS={RESTIC_HELPER_GOMAXPROCS}",
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
        )


def _is_retryable_repository_failure(failure_detail: str) -> bool:
    # Restic treats AccessDenied as permanent and does not retry it, but AWS can
    # return it temporarily while a newly attached IAM policy is propagating.
    retryable_failure_markers = ("accessdenied",)
    # Normalization matches both "Access Denied" and compact "AccessDenied" forms.
    normalized_detail = "".join(failure_detail.casefold().split())
    return any(marker in normalized_detail for marker in retryable_failure_markers)


def _credentials_expired(error_text: str) -> bool:
    # S3: "ExpiredToken: The provided token has expired."; minio-go often prints only the message.
    # Only the last ": " component counts, so a file name like "open ExpiredToken.java: ..." never matches.
    normalized = "".join(error_text.rsplit(":", 1)[-1].casefold().split()).rstrip(".")
    return normalized in ("expiredtoken", "theprovidedtokenhasexpired")


def _error_text(payload: Mapping[str, object]) -> str:
    """the message of a restic error/exit_error event, never its file names (item, current_files)"""
    if payload.get("message_type") not in ("error", "exit_error"):
        return ""
    error = payload.get("error")
    message = error.get("message") if isinstance(error, Mapping) else payload.get("message")
    if not isinstance(message, str):
        return ""
    item = payload.get("item")
    # restic repeats the file name in the message ("open /workspace/ExpiredToken.java: permission denied")
    return message.replace(item, "") if isinstance(item, str) and item else message


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
