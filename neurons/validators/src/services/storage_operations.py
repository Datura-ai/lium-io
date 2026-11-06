from __future__ import annotations

import asyncio
import json
import shlex
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from uuid import UUID, uuid4

import aiohttp
import asyncssh

REMOTE_OPERATION_DIRECTORY = PurePosixPath("/root/app/storage-operations")
REMOTE_RUNNER_PATH = "/root/app/src/storage_runner.py"
# Helpers with this label run storage_runner.py as their entrypoint: they run restic, retry and report by themselves
# and outlive the executor container, so their docker state is the operation's state.
WRAPPER_LABEL = "lium.storage-wrapper"
KILLED_EXIT = 137
REMOTE_SECRETS_DIRECTORY = PurePosixPath("/root/app/storage-secrets")


class StorageOperationLaunchError(RuntimeError):
    pass


@dataclass(frozen=True)
class StorageOperationFiles:
    spec: PurePosixPath
    log: PurePosixPath
    pid: PurePosixPath
    status: PurePosixPath
    cancel: PurePosixPath
    # the operation's helper container: it, not the files above, outlives an executor restart
    helper: str = ""

    @classmethod
    def for_operation(cls, operation_id: UUID) -> StorageOperationFiles:
        stem = str(operation_id)
        return cls(
            spec=REMOTE_OPERATION_DIRECTORY / f"{stem}.json",
            log=REMOTE_OPERATION_DIRECTORY / f"{stem}.log",
            pid=REMOTE_OPERATION_DIRECTORY / f"{stem}.pid",
            status=REMOTE_OPERATION_DIRECTORY / f"{stem}.status",
            cancel=REMOTE_OPERATION_DIRECTORY / f"{stem}.cancel",
            helper=helper_container_name(operation_id),
        )


def helper_container_name(operation_id: UUID) -> str:
    return f"lium-storage-{str(operation_id)[:12]}"


async def supports_storage_operation(
    ssh_client: asyncssh.SSHClientConnection,
    engine: object,
) -> bool:
    required_binary = "/usr/local/bin/restic" if engine == "restic" else "/usr/bin/aws"
    result = await ssh_client.run(
        f"test -r {shlex.quote(REMOTE_RUNNER_PATH)} && test -x {shlex.quote(required_binary)}",
        check=False,
    )
    return result.exit_status == 0


# Asks the executor's own storage models whether `workspace.bootstrap` exists (DAH-3274). An
# executor image from before it drops the unknown key and refuses the non-empty target the
# pod's entrypoint has already written to, so the create must stop before `docker run`.
# Exit 0 = the field exists, exit 3 = the models import but have no such field (an old image);
# any other status is the probe itself failing (no python at that path = 127, an import error =
# 1, …) and is reported as such, not read as an old image (review, taiberium).
# DAH-2002: the same probe for `workspace.exclude_patterns`; an old image would back up what the excludes skip.
# An image with no `storage` package at all predates every field, so it is an old image too (exit 3);
# any other import error re-raises, so its traceback reaches the probe's stderr.
_WORKSPACE_FIELD_PROBE = (
    "import sys\n"
    "sys.path.insert(0, '/root/app/src')\n"
    "try:\n"
    "    from storage.models import WorkspaceSpec\n"
    "except ModuleNotFoundError as e:\n"
    "    if e.name not in ('storage', 'storage.models'):\n"
    "        raise\n"
    "    sys.exit(3)\n"
    "sys.exit(0 if sys.argv[1] in WorkspaceSpec.__dataclass_fields__ else 3)"
)
_BOOTSTRAP_PROBE_FIELD_MISSING = 3


async def supports_bootstrap_restore(
    ssh_client: asyncssh.SSHClientConnection,
    python_path: str,
) -> bool:
    return await supports_workspace_field(ssh_client, python_path, "bootstrap")


async def supports_workspace_field(
    ssh_client: asyncssh.SSHClientConnection,
    python_path: str,
    field: str,
) -> bool:
    """True when the executor's models know `workspace.<field>`, False when they do not.

    Raises RuntimeError, with the exit status and the tail of stderr, when the probe could not
    answer — the caller must not take that for an old executor image.
    """
    result = await ssh_client.run(
        f"{shlex.quote(python_path)} -c {shlex.quote(_WORKSPACE_FIELD_PROBE)} {shlex.quote(field)}",
        check=False,
    )
    if result.exit_status == 0:
        return True
    if result.exit_status == _BOOTSTRAP_PROBE_FIELD_MISSING:
        return False
    stderr = result.stderr if isinstance(result.stderr, str) else ""
    tail = " ".join(stderr.strip().splitlines()[-3:])[-400:]
    raise RuntimeError(
        f"{field} probe could not run on the executor (exit {result.exit_status}"
        f"{': ' + tail if tail else ''})"
    )


async def start_storage_operation(
    ssh_client: asyncssh.SSHClientConnection,
    python_path: str,
    operation_id: UUID,
    spec: Mapping[str, object],
    *,
    retain_terminal_artifacts: bool,
) -> StorageOperationFiles:
    files = StorageOperationFiles.for_operation(operation_id)
    try:
        engine = spec.get("engine")
        if not await supports_storage_operation(ssh_client, engine):
            raise StorageOperationLaunchError(
                f"executor does not support the requested {engine or 'storage'} operation"
            )
        await ssh_client.run(
            f"install -d -m 0700 {shlex.quote(str(REMOTE_OPERATION_DIRECTORY))}",
            check=True,
        )
        existing_state = await _operation_state(ssh_client, files)
        if existing_state == "STARTING" and await _remote_file_exists(ssh_client, files.spec):
            # A repeated backend dispatch can arrive while the first wrapper is between
            # writing its spec and PID. Give that deterministic operation a chance to appear.
            for _ in range(20):
                await asyncio.sleep(0.25)
                existing_state = await _operation_state(ssh_client, files)
                if existing_state != "STARTING":
                    break
        if existing_state == f"STATUS:{KILLED_EXIT}":
            await report_killed_helper(operation_id, spec)  # a killed helper could not report for itself
        if existing_state.startswith("STATUS:") and not retain_terminal_artifacts:
            # an online job's finished helper (its files are gone already): a new dispatch runs it again
            await _remove_operation_artifacts(ssh_client, files)
            existing_state = "STARTING"
        if existing_state == "RUNNING" or existing_state.startswith("STATUS:"):
            return files
        if await _helper_owned_by_live_runner(ssh_client, operation_id, files):
            # an earlier dispatch's helper is still working; removing it would kill that run.
            # A helper whose runner died is an orphan: it is removed below and the run relaunched.
            return files
        if existing_state in {"EXITED", "INVALID"} or await _remote_file_exists(
            ssh_client, files.spec
        ):
            await _remove_operation_artifacts(ssh_client, files)
        await _remove_stale_helper_container(ssh_client, operation_id)

        async with ssh_client.start_sftp_client() as sftp:
            async with sftp.open(str(files.spec), "w") as remote_spec:
                await remote_spec.write(
                    json.dumps(dict(spec), separators=(",", ":"), sort_keys=True)
                )
            await sftp.chmod(str(files.spec), 0o600)

        runner = " ".join(
            shlex.quote(item)
            for item in [
                python_path,
                REMOTE_RUNNER_PATH,
                "--spec",
                str(files.spec),
                "--cancel-file",
                str(files.cancel),
            ]
        )
        terminal_cleanup = (
            ""
            if retain_terminal_artifacts
            else f"; rm -f {shlex.quote(str(files.status))} {shlex.quote(str(files.log))}"
        )
        # Bootstrap callers consume the terminal files; detached online jobs
        # remove them themselves after the runner has finished reporting.
        wrapper = (
            f"rm -f {shlex.quote(str(files.status))} {shlex.quote(str(files.cancel))}; "
            f"{runner} & child=$!; "
            f"printf '%s\\n' \"$child\" > {shlex.quote(str(files.pid))}; "
            "trap 'kill -TERM \"$child\" 2>/dev/null || true' TERM INT; "
            'wait "$child"; result=$?; '
            f"printf '%s\\n' \"$result\" > {shlex.quote(str(files.status))}; "
            f"rm -f {shlex.quote(str(files.spec))} {shlex.quote(str(files.pid))} "
            f"{shlex.quote(str(files.cancel))}{terminal_cleanup}"
        )
        launch_command = (
            f"nohup /bin/sh -c {shlex.quote(wrapper)} "
            f"> {shlex.quote(str(files.log))} 2>&1 < /dev/null &"
        )
        await ssh_client.run(launch_command, check=True)
        return files
    except Exception as error:
        await report_launch_failure(operation_id, spec, error)
        raise


async def wait_for_storage_operation(
    ssh_client: asyncssh.SSHClientConnection,
    files: StorageOperationFiles,
    poll_interval_seconds: float = 5.0,
    launch_grace_polls: int = 6,
    *,
    reconnect: Callable[[], Awaitable[asyncssh.SSHClientConnection]] | None = None,
    reconnect_window_seconds: float = 600.0,
    spec: Mapping[str, object] | None = None,
) -> asyncssh.SSHClientConnection:
    """Wait for the operation's terminal status; return the live connection the wait ended on.

    With ``reconnect``, a lost SSH connection (the executor container restarting) is reopened and the wait
    reattaches to the operation: its helper keeps running, and its state is read from that container.
    The caller owns the reopened connection and continues on it.
    """
    try:
        await _wait_for_storage_operation(ssh_client, files, poll_interval_seconds, launch_grace_polls, False, spec)
        return ssh_client
    except (asyncssh.Error, OSError):
        if reconnect is None:
            raise
    deadline = asyncio.get_running_loop().time() + reconnect_window_seconds
    while True:
        await asyncio.sleep(poll_interval_seconds)
        try:
            fresh_client = await reconnect()
            await _wait_for_storage_operation(
                fresh_client, files, poll_interval_seconds, launch_grace_polls, True, spec
            )
            return fresh_client
        except (asyncssh.Error, OSError):
            if asyncio.get_running_loop().time() >= deadline:
                raise


async def _wait_for_storage_operation(
    ssh_client: asyncssh.SSHClientConnection,
    files: StorageOperationFiles,
    poll_interval_seconds: float,
    launch_grace_polls: int,
    runner_seen: bool,
    spec: Mapping[str, object] | None,
) -> None:
    starting_polls = 0
    while True:
        # Healthy large operations have no wall-clock deadline; only terminal
        # status or loss of the supervised runner ends this wait.
        state = await _operation_state(ssh_client, files)
        if state.startswith("STATUS:"):
            try:
                exit_code = int(state.removeprefix("STATUS:").strip())
            except ValueError as error:
                await _remove_operation_artifacts(ssh_client, files)
                raise StorageOperationLaunchError(
                    "storage operation returned an invalid status"
                ) from error
            if exit_code == KILLED_EXIT and spec is not None:
                await report_killed_helper(UUID(str(spec["operation_id"])), spec)
            if exit_code != 0:
                detail = await _tail_operation_log(ssh_client, files.log, files.helper)
                await _remove_operation_artifacts(ssh_client, files)
                raise StorageOperationLaunchError(
                    f"storage operation failed with exit {exit_code}: {detail}"
                )
            await _remove_operation_artifacts(ssh_client, files)
            return
        if state == "RUNNING":
            runner_seen = True
        elif state in {"EXITED", "INVALID"} or runner_seen:
            detail = await _tail_operation_log(ssh_client, files.log, files.helper)
            await _remove_operation_artifacts(ssh_client, files)
            raise StorageOperationLaunchError(
                f"storage operation runner disappeared without a terminal status: {detail}"
            )
        else:
            starting_polls += 1
            if starting_polls >= launch_grace_polls:
                detail = await _tail_operation_log(ssh_client, files.log, files.helper)
                await _remove_operation_artifacts(ssh_client, files)
                raise StorageOperationLaunchError(
                    f"storage operation runner did not start: {detail}"
                )
        await asyncio.sleep(poll_interval_seconds)


async def _operation_state(
    ssh_client: asyncssh.SSHClientConnection,
    files: StorageOperationFiles,
) -> str:
    status_path = shlex.quote(str(files.status))
    pid_path = shlex.quote(str(files.pid))
    helper = shlex.quote(files.helper)
    inspect_format = (
        f'{{{{index .Config.Labels "{WRAPPER_LABEL}"}}}} {{{{.State.Status}}}} '
        "{{.State.ExitCode}} {{.State.OOMKilled}}"
    )
    command = (
        # a helper running storage_runner.py is the operation: it survives an executor restart, the files do not.
        # A killed one cannot report: OOMKilled reads as exit 137.
        f"if state=$(/usr/bin/docker inspect -f {shlex.quote(inspect_format)} {helper} 2>/dev/null) "
        '&& test "${state%% *}" = 1; then set -- $state; '
        'case "$2" in (exited|dead) if test "$4" = true; then echo STATUS:137; else echo "STATUS:$3"; fi;; '
        "(*) echo RUNNING;; esac; "
        f"elif test -s {status_path}; then printf 'STATUS:'; cat {status_path}; "
        # executors before the wrapper helper: their runner is a process in the executor container
        f"elif test -s {pid_path}; then pid=$(cat {pid_path}); "
        "case \"$pid\" in (*[!0-9]*|'') echo INVALID;; "
        '(*) if kill -0 "$pid" 2>/dev/null; then echo RUNNING; else echo EXITED; fi;; esac; '
        "else echo STARTING; fi"
    )
    result = await ssh_client.run(command, check=False)
    return (result.stdout or "").strip()


async def cancel_storage_operation(
    ssh_client: asyncssh.SSHClientConnection,
    operation_id: UUID,
) -> None:
    files = StorageOperationFiles.for_operation(operation_id)
    spec = await _read_operation_spec(ssh_client, files.spec)
    helper = shlex.quote(files.helper)
    legacy_runner = shlex.quote(f"[-]-spec {files.spec}")  # "[-]" keeps pgrep from matching this shell
    # Targets are named, never a stored PID (the executor shares the host's PID namespace, so a stale PID
    # could be an unrelated process). SIGTERM to the helper: its storage_runner.py stops restic and reports CANCELLED.
    command = (
        f"touch {shlex.quote(str(files.cancel))}; "
        f"/usr/bin/docker kill --signal TERM {helper} >/dev/null 2>&1 || true; "
        f'attempt=0; while test "$(/usr/bin/docker inspect -f {{{{.State.Running}}}} {helper} 2>/dev/null)" = true '
        '&& test "$attempt" -lt 120; do sleep 0.25; attempt=$((attempt + 1)); done; '
        # an executor before the wrapper helper: its runner is a process found by its command line
        f"pkill -TERM -f -- {legacy_runner} 2>/dev/null; "
        f'attempt=0; while pgrep -f -- {legacy_runner} >/dev/null && test "$attempt" -lt 100; do '
        "sleep 0.25; attempt=$((attempt + 1)); done; "
        f"pkill -KILL -f -- {legacy_runner} 2>/dev/null; "
        f"/usr/bin/docker stop --time 5 {helper} >/dev/null 2>&1 || true; "
        # the stopped helper still holds the repository credentials; remove it, not just stop it
        f"/usr/bin/docker rm -f -- {helper} >/dev/null 2>&1 || true; "
        f'attempt=0; while pgrep -f -- {legacy_runner} >/dev/null && test "$attempt" -lt 40; do '
        "sleep 0.25; attempt=$((attempt + 1)); done; "
        f"! pgrep -f -- {legacy_runner} >/dev/null"
    )
    result = await ssh_client.run(command, check=False)
    if result.exit_status != 0:
        raise StorageOperationLaunchError("failed to cancel storage operation")
    if spec:
        await _report_cancelled(operation_id, spec)
    await _remove_operation_artifacts(ssh_client, files)


async def _read_operation_spec(
    ssh_client: asyncssh.SSHClientConnection,
    spec_path: PurePosixPath,
) -> Mapping[str, object] | None:
    result = await ssh_client.run(
        f"cat {shlex.quote(str(spec_path))}",
        check=False,
    )
    if result.exit_status != 0:
        return None
    try:
        value = json.loads(result.stdout or "")
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, Mapping) else None


async def _remote_file_exists(
    ssh_client: asyncssh.SSHClientConnection,
    path: PurePosixPath,
) -> bool:
    result = await ssh_client.run(f"test -e {shlex.quote(str(path))}", check=False)
    return result.exit_status == 0


async def _helper_owned_by_live_runner(
    ssh_client: asyncssh.SSHClientConnection,
    operation_id: UUID,
    files: StorageOperationFiles,
) -> bool:
    helper_name = f"lium-storage-{str(operation_id)[:12]}"
    # "[-]-spec" keeps pgrep from matching this command's own shell
    runner_pattern = f"[-]-spec {files.spec}"
    # an executor before the wrapper helper (a running wrapper helper already reads as RUNNING)
    result = await ssh_client.run(
        f"/usr/bin/docker ps -q --filter {shlex.quote(f'name=^{helper_name}$')} --filter status=running"
        f" | grep -q . && pgrep -f -- {shlex.quote(runner_pattern)} >/dev/null",
        check=False,
    )
    return result.exit_status == 0


async def _remove_stale_helper_container(
    ssh_client: asyncssh.SSHClientConnection,
    operation_id: UUID,
) -> None:
    helper_name = f"lium-storage-{str(operation_id)[:12]}"
    await ssh_client.run(
        f"/usr/bin/docker rm -f -- {shlex.quote(helper_name)} >/dev/null 2>&1 || true",
        check=False,
    )


async def _tail_operation_log(
    ssh_client: asyncssh.SSHClientConnection,
    log_path: PurePosixPath,
    helper: str = "",
) -> str:
    # the launcher's log, then the wrapper helper's own (redacted) event log, which outlives an executor restart
    fallback = f" 2>/dev/null; /usr/bin/docker logs --tail 20 {shlex.quote(helper)} 2>&1" if helper else ""
    result = await ssh_client.run(
        f"tail -n 20 {shlex.quote(str(log_path))}{fallback}",
        check=False,
    )
    compact = " ".join((result.stdout or result.stderr or "no operation log").split())
    return compact[-2000:]


async def _remove_operation_artifacts(
    ssh_client: asyncssh.SSHClientConnection,
    files: StorageOperationFiles,
) -> None:
    await ssh_client.run(
        "rm -f "
        + " ".join(
            shlex.quote(str(path))
            for path in (files.status, files.log, files.pid, files.spec, files.cancel)
        )
        + (f"; /usr/bin/docker rm -f -- {shlex.quote(files.helper)} >/dev/null 2>&1 || true" if files.helper else ""),
        check=False,
    )


async def report_launch_failure(
    operation_id: UUID,
    spec: Mapping[str, object],
    error: Exception,
) -> None:
    await _report_operation_status(
        operation_id,
        spec,
        status="FAILED",
        stage="LAUNCH",
        message=f"Storage runner launch failed: {type(error).__name__}: {error}",
    )


async def report_killed_helper(operation_id: UUID, spec: Mapping[str, object]) -> None:
    await _report_operation_status(
        operation_id,
        spec,
        status="FAILED",
        # the stage the helper was in: the executor reports backups under BACKUP and restores under RESTORING
        stage="BACKUP" if spec.get("action") == "backup" else "RESTORING",
        message="Storage helper was killed (often out of memory)",
        attempts=3,
    )


async def _report_cancelled(
    operation_id: UUID,
    spec: Mapping[str, object],
) -> bool:
    return await _report_operation_status(
        operation_id,
        spec,
        status="CANCELLED",
        stage="CANCELLED",
        message="Storage operation cancelled by user",
        attempts=3,
    )


async def _report_operation_status(
    operation_id: UUID,
    spec: Mapping[str, object],
    *,
    status: str,
    stage: str,
    message: str,
    attempts: int = 1,
) -> bool:
    reporter = spec.get("reporter")
    if not isinstance(reporter, Mapping):
        return False
    api_url = reporter.get("api_url")
    auth_token = reporter.get("auth_token")
    resource = reporter.get("resource")
    if not isinstance(api_url, str) or not isinstance(auth_token, str):
        return False
    if resource not in ("backup", "restore"):
        return False

    resource_path = "backup-logs" if resource == "backup" else "restore-logs"
    url = f"{api_url.rstrip('/')}/{resource_path}/{operation_id}/progress"
    payload = {
        "operation_id": str(operation_id),
        "progress": 0,
        "status": status,
        "stage": stage,
        "error_message": message[:2000],
        "logs": [message[:2000]],
    }
    for attempt in range(attempts):
        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.put(
                    url,
                    json=payload,
                    headers={"Authorization": f"Bearer {auth_token}"},
                ) as response:
                    await response.read()
                    if getattr(response, "status", 200) < 400:
                        return True
        except Exception:
            pass
        if attempt + 1 < attempts:
            await asyncio.sleep(2**attempt)
    return False


async def upload_secrets_file(ssh_client: asyncssh.SSHClientConnection, auth_token: str, volume_info) -> str:
    """Hand the legacy tar backup/restore scripts their credentials as a fresh 0600 file.

    Keeps the values out of the remote command line. The file is
    created exclusively with mode 0600 inside a 0700 directory; the script deletes it on read.
    """
    path = str(REMOTE_SECRETS_DIRECTORY / f"{uuid4().hex}.json")
    await ssh_client.run(f"install -d -m 0700 {shlex.quote(str(REMOTE_SECRETS_DIRECTORY))}", check=True)
    async with ssh_client.start_sftp_client() as sftp:
        async with sftp.open(path, "x", asyncssh.SFTPAttrs(permissions=0o600)) as remote_file:
            await remote_file.write(
                json.dumps(
                    {
                        "auth_token": auth_token,
                        "aws_access_key_id": volume_info.iam_user_access_key,
                        "aws_secret_access_key": volume_info.iam_user_secret_key,
                    }
                )
            )
    return path
