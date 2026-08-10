from __future__ import annotations

import asyncio
import json
import shlex
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from uuid import UUID

import aiohttp
import asyncssh

REMOTE_OPERATION_DIRECTORY = PurePosixPath("/root/app/storage-operations")
REMOTE_RUNNER_PATH = "/root/app/src/storage_runner.py"


class StorageOperationLaunchError(RuntimeError):
    pass


@dataclass(frozen=True)
class StorageOperationFiles:
    spec: PurePosixPath
    log: PurePosixPath
    pid: PurePosixPath
    status: PurePosixPath
    cancel: PurePosixPath

    @classmethod
    def for_operation(cls, operation_id: UUID) -> StorageOperationFiles:
        stem = str(operation_id)
        return cls(
            spec=REMOTE_OPERATION_DIRECTORY / f"{stem}.json",
            log=REMOTE_OPERATION_DIRECTORY / f"{stem}.log",
            pid=REMOTE_OPERATION_DIRECTORY / f"{stem}.pid",
            status=REMOTE_OPERATION_DIRECTORY / f"{stem}.status",
            cancel=REMOTE_OPERATION_DIRECTORY / f"{stem}.cancel",
        )


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
        await _report_launch_failure(operation_id, spec, error)
        raise


async def wait_for_storage_operation(
    ssh_client: asyncssh.SSHClientConnection,
    files: StorageOperationFiles,
    poll_interval_seconds: float = 5.0,
    launch_grace_polls: int = 6,
) -> None:
    runner_seen = False
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
            if exit_code != 0:
                detail = await _tail_operation_log(ssh_client, files.log)
                await _remove_operation_artifacts(ssh_client, files)
                raise StorageOperationLaunchError(
                    f"storage operation failed with exit {exit_code}: {detail}"
                )
            await _remove_operation_artifacts(ssh_client, files)
            return
        if state == "RUNNING":
            runner_seen = True
        elif state in {"EXITED", "INVALID"} or runner_seen:
            detail = await _tail_operation_log(ssh_client, files.log)
            await _remove_operation_artifacts(ssh_client, files)
            raise StorageOperationLaunchError(
                f"storage operation runner disappeared without a terminal status: {detail}"
            )
        else:
            starting_polls += 1
            if starting_polls >= launch_grace_polls:
                detail = await _tail_operation_log(ssh_client, files.log)
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
    command = (
        f"if test -s {status_path}; then printf 'STATUS:'; cat {status_path}; "
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
    helper_name = f"lium-storage-{str(operation_id)[:12]}"
    cancel_path = shlex.quote(str(files.cancel))
    pid_path = shlex.quote(str(files.pid))
    # Let the runner terminate its Restic process and helper first. Direct
    # process termination is only a bounded fallback for an unresponsive runner.
    command = (
        f"touch {cancel_path}; "
        f"if test -s {pid_path}; then "
        f"pid=$(cat {pid_path}); "
        "case \"$pid\" in (*[!0-9]*|'') exit 2;; esac; "
        'attempt=0; while kill -0 "$pid" 2>/dev/null && test "$attempt" -lt 100; do '
        "sleep 0.25; attempt=$((attempt + 1)); done; "
        'if kill -0 "$pid" 2>/dev/null; then '
        f"/usr/bin/docker stop --time 5 {shlex.quote(helper_name)} >/dev/null 2>&1 || true; "
        'kill -TERM "$pid" 2>/dev/null || true; sleep 2; '
        'if kill -0 "$pid" 2>/dev/null; then kill -KILL "$pid" 2>/dev/null || true; '
        'attempt=0; while kill -0 "$pid" 2>/dev/null && test "$attempt" -lt 20; do '
        "sleep 0.1; attempt=$((attempt + 1)); done; fi; "
        "fi; "
        'kill -0 "$pid" 2>/dev/null && exit 3 || true; '
        "else "
        f"/usr/bin/docker stop --time 5 {shlex.quote(helper_name)} >/dev/null 2>&1 || true; "
        "fi"
    )
    result = await ssh_client.run(command, check=False)
    if result.exit_status == 2:
        raise StorageOperationLaunchError("storage operation had an invalid process ID")
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


async def _tail_operation_log(
    ssh_client: asyncssh.SSHClientConnection,
    log_path: PurePosixPath,
) -> str:
    result = await ssh_client.run(
        f"tail -n 20 {shlex.quote(str(log_path))}",
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
        ),
        check=False,
    )


async def _report_launch_failure(
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
