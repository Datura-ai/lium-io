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

    @classmethod
    def for_operation(cls, operation_id: UUID) -> StorageOperationFiles:
        stem = str(operation_id)
        return cls(
            spec=REMOTE_OPERATION_DIRECTORY / f"{stem}.json",
            log=REMOTE_OPERATION_DIRECTORY / f"{stem}.log",
            pid=REMOTE_OPERATION_DIRECTORY / f"{stem}.pid",
            status=REMOTE_OPERATION_DIRECTORY / f"{stem}.status",
        )


async def start_storage_operation(
    ssh_client: asyncssh.SSHClientConnection,
    python_path: str,
    operation_id: UUID,
    spec: Mapping[str, object],
) -> StorageOperationFiles:
    files = StorageOperationFiles.for_operation(operation_id)
    try:
        await ssh_client.run(
            f"install -d -m 0700 {shlex.quote(str(REMOTE_OPERATION_DIRECTORY))}; "
            f"find {shlex.quote(str(REMOTE_OPERATION_DIRECTORY))} -type f -mtime +1 -delete",
            check=True,
        )
        async with ssh_client.start_sftp_client() as sftp:
            async with sftp.open(str(files.spec), "w") as remote_spec:
                await remote_spec.write(json.dumps(dict(spec), separators=(",", ":"), sort_keys=True))
            await sftp.chmod(str(files.spec), 0o600)

        runner = " ".join(
            shlex.quote(item)
            for item in [python_path, REMOTE_RUNNER_PATH, "--spec", str(files.spec)]
        )
        wrapper = (
            f"rm -f {shlex.quote(str(files.status))}; "
            f"{runner} & child=$!; "
            f"printf '%s\\n' \"$child\" > {shlex.quote(str(files.pid))}; "
            "trap 'kill -TERM \"$child\" 2>/dev/null || true' TERM INT; "
            "wait \"$child\"; result=$?; "
            f"printf '%s\\n' \"$result\" > {shlex.quote(str(files.status))}; "
            f"rm -f {shlex.quote(str(files.spec))} {shlex.quote(str(files.pid))}"
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
) -> None:
    while True:
        result = await ssh_client.run(
            f"test -f {shlex.quote(str(files.status))} && cat {shlex.quote(str(files.status))}",
            check=False,
        )
        if result.exit_status == 0 and (result.stdout or "").strip():
            try:
                exit_code = int(result.stdout.strip())
            except ValueError as error:
                raise StorageOperationLaunchError("storage operation returned an invalid status") from error
            if exit_code != 0:
                detail = await _tail_operation_log(ssh_client, files.log)
                raise StorageOperationLaunchError(
                    f"storage operation failed with exit {exit_code}: {detail}"
                )
            await _remove_operation_status(ssh_client, files)
            return
        await asyncio.sleep(poll_interval_seconds)


async def cancel_storage_operation(
    ssh_client: asyncssh.SSHClientConnection,
    operation_id: UUID,
) -> None:
    files = StorageOperationFiles.for_operation(operation_id)
    helper_name = f"lium-storage-{str(operation_id)[:12]}"
    command = (
        f"/usr/bin/docker stop --time 5 {shlex.quote(helper_name)} >/dev/null 2>&1 || true; "
        f"if test -f {shlex.quote(str(files.pid))}; then "
        f"pid=$(cat {shlex.quote(str(files.pid))}); "
        "case \"$pid\" in (*[!0-9]*|'') exit 2;; esac; "
        "kill -TERM \"$pid\" 2>/dev/null || true; "
        "fi"
    )
    result = await ssh_client.run(command, check=False)
    if result.exit_status not in (0, 2):
        raise StorageOperationLaunchError("failed to cancel storage operation")


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


async def _remove_operation_status(
    ssh_client: asyncssh.SSHClientConnection,
    files: StorageOperationFiles,
) -> None:
    await ssh_client.run(
        "rm -f " + " ".join(shlex.quote(str(path)) for path in (files.status, files.log)),
        check=False,
    )


async def _report_launch_failure(
    operation_id: UUID,
    spec: Mapping[str, object],
    error: Exception,
) -> None:
    reporter = spec.get("reporter")
    if not isinstance(reporter, Mapping):
        return
    api_url = reporter.get("api_url")
    auth_token = reporter.get("auth_token")
    resource = reporter.get("resource")
    if not isinstance(api_url, str) or not isinstance(auth_token, str):
        return
    if resource not in ("backup", "restore"):
        return

    resource_path = "backup-logs" if resource == "backup" else "restore-logs"
    url = f"{api_url.rstrip('/')}/{resource_path}/{operation_id}/progress"
    message = f"Storage runner launch failed: {type(error).__name__}: {error}"
    payload = {
        "operation_id": str(operation_id),
        "progress": 0,
        "status": "FAILED",
        "stage": "LAUNCH",
        "error_message": message[:2000],
        "logs": [message[:2000]],
    }
    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.put(
                url,
                json=payload,
                headers={"Authorization": f"Bearer {auth_token}"},
            ) as response:
                await response.read()
    except Exception:
        return
