from __future__ import annotations

import asyncio
import json
import logging
import shlex
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING
from uuid import UUID

import aiohttp
import asyncssh

from services import executor_payload

if TYPE_CHECKING:
    from services.redis_service import RedisService

logger = logging.getLogger(__name__)

REMOTE_OPERATION_DIRECTORY = PurePosixPath("/root/app/storage-operations")
# the executor image's own runner: its presence marks an image that supports storage operations. It is never run; the
# validator runs its own copy (executor_payload.STORAGE)
REMOTE_RUNNER_PATH = "/root/app/src/storage_runner.py"

# DAH-6148: an operation runs restic in a `lium-storage-<operation id[:12]>` helper container on
# the node, and reruns it under the same name when it renews credentials or retries an upload.
# The operations this validator started are recorded per executor, and the provider-side load
# check excuses a helper only by its exact name, only while the backend has that operation
# open (never by files on the node, which the provider can write), and only up to the helper
# cap: never a container that only wears the prefix, and never one after its operation ended.
STORAGE_HELPER_PREFIX = "lium-storage-"
STARTED_STORAGE_HELPERS_PREFIX = "started_storage_helpers:"
# a docker container name cannot hold ":", so no container's name or id ever matches this key
STORAGE_HELPER_KEY_MARK = "helper:"
# the backend fails a backup still open after 4 h (BACKUP_AUTH_TOKEN_EXPIRY_HOURS): no helper
# legitimately outlives that, so no record does either
STARTED_STORAGE_HELPERS_TTL_SECONDS = 4 * 3600
_STARTED_STORAGE_HELPERS_REDIS_TIMEOUT_SEC = 2


def storage_helper_name(operation_id: UUID | str) -> str:
    return f"{STORAGE_HELPER_PREFIX}{str(operation_id)[:12]}"


def storage_helper_key(operation_id: UUID | str) -> str:
    return f"{STORAGE_HELPER_KEY_MARK}{storage_helper_name(operation_id)}"


@dataclass(frozen=True)
class StorageHelperRecord:
    """An operation this validator started on the executor, and when."""

    operation_id: str
    started: int

    def format(self) -> str:
        return f"{self.operation_id}|{self.started}"

    @classmethod
    def parse(cls, raw: str) -> StorageHelperRecord | None:
        operation_id, _, started = raw.partition("|")
        try:
            return cls(str(UUID(operation_id)), int(started))
        except ValueError:
            return None


def _key(executor_id: str) -> str:
    return f"{STARTED_STORAGE_HELPERS_PREFIX}{executor_id}"


async def _redis_call(what: str, call: Callable[[], Awaitable[None]]) -> bool:
    """Best effort: Redis down or slow leaves the records as they are (an unrecorded helper's
    cores stay on the provider, as before DAH-6148)."""
    try:
        await asyncio.wait_for(call(), timeout=_STARTED_STORAGE_HELPERS_REDIS_TIMEOUT_SEC)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Storage helper records: %s failed: %s", what, exc)
        return False


async def _raw_records(redis: RedisService, executor_id: str) -> list[str]:
    members = await redis.smembers(_key(executor_id))
    return [m.decode() if isinstance(m, bytes) else str(m) for m in members or ()]


async def note_started_storage_helper(
    redis: RedisService, executor_id: str, operation_id: UUID | str
) -> None:
    """Records the operation starting its helper on this executor."""

    async def note() -> None:
        record = StorageHelperRecord(str(operation_id), int(time.time()))
        await redis.sadd(_key(executor_id), record.format())
        await redis.expire(_key(executor_id), STARTED_STORAGE_HELPERS_TTL_SECONDS)

    await _redis_call(f"record {operation_id} on {executor_id}", note)


async def forget_storage_helper(
    redis: RedisService, executor_id: str, operation_id: UUID | str
) -> None:
    """The operation ended where the validator saw it end (launch failure, cancel, a bootstrap
    restore it waited for): its helper is no longer Lium's."""

    async def forget() -> None:
        for raw in await _raw_records(redis, executor_id):
            record = StorageHelperRecord.parse(raw)
            if record and record.operation_id == str(operation_id):
                await redis.srem(_key(executor_id), raw)

    await _redis_call(f"forget {operation_id} on {executor_id}", forget)


async def settle_excused_storage_helper_keys(
    redis: RedisService,
    executor_id: str,
    is_open: Callable[[str], Awaitable[bool | None]],
) -> set[str]:
    """Drops the records whose operation ended and returns the helper keys to excuse.

    `is_open` asks the backend whether the operation is still open (None: no answer, or open but
    its runner went quiet). The node's
    own runner files are never read: the provider can write them (DAH-6148). An open operation's
    helper is excused by its exact name; an ended one, or one past the TTL, loses its record. An
    operation the backend did not answer for keeps its record and excuses nothing this cycle, and
    so does Redis not answering.
    """
    raw_records: list[str] = []

    async def read_raw_records() -> None:
        raw_records.extend(await _raw_records(redis, executor_id))

    if not await _redis_call(f"read helpers of {executor_id}", read_raw_records):
        return set()
    now = int(time.time())
    records = {raw: StorageHelperRecord.parse(raw) for raw in raw_records}
    unexpired_records = {
        raw: record
        for raw, record in records.items()
        if record and 0 <= now - record.started <= STARTED_STORAGE_HELPERS_TTL_SECONDS
    }
    ended = list(records.keys() - unexpired_records.keys())
    keys: set[str] = set()
    # outside the Redis timeout: the backend call has its own, longer one
    operation_ids = sorted({r.operation_id for r in unexpired_records.values()})
    answers = dict(zip(operation_ids, await asyncio.gather(*map(is_open, operation_ids))))
    for raw, record in unexpired_records.items():
        if answers[record.operation_id]:
            keys.add(storage_helper_key(record.operation_id))
        elif answers[record.operation_id] is False:
            ended.append(raw)

    async def drop_ended_records() -> None:
        for raw in ended:
            await redis.srem(_key(executor_id), raw)

    if ended:
        await _redis_call(f"drop ended helpers of {executor_id}", drop_ended_records)
    return keys


class StorageOperationLaunchError(RuntimeError):
    pass


class StorageDispatchOutcomeUnknown(StorageOperationLaunchError):
    """A dispatch failed and the node could not say whether another dispatch launched the operation, nor record this
    failure under the operation's lock: nothing is reported, so a run that may be live is never failed; the backend's
    next dispatch of the operation reconciles (it finds the run, its result or nothing to launch)."""


@dataclass(frozen=True)
class StorageOperationFiles:
    spec: PurePosixPath
    log: PurePosixPath
    pid: PurePosixPath
    status: PurePosixPath
    cancel: PurePosixPath
    # the operation's launch lock, its launcher's PID and an online run's done marker: none holds a credential
    lock: PurePosixPath
    launcher: PurePosixPath
    done: PurePosixPath

    @classmethod
    def for_operation(cls, operation_id: UUID) -> StorageOperationFiles:
        stem = str(operation_id)
        return cls(
            spec=REMOTE_OPERATION_DIRECTORY / f"{stem}.json",
            log=REMOTE_OPERATION_DIRECTORY / f"{stem}.log",
            pid=REMOTE_OPERATION_DIRECTORY / f"{stem}.pid",
            status=REMOTE_OPERATION_DIRECTORY / f"{stem}.status",
            cancel=REMOTE_OPERATION_DIRECTORY / f"{stem}.cancel",
            lock=REMOTE_OPERATION_DIRECTORY / f"{stem}.lock",
            launcher=REMOTE_OPERATION_DIRECTORY / f"{stem}.launcher",
            done=REMOTE_OPERATION_DIRECTORY / f"{stem}.done",
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


# Asks the storage models the operation will run with, the validator's copy staged on the executor and loaded by the
# executor's interpreter, whether `workspace.bootstrap` exists (DAH-3274) or `workspace.exclude_patterns` (DAH-2002).
# Exit 0 = the field exists, exit 3 = the models import but have no such field; any other status is the probe itself
# failing (no python at that path = 127, an import error = 1, …) and is reported as such (review, taiberium).
_WORKSPACE_FIELD_PROBE = (
    "import sys\n"
    "sys.path.insert(0, sys.argv[2])\n"
    "from storage.models import WorkspaceSpec\n"
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
    """True when the storage models the operation runs with know `workspace.<field>`, False when they do not.

    Raises RuntimeError, with the exit status and the tail of stderr, when the probe could not
    answer — the caller must not take that for a missing field.
    """
    result = await ssh_client.run(
        f"{executor_payload.unpack('storage', cleanup=True)}; "
        f"{shlex.quote(python_path)} -E -s -c {shlex.quote(_WORKSPACE_FIELD_PROBE)} {shlex.quote(field)} \"$d\"",
        input=executor_payload.archive(executor_payload.STORAGE).decode(),
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
        # a repeat dispatch of a running or finished operation stages nothing: a staging failure must not fail it
        if await storage_operation_has_incumbent(ssh_client, operation_id, spec):
            return files
        engine = spec.get("engine")
        if not await supports_storage_operation(ssh_client, engine):
            raise StorageOperationLaunchError(
                f"executor does not support the requested {engine or 'storage'} operation"
            )
        await ssh_client.run(
            f"install -d -m 0700 {shlex.quote(str(REMOTE_OPERATION_DIRECTORY))}",
            check=True,
        )
        # stage and verify the runner first: the spec carries the repository credentials, so it is written only once the
        # runner that consumes and removes it is in place; a failed unpack or check leaves no spec and no partial directory
        staged = await ssh_client.run(
            f"{executor_payload.unpack('storage')}; {executor_payload.verify(executor_payload.STORAGE)}; "
            'printf "%s" "$d"',
            input=executor_payload.archive(executor_payload.STORAGE).decode(),
            check=True,
        )
        stage_dir = _staged_directory(staged.stdout)
        result = await ssh_client.run(
            _launch_command(files, operation_id, python_path, stage_dir, retain_terminal_artifacts),
            input=json.dumps(dict(spec), separators=(",", ":"), sort_keys=True),
            check=False,
        )
        if result.exit_status != 0 or (result.stdout or "").strip() not in (_LAUNCHED, _ALREADY_LAUNCHED):
            raise StorageOperationLaunchError(f"storage runner launch failed (exit {result.exit_status})")
        return files
    except Exception as error:
        await settle_dispatch_failure(
            ssh_client, operation_id, spec, error, retain_terminal_artifacts=retain_terminal_artifacts
        )
        return files


async def settle_dispatch_failure(
    ssh_client: asyncssh.SSHClientConnection,
    operation_id: UUID,
    spec: Mapping[str, object],
    error: Exception,
    *,
    retain_terminal_artifacts: bool,
) -> None:
    """Settles a dispatch that failed before its launch was acknowledged, in one of three ways. The operation has a
    result or a live run (another dispatch launched it): returns, its run is the operation's. This dispatch's failure
    was recorded on the node under the operation's lock (no later dispatch launches it): reports FAILED and raises
    `error`. Neither could be established (the node could not be asked, or no lock): raises
    StorageDispatchOutcomeUnknown and reports nothing. Only a recorded failure is ever reported."""
    files = StorageOperationFiles.for_operation(operation_id)
    message = _launch_failure_message(error)
    try:
        outcome = await _settle_failed_dispatch(ssh_client, files, retain_terminal_artifacts, message)
    except Exception:  # noqa: BLE001 - the node could not be asked: unknown
        outcome = _UNKNOWN
    if outcome == _INCUMBENT:
        await _redeliver_recorded_failure(ssh_client, files, operation_id, spec)
        return
    if outcome != _RECORDED:
        raise StorageDispatchOutcomeUnknown(f"storage dispatch outcome unknown after: {message}") from error
    # recorded with its reason (online runs): a report the backend did not take is sent again by the next dispatch
    if await report_launch_failure(operation_id, spec, error) and not retain_terminal_artifacts:
        await _acknowledge_recorded_failure(ssh_client, files)
    raise error


async def _redeliver_recorded_failure(
    ssh_client: asyncssh.SSHClientConnection,
    files: StorageOperationFiles,
    operation_id: UUID,
    spec: Mapping[str, object],
) -> None:
    """An online operation's done marker holding a reason is a recorded launch failure the backend has not taken yet:
    it is reported again with this dispatch's reporter (the node keeps no credentials for it) and, once taken, the
    marker is emptied (an ordinary finished run). Best effort: the marker keeps it for the next dispatch."""
    try:
        result = await ssh_client.run(f"cat {shlex.quote(str(files.done))} 2>/dev/null", check=False)
        message = (result.stdout or "").strip() if isinstance(result.stdout, str) else ""
        if message and await _report_operation_status(
            operation_id, spec, status="FAILED", stage="LAUNCH", message=message
        ):
            await _acknowledge_recorded_failure(ssh_client, files)
    except Exception:  # noqa: BLE001 - the marker stays for the next dispatch
        logger.warning("A recorded storage launch failure could not be reported again", exc_info=True)


async def _acknowledge_recorded_failure(ssh_client: asyncssh.SSHClientConnection, files: StorageOperationFiles) -> None:
    await ssh_client.run(f": > {shlex.quote(str(files.done))}", check=False)


_LAUNCHED = "LAUNCHED"
_ALREADY_LAUNCHED = "ALREADY_LAUNCHED"
# _settle_failed_dispatch's outcomes
_INCUMBENT = "INCUMBENT"
_RECORDED = "RECORDED"
_UNKNOWN = "UNKNOWN"
# an online run's done marker outlives the backend's re-dispatches of the operation: they stop once its terminal report
# lands or its stale sweep fails it (15 min without a heartbeat). Swept only as housekeeping, a day after
_FINISHED_MARKER_SWEEP_MINUTES = 24 * 60


def _incumbent_condition(files: StorageOperationFiles) -> str:
    """Shell test, true when the operation has a result or a live run: its done marker or status, or a launcher (this
    validator's) or runner PID (an older validator's, which records no launcher) still running this operation's spec. A
    PID counts only while its process carries this operation's exact --spec path (a reused PID does not)."""
    q = lambda path: shlex.quote(str(path))  # noqa: E731
    live = lambda path: _live(path, files)  # noqa: E731
    # liveness first, then the status, then the done marker: the order the wrapper publishes them in reverse (status,
    # then done, then it exits), so a run that ends between two reads is still found by a later one
    return f"{{ {live(files.launcher)} || {live(files.pid)} || [ -s {q(files.status)} ] || [ -e {q(files.done)} ]; }}"


def _live(path, files: StorageOperationFiles) -> str:
    """Shell test, true while the PID in `path` runs this operation's exact --spec path (a reused PID does not)."""
    # argv is NUL-separated: as spaces, both the runner's own argv and the launcher's `sh -c` text carry it
    spec_arg = shlex.quote(f"--spec {files.spec} ")
    return (
        f'{{ p=$(cat {shlex.quote(str(path))} 2>/dev/null); case "$p" in (*[!0-9]*|"") false;; '
        f"(*) kill -0 \"$p\" 2>/dev/null && tr '\\0' ' ' < /proc/\"$p\"/cmdline | grep -qF -- {spec_arg};; esac; }}"
    )


async def storage_operation_has_incumbent(
    ssh_client: asyncssh.SSHClientConnection, operation_id: UUID, spec: Mapping[str, object]
) -> bool:
    """The operation already has a result or a live run on the node (_has_incumbent): a repeat dispatch does nothing,
    except report again a recorded launch failure the backend has not taken (_redeliver_recorded_failure)."""
    files = StorageOperationFiles.for_operation(operation_id)
    if not await _has_incumbent(ssh_client, files):
        return False
    await _redeliver_recorded_failure(ssh_client, files, operation_id, spec)
    return True


async def _has_incumbent(ssh_client: asyncssh.SSHClientConnection, files: StorageOperationFiles) -> bool:
    """The operation already has a result or a live run (see _incumbent_condition). Read under the operation's lock when
    its file can be created (_lock_if_possible): a launch holds it until its launcher is recorded, so a launch in its
    handoff is seen once it has one."""
    result = await ssh_client.run(
        f"{_lock_if_possible(files)}if {_incumbent_condition(files)}; then echo YES; fi", check=False
    )
    return result.exit_status == 0 and (result.stdout or "").strip() == "YES"


def _lock_if_possible(files: StorageOperationFiles) -> str:
    """Shell that takes the operation's lock and sets `locked`, or, when its file cannot be created (a full disk; an
    older validator's run has none), goes on unlocked: a launch cannot take the lock either, and the PID and result
    files an older validator's run left are read all the same."""
    q = shlex.quote(str(files.lock))
    # `exec` with a failed redirection would end the shell, so the file is opened by a plain command first
    return f"locked=; if ( : >>{q} ) 2>/dev/null; then exec 9>>{q} && flock -x 9 || exit 1; locked=1; fi; "


async def _settle_failed_dispatch(
    ssh_client: asyncssh.SSHClientConnection, files: StorageOperationFiles, retain_terminal_artifacts: bool, message: str
) -> str:
    """_INCUMBENT when the operation has a result or a live run (see _incumbent_condition); _RECORDED when, holding the
    lock, it recorded this dispatch's failure as the operation's result, as a finished run would (an online run's done
    marker, holding `message` until the backend takes the report; a bootstrap run's non-zero status), so no dispatch
    launches it after; _UNKNOWN when neither (no lock to hold, or the write or the read failed)."""
    q = lambda path: shlex.quote(str(path))  # noqa: E731
    # published whole (written aside, then renamed): a marker is either the full reason or absent
    target, content = (files.status, "1") if retain_terminal_artifacts else (files.done, message[:2000])
    tmp = q(f"{target}.tmp")
    failed = f"{{ printf '%s\\n' {shlex.quote(content)} > {tmp} && mv {tmp} {q(target)} || {{ rm -f {tmp}; false; }}; }}"
    result = await ssh_client.run(
        f"install -d -m 0700 {q(REMOTE_OPERATION_DIRECTORY)} 2>/dev/null; {_lock_if_possible(files)}"
        f'if {_incumbent_condition(files)}; then echo {_INCUMBENT}; '
        # unlocked, "no run" may change before a report: nothing is recorded and the outcome is unknown
        f'elif [ -n "$locked" ] && {failed}; then echo {_RECORDED}; fi',
        check=False,
    )
    outcome = (result.stdout or "").strip() if result.exit_status == 0 and isinstance(result.stdout, str) else ""
    return outcome if outcome in (_INCUMBENT, _RECORDED) else _UNKNOWN


def _launch_command(
    files: StorageOperationFiles,
    operation_id: UUID,
    python_path: str,
    stage_dir: str,
    retain_terminal_artifacts: bool,
) -> str:
    """One shell, under the operation's own lock, that decides whether this dispatch launches the operation and, if so,
    launches it. Under that lock it checks for a terminal result or a live launcher, removes a dead run's leftovers,
    writes the spec from stdin and records the launcher's PID, so a second dispatch of the operation waits for the lock
    and then finds this one's launcher (or its result) and launches nothing. The lock is the operation's, not its spec,
    which the runner removes once it has read it. A dispatch that launches nothing removes only the runner it staged."""
    q = lambda path: shlex.quote(str(path))  # noqa: E731
    stage = shlex.quote(stage_dir)
    runner = (
        f'{shlex.quote(python_path)} -E -s "$D"/{executor_payload.STORAGE[0]} '
        f"--spec {q(files.spec)} --cancel-file {q(files.cancel)}"
    )
    terminal_cleanup = (
        ""
        if retain_terminal_artifacts
        # the done marker first: a re-dispatch after this run must find it finished, not relaunch it. A failed run
        # keeps its log: a runner that dies before reporting leaves no other trace
        else f"; : > {q(files.done)}; rm -f {q(files.status)}; "
        f'if [ "$result" = 0 ]; then rm -f {q(files.log)}; '
        f"else echo \"runner exited $result\" >> {q(files.log)}; fi"
    )
    # Bootstrap callers consume the terminal files; detached online jobs remove them themselves after the runner
    # has finished reporting.
    status_tmp = q(f"{files.status}.tmp")
    wrapper = (
        # the runner starts only once the launch recorded this shell as the operation's launcher and let go of the lock;
        # a launch that could not record it leaves a wrapper that starts nothing
        f'flock {q(files.lock)} true && [ "$(cat {q(files.launcher)} 2>/dev/null)" = "$$" ] || {{ rm -rf "$D"; exit 1; }}; '
        f"rm -f {q(files.status)} {q(files.cancel)}; "
        f"{runner} & child=$!; "
        f"printf '%s\\n' \"$child\" > {q(files.pid)}; "
        "trap 'kill -TERM \"$child\" 2>/dev/null || true' TERM INT; "
        # the status first, published whole, and only then the staged tree: a waiter that finds the runner gone reads it
        f'wait "$child"; result=$?; printf \'%s\\n\' "$result" > {status_tmp} && mv {status_tmp} {q(files.status)}; '
        'rm -rf "$D"; '
        f"rm -f {q(files.spec)} {q(files.pid)} {q(files.cancel)}{terminal_cleanup}"
    )
    already = f'rm -rf -- {stage}; cat > /dev/null; echo {_ALREADY_LAUNCHED}; exit 0'
    return (
        # a lock file that cannot be created ends the shell through the `||`, not through a failed `exec`
        f"( : >> {q(files.lock)} ) 2>/dev/null && exec 9>>{q(files.lock)} && flock -x 9 || {{ rm -rf -- {stage}; exit 1; }}; "
        # lock files are never swept: one held or waited on for longer than the sweep's age would lose its exclusion
        f"find {q(REMOTE_OPERATION_DIRECTORY)} -maxdepth 1 \\( -name '*.done' -o -name '*.launcher' \\) "
        f"-mmin +{_FINISHED_MARKER_SWEEP_MINUTES} -delete 2>/dev/null; "
        f"if {_incumbent_condition(files)}; then {already}; fi; "
        # no result and no live launcher: whatever an earlier run left is stale, its helper container included
        f"rm -f {q(files.status)} {q(files.log)} {q(files.pid)} {q(files.spec)} {q(files.cancel)} {q(files.launcher)}; "
        f"/usr/bin/docker rm -f -- {shlex.quote(storage_helper_name(operation_id))} >/dev/null 2>&1 || true; "
        # the log is opened here, before the spec: a node that cannot create it gets no spec (no credentials left behind)
        f"if ( : > {q(files.log)} ) 2>/dev/null; then exec 8>>{q(files.log)}; else rm -rf -- {stage}; exit 1; fi; "
        f"(umask 077 && cat > {q(files.spec)}) || {{ rm -rf -- {stage} {q(files.spec)}; exit 1; }}; "
        f"D={stage} nohup /bin/sh -c {shlex.quote(wrapper)} >&8 2>&8 < /dev/null 8>&- 9>&- & "
        # acknowledged only once the launcher is recorded; without the record its wrapper starts nothing, so the spec goes
        f"printf '%s\\n' \"$!\" > {q(files.launcher)} || {{ rm -f {q(files.spec)}; exit 1; }}; echo {_LAUNCHED}"
    )


def _staged_directory(stdout: object) -> str:
    try:
        return executor_payload.staged_directory("storage", stdout)
    except ValueError as error:
        raise StorageOperationLaunchError(str(error)) from None


async def run_job_with_secrets(
    ssh_client: asyncssh.SSHClientConnection,
    purpose: str,
    files: tuple[str, ...],
    auth_token: str,
    volume_info,
    argv: Callable[[str], list[str]],
    log: str | None = None,
    **run_kwargs,
) -> None:
    """Stage a legacy tar job, then hand it its credentials file (upload_secrets_file) and start it: the credentials reach
    the node only once the job that reads and deletes them is in place. A failed start removes both."""
    staged = await ssh_client.run(executor_payload.stage_job(purpose, files), check=True)
    stage_dir = executor_payload.staged_directory(purpose, staged.stdout)
    try:
        # inside the job's own private directory: whatever fails (the upload's close, the start, the job), removing that
        # directory removes the credentials, and the job's shell removes it when it ends
        secrets_file = await upload_secrets_file(ssh_client, auth_token, volume_info, stage_dir)
        await ssh_client.run(executor_payload.run_job(stage_dir, files, argv(secrets_file), log=log), check=True, **run_kwargs)
    except Exception:
        await ssh_client.run(f"rm -rf -- {shlex.quote(stage_dir)}", check=False)
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
        '(*) if kill -0 "$pid" 2>/dev/null; then echo RUNNING; '
        # the runner has exited but its wrapper may still be publishing the status: wait for it, then read it again
        f"elif {_live(files.launcher, files)}; then echo RUNNING; "
        f"elif test -s {status_path}; then printf 'STATUS:'; cat {status_path}; "
        "else echo EXITED; fi;; esac; "
        # the wrapper publishes the status before it removes the pid, so a pid gone since the first read means the
        # status is there now: read it again before calling the runner not started
        f"elif test -s {status_path}; then printf 'STATUS:'; cat {status_path}; "
        "else echo STARTING; fi"
    )
    result = await ssh_client.run(command, check=False)
    return (result.stdout or "").strip()


async def cancel_storage_operation(
    ssh_client: asyncssh.SSHClientConnection,
    operation_id: UUID,
) -> None:
    files = StorageOperationFiles.for_operation(operation_id)
    # once the runner has read it, the spec holds only the reporter (DAH-6463)
    spec = await _read_operation_spec(ssh_client, files.spec)
    helper_name = storage_helper_name(operation_id)
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
        # the stopped helper still holds the repository credentials; remove it, not just stop it
        f"/usr/bin/docker rm -f -- {shlex.quote(helper_name)} >/dev/null 2>&1 || true; "
        "fi; "
        'kill -0 "$pid" 2>/dev/null && exit 3 || true; '
        "else "
        f"/usr/bin/docker stop --time 5 {shlex.quote(helper_name)} >/dev/null 2>&1 || true; "
        f"/usr/bin/docker rm -f -- {shlex.quote(helper_name)} >/dev/null 2>&1 || true; "
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


def _launch_failure_message(error: Exception) -> str:
    return f"Storage runner launch failed: {type(error).__name__}: {error}"[:2000]


async def report_launch_failure(
    operation_id: UUID,
    spec: Mapping[str, object],
    error: Exception,
) -> bool:
    return await _report_operation_status(
        operation_id, spec, status="FAILED", stage="LAUNCH", message=_launch_failure_message(error)
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


async def upload_secrets_file(
    ssh_client: asyncssh.SSHClientConnection, auth_token: str, volume_info, directory: str
) -> str:
    """Hand the legacy tar backup/restore scripts their credentials as a fresh 0600 file in `directory`, the job's own
    0700 staging directory (stage_job), so the file goes with it.

    Keeps the values out of the remote command line. The file is created exclusively with mode 0600; the script deletes
    it on read.
    """
    path = f"{directory}/secrets.json"
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
