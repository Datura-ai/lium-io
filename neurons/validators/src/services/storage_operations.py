from __future__ import annotations

import asyncio
import json
import logging
import shlex
import time
import weakref
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import aiohttp
import asyncssh

if TYPE_CHECKING:
    from services.redis_service import RedisService

logger = logging.getLogger(__name__)

REMOTE_OPERATION_DIRECTORY = PurePosixPath("/root/app/storage-operations")
REMOTE_RUNNER_PATH = "/root/app/src/storage_runner.py"
REMOTE_SECRETS_DIRECTORY = PurePosixPath("/root/app/storage-secrets")
# cancel markers and the per-operation launch/cancel lock live on the executor's reserve_data volume:
# /root/app's overlay upper is wiped on every executor start, and a marker must outlive a restart
REMOTE_CANCEL_DIRECTORY = PurePosixPath("/var/lium-reserve/storage-cancels")

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


class StorageOperationCancelledBeforeLaunch(StorageOperationLaunchError):
    pass


@dataclass(frozen=True)
class StorageOperationFiles:
    spec: PurePosixPath
    log: PurePosixPath
    pid: PurePosixPath
    status: PurePosixPath
    cancel: PurePosixPath
    lock: PurePosixPath
    start: PurePosixPath
    part: PurePosixPath

    @classmethod
    def for_operation(cls, operation_id: UUID) -> StorageOperationFiles:
        stem = str(operation_id)
        return cls(
            spec=REMOTE_OPERATION_DIRECTORY / f"{stem}.json",
            log=REMOTE_OPERATION_DIRECTORY / f"{stem}.log",
            pid=REMOTE_OPERATION_DIRECTORY / f"{stem}.pid",
            status=REMOTE_OPERATION_DIRECTORY / f"{stem}.status",
            cancel=REMOTE_CANCEL_DIRECTORY / f"{stem}.cancel",
            # one lock for every operation: each holder keeps it only for a check and a write
            lock=REMOTE_CANCEL_DIRECTORY / "launch.lock",
            # the recorded process's start time: with the PID it names one process, never one that reused the PID
            start=REMOTE_OPERATION_DIRECTORY / f"{stem}.start",
            # the spec while the launch receives it; one name per operation, so a cut launch's copy is overwritten
            part=REMOTE_OPERATION_DIRECTORY / f"{stem}.json.part",
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


# A repeated backend dispatch of one operation reaches this validator while the first is still between its state
# checks and its launch; both would pass the checks, and the second would rewrite the PID and spec the first's wrapper
# checks. In-process, like the rest of an operation's dispatch: launches of one operation run one after another, so a
# repeat finds the first launch recorded and returns it.
_launch_locks: weakref.WeakValueDictionary[UUID, asyncio.Lock] = weakref.WeakValueDictionary()


async def start_storage_operation(
    ssh_client: asyncssh.SSHClientConnection,
    python_path: str,
    operation_id: UUID,
    spec: Mapping[str, object],
    *,
    retain_terminal_artifacts: bool,
) -> StorageOperationFiles:
    lock = _launch_locks.get(operation_id)
    if lock is None:
        lock = _launch_locks[operation_id] = asyncio.Lock()
    async with lock:
        return await _start_storage_operation(
            ssh_client, python_path, operation_id, spec, retain_terminal_artifacts=retain_terminal_artifacts
        )


async def _start_storage_operation(
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
            f"install -d -m 0700 {shlex.quote(str(REMOTE_OPERATION_DIRECTORY))} "
            f"{shlex.quote(str(REMOTE_CANCEL_DIRECTORY))}",
            check=True,
        )
        # the cancel marker outlives the cancel: a dispatch that reaches the node after it never runs
        if await _remote_file_exists(ssh_client, files.cancel):
            raise StorageOperationCancelledBeforeLaunch("storage operation was cancelled before launch")
        existing_state = await _operation_state(ssh_client, files)
        if existing_state == "STARTING" and await _remote_file_exists(ssh_client, files.spec):
            # A repeated backend dispatch can arrive while the first wrapper is between
            # writing its spec and PID. Give that deterministic operation a chance to appear.
            for _ in range(20):
                await asyncio.sleep(0.25)
                existing_state = await _operation_state(ssh_client, files)
                if existing_state != "STARTING":
                    break
        if existing_state == "RUNNING" or existing_state.startswith("STATUS:"):
            return files
        if await _runner_alive(ssh_client, files):
            # an earlier dispatch's runner is still working, even before it wrote its PID or started
            # its helper; a second runner would interrupt it.
            # A helper whose runner died is an orphan: it is removed below and the run relaunched.
            return files
        # The spec (it holds the repository credentials) goes over the launch command's stdin and is written only
        # under the node's launch lock, after the operation has been checked again there, to one file per operation
        # that the launch's own exit, the next launch and the artifact cleanup all remove: a launch cut at any
        # point leaves no copy under a name nothing cleans.
        payload = json.dumps(dict(spec), separators=(",", ":"), sort_keys=True)
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
            # a failed run keeps its log: a runner that dies before reporting leaves no other trace
            else f"; rm -f {shlex.quote(str(files.status))}; "
            f'if [ "$result" = 0 ]; then rm -f {shlex.quote(str(files.log))}; '
            f"else echo \"runner exited $result\" >> {shlex.quote(str(files.log))}; fi"
        )
        # Bootstrap callers consume the terminal files; detached online jobs
        # remove them themselves after the runner has finished reporting.
        lock_path = shlex.quote(str(files.lock))
        pid_path = shlex.quote(str(files.pid))
        # The wrapper takes the launch lock before it starts the runner, so it runs only after the
        # launch has recorded its PID, and never once a cancel has marked the operation. It holds the
        # lock until the runner has started (the runner does not inherit it), so a cancel's marker
        # either stops the runner here or comes after it started, as for any running operation.
        wrapper = (
            # `command`: a lock file that cannot be opened must not end the shell before its status is written
            f"command exec 8>>{lock_path} 2>/dev/null; "
            "flock -w 30 8 && locked=1; "
            # only the wrapper the launch recorded as the operation's process group owns its files: one whose PID
            # write failed, or that another launch's record replaced, writes, removes and starts nothing
            f'[ "$(cat {pid_path} 2>/dev/null)" = "$$" ] || exit 0; '
            f"rm -f {shlex.quote(str(files.status))}; "
            f'if [ -n "$locked" ] && ! test -e {shlex.quote(str(files.cancel))}; then '
            f"{runner} 8>&- & child=$!; "
            "trap 'kill -TERM \"$child\" 2>/dev/null || true' TERM INT; exec 8>&-; "
            'wait "$child"; result=$?; '
            "else exec 8>&-; result=5; fi; "
            f"printf '%s\\n' \"$result\" > {shlex.quote(str(files.status))}; "
            f"rm -f {shlex.quote(str(files.spec))} {pid_path} {shlex.quote(str(files.start))}{terminal_cleanup}"
        )
        # Under the launch lock (the cancel takes it too): refuse if a cancel has marked it, else
        # start the wrapper as its own process group and record that group before releasing the
        # lock, so a cancel that comes after always finds every process of the operation.
        # The wrapper must not inherit the held lock (9>&-).
        part_path = shlex.quote(str(files.part))
        start_path = shlex.quote(str(files.start))
        launch_command = (
            _owned_shell_function(files)
            # a cut session (HUP) or a kill ends the shell through its EXIT trap, which removes the partial spec
            + f"trap 'rm -f {part_path}' EXIT; trap 'exit 1' HUP INT TERM PIPE; "
            f"command exec 9>>{lock_path} 2>/dev/null; flock -w 30 9 || exit 6; "
            f"if test -e {shlex.quote(str(files.cancel))}; then rm -f {shlex.quote(str(files.spec))}; exit 5; fi; "
            # another launch (this or another validator process) got there first: a live recorded wrapper (or a
            # runner recorded before process groups) is the operation, and a written status is its result;
            # their PID, spec and status stay as they are
            f"test -s {shlex.quote(str(files.status))} && exit 0; "
            f'if p=$(cat {pid_path} 2>/dev/null) && [ -n "$p" ] && owned "$p" '
            '&& { kill -0 -"$p" 2>/dev/null || kill -0 "$p" 2>/dev/null; }; then exit 0; fi; '
            # nothing of the operation runs: a dead run's files and its orphaned helper go, then this spec is written
            f"rm -f {' '.join(shlex.quote(str(path)) for path in (files.status, files.log, files.pid, files.start))} "
            f"{shlex.quote(str(files.spec))}; "
            f"/usr/bin/docker rm -f -- {shlex.quote(storage_helper_name(operation_id))} >/dev/null 2>&1; "
            # all of it or nothing: a session cut mid-transfer ends stdin early, and that spec must not run
            f"{{ (umask 077; cat > {part_path}) && test \"$(wc -c < {part_path})\" -eq {len(payload.encode())}; }} || exit 9; "
            f"mv {part_path} {shlex.quote(str(files.spec))} || exit 8; "
            f"setsid nohup /bin/sh -c {shlex.quote(wrapper)} "
            f"> {shlex.quote(str(files.log))} 2>&1 < /dev/null 9>&- & "
            # the start time first: a PID file is never there without the start time that names its process
            "wrapper=$!; started_at=$(sed 's/.*) //' /proc/\"$wrapper\"/stat 2>/dev/null | cut -d' ' -f20); "
            f"if ! {{ test -n \"$started_at\" && printf '%s\\n' \"$started_at\" > {start_path} "
            f"&& printf '%s\\n' \"$wrapper\" > {pid_path}; }}; then "
            # the wrapper is still waiting for this lock, so nothing of the operation has run yet
            'kill -KILL -"$wrapper" 2>/dev/null; '
            f"rm -f {shlex.quote(str(files.spec))} {pid_path} {start_path}; exit 7; fi"
        )
        launched = await ssh_client.run(launch_command, check=False, input=payload)
        if launched.exit_status == 5:
            raise StorageOperationCancelledBeforeLaunch("storage operation was cancelled before launch")
        if launched.exit_status != 0:
            raise StorageOperationLaunchError(f"storage operation launch failed (exit {launched.exit_status})")
        return files
    except Exception as error:
        if not isinstance(error, StorageOperationCancelledBeforeLaunch):  # the cancel already reported it
            await report_launch_failure(operation_id, spec, error)
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
        _owned_shell_function(files)
        + f"if test -s {status_path}; then printf 'STATUS:'; cat {status_path}; "
        f"elif test -s {pid_path}; then pid=$(cat {pid_path}); "
        "case \"$pid\" in (*[!0-9]*|'') echo INVALID;; "
        '(*) if owned "$pid" && kill -0 "$pid" 2>/dev/null; then echo RUNNING; else echo EXITED; fi;; esac; '
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
    # the marker is what stops a later launch, so a cancel that cannot write it fails, but only
    # after it has still stopped the runner and helper
    # under the launch lock, so a launch either sees this marker or has recorded its process group
    # first; the PID is that group's leader, so every check and signal below covers the whole group
    lock_path = shlex.quote(str(files.lock))
    command = (
        _owned_shell_function(files)
        + "marked=0; "
        # `command`: a failed open of the lock fails the step instead of ending the shell before the stop fallback
        f"{{ install -d -m 0700 {shlex.quote(str(REMOTE_CANCEL_DIRECTORY))} && command exec 9>>{lock_path} 2>/dev/null "
        f"&& flock -w 30 9 && touch {cancel_path}; }} || marked=4; "
        # read under the lock: a launch that took it first has written its PID by now; release it before
        # the bounded wait so other operations' launches never queue behind this one
        f"pid=$(cat {pid_path} 2>/dev/null); exec 9>&-; "
        "case \"$pid\" in (*[!0-9]*) exit 2;; esac; "
        # a recorded PID the host has since given to another process is not signalled; only the helper is removed.
        # A signal needs the recorded start time to match (`started`): a PID recorded without one (before this
        # release) is only waited for, as its runner stops on the marker itself
        'if test -n "$pid" && owned "$pid"; then '
        # an operation launched before process groups recorded its runner's PID, which leads no group; owned is
        # checked again before every signal, as the PID can be reused while this waits
        'alive() { owned "$pid" && { kill -0 -"$pid" 2>/dev/null || kill -0 "$pid" 2>/dev/null; }; }; '
        'attempt=0; while alive && test "$attempt" -lt 100; do '
        "sleep 0.25; attempt=$((attempt + 1)); done; "
        'if alive; then '
        f"/usr/bin/docker stop --time 5 {shlex.quote(helper_name)} >/dev/null 2>&1 || true; "
        'started "$pid" && { kill -TERM -"$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null; }; sleep 2; '
        'if alive && started "$pid"; then kill -KILL -"$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true; '
        'attempt=0; while alive && test "$attempt" -lt 20; do '
        "sleep 0.1; attempt=$((attempt + 1)); done; fi; "
        # the stopped helper still holds the repository credentials; remove it, not just stop it
        f"/usr/bin/docker rm -f -- {shlex.quote(helper_name)} >/dev/null 2>&1 || true; "
        "fi; "
        'alive && exit 3 || true; '
        "else "
        f"/usr/bin/docker stop --time 5 {shlex.quote(helper_name)} >/dev/null 2>&1 || true; "
        f"/usr/bin/docker rm -f -- {shlex.quote(helper_name)} >/dev/null 2>&1 || true; "
        "fi; "
        'exit "$marked"'
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


def _owned_shell_function(files: StorageOperationFiles) -> str:
    """Shell `started PID` and `owned PID`.

    `started`: the PID's process started at the time the launch recorded with it, so it is the process the launch
    started and not one the host gave the PID to since. Only this allows a signal.

    `owned`: the PID is still the operation's for a state read or a launch's check. A PID with no process is owned:
    Linux does not reuse a PID while a process group of that ID is alive, so a group found under it is still the
    operation's. With a recorded start time it must match; a PID recorded without one (before this release) is
    owned while its command line carries the operation's spec path, as every process of the operation does."""
    start = shlex.quote(str(files.start))
    return (
        f'started() {{ s=$(cat {start} 2>/dev/null) && test -n "$s" '
        "&& test \"$(sed 's/.*) //' /proc/\"$1\"/stat 2>/dev/null | cut -d' ' -f20)\" = \"$s\"; }; "
        'owned() { test ! -e /proc/"$1" || started "$1" || { ! test -s '
        + start
        + " && tr '\\000' ' ' < /proc/\"$1\"/cmdline 2>/dev/null | grep -qF -- "
        + shlex.quote(str(files.spec))
        + "; }; }; "
    )


async def _remote_file_exists(
    ssh_client: asyncssh.SSHClientConnection,
    path: PurePosixPath,
) -> bool:
    result = await ssh_client.run(f"test -e {shlex.quote(str(path))}", check=False)
    return result.exit_status == 0


async def _runner_alive(
    ssh_client: asyncssh.SSHClientConnection,
    files: StorageOperationFiles,
) -> bool:
    # "[-]-spec" keeps pgrep from matching this command's own shell
    runner_pattern = f"[-]-spec {files.spec}"
    result = await ssh_client.run(
        f"pgrep -f -- {shlex.quote(runner_pattern)} >/dev/null",
        check=False,
    )
    return result.exit_status == 0


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
            # the cancel marker stays: the operation id is never reused, and a late launch must see it
            # shortcut: one empty marker per cancelled operation stays on the reserve volume; sweep old ones if they pile up
            for path in (files.status, files.log, files.pid, files.start, files.spec, files.part)
        ),
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
