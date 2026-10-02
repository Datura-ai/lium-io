"""LIUM-65 (DAH-3980) — volume create → `docker run` → running check as ONE SSH exec.

On the cached-image, encrypted-volume, no-filler create these three steps cost the rent a round
trip to the executor each (and the SDK's `docker run` several). Here they run as one exec of a
fixed python3 stdlib script on the executor's docker socket (the connector's SSH lands in the
executor container: python 3, the docker socket, no jq).

The command line is a constant. Every value of the rent (image, env, ports, names, sizes) travels
as ONE JSON document on stdin: the Engine API bodies docker-py would send, built locally from the
same `ContainerRunSpec`. The script never builds a shell string; the values reach dockerd only as
JSON bodies or URL-quoted path parts.

The script prints one JSON line per finished step (`{"step": "volume_created", "ms": 41}`) and, on
a failure, removes what it created and prints `{"step": "failed", "failed_step": …}`. It touches
only names that were absent when it started, records each creation on the host before making it,
and removes its own work when the channel drops or SIGTERM comes. After a lost reply the same
script, as `settle`, waits for the attempt to exit and adopts or removes what the record names.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import shlex
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import asyncssh

from core.utils import _m, get_extra_info
from payload_models.payloads import ProfilerStep, ProfilerStepName
from services.rental_docker_sdk import ContainerRunSpec, build_container_create_body

logger = logging.getLogger(__name__)

# The step lines the script prints, in order. `network` (the isolated rental bridge exists, as
# `_ensure_rental_network_sync` requires) runs first and creates nothing.
NETWORK_STEP = "network"
VOLUME_CREATED_STEP = "volume_created"
CONTAINER_CREATED_STEP = "container_created"
STARTED_STEP = "started"
RUNNING_STEP = "running"
FAILED_LINE = "failed"

# volume ≤ 180 s (`_LOCAL_VOLUME_TIMEOUT_MAX_SEC`) + create 60 + start 60 + running check 10, plus margin
ONE_EXEC_CREATE_HOST_DEADLINE_SEC = 330
ONE_EXEC_CREATE_CONNECTOR_DEADLINE_SEC = ONE_EXEC_CREATE_HOST_DEADLINE_SEC + 15
# a settle waits for the attempt's lock (the script's SETTLE_WAIT_SEC), then removes
ONE_EXEC_SETTLE_CONNECTOR_DEADLINE_SEC = 360 + 30

ONE_EXEC_CREATE_SCRIPT = r'''
import fcntl, hashlib, http.client, json, os, signal, socket, sys, time
from urllib.parse import quote

DOCKER_SOCKET = "/var/run/docker.sock"
CALL_TIMEOUT_SEC = 60
RUNNING_CHECK_SEC = 10
SETTLE_WAIT_SEC = 360
ICC_OPTION = "com.docker.network.bridge.enable_icc"
CREATE_FIELDS = {"action", "attempt", "network", "volume", "volume_timeout_s", "container_name", "container"}
SETTLE_FIELDS = {"action", "attempt", "container_name"}


class StepFailed(Exception):
    def __init__(self, step, error, **details):
        super().__init__(str(error))
        self.step = step
        self.details = details


class ReplyLost(Exception):
    # stdout is gone (the connector closed the channel) or SIGTERM came: nobody reads the rest
    pass


def on_sigterm(signum, frame):
    raise ReplyLost("terminated")


class DockerSocketConnection(http.client.HTTPConnection):
    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(DOCKER_SOCKET)


def call(method, path, body=None, timeout=CALL_TIMEOUT_SEC):
    connection = DockerSocketConnection("docker", timeout=timeout)
    try:
        payload = None if body is None else json.dumps(body)
        connection.request(method, path, payload, {"Content-Type": "application/json"})
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


def answer(method, path, body=None, timeout=CALL_TIMEOUT_SEC):
    status, raw = call(method, path, body, timeout)
    if status >= 300:
        try:
            message = json.loads(raw)["message"]
        except Exception:
            message = raw.decode("utf-8", "replace")
        raise RuntimeError("%d %s" % (status, message))
    return json.loads(raw) if raw else None


def q(value):
    return quote(value, safe="")


def emit(**fields):
    try:
        sys.stdout.write(json.dumps(fields) + "\n")
        sys.stdout.flush()
    except OSError as error:
        raise ReplyLost(error)


def timed(step, action):
    started = time.monotonic()
    try:
        value = action()
    except (StepFailed, ReplyLost):
        raise
    except Exception as error:
        raise StepFailed(step, error)
    emit(step=step, ms=round((time.monotonic() - started) * 1000))
    return value


def hold_attempt_lock(container_name, wait_sec):
    # one attempt per pod at a time; a settle holding it knows the create attempt has exited
    base = "/tmp/lium-one-exec-" + hashlib.sha256(container_name.encode()).hexdigest()[:32]
    handle = open(base + ".lock", "a")
    deadline = time.monotonic() + wait_sec
    while True:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return handle, base + ".json"
        except BlockingIOError:
            if time.monotonic() >= deadline:
                raise RuntimeError("another attempt holds the lock")
            time.sleep(0.1)


class Record:
    # what this attempt may have made, written BEFORE each creation, so a settle finds it all
    def __init__(self, path, state):
        self.path = path
        self.state = state

    def save(self, **changes):
        self.state.update(changes)
        with open(self.path + ".tmp", "w") as handle:
            json.dump(self.state, handle)
        os.replace(self.path + ".tmp", self.path)


def is_absent(path):
    return call("GET", path)[0] == 404


def require_isolated_bridge(name):
    network = answer("GET", "/networks/" + q(name))
    if network.get("Driver") != "bridge" or (network.get("Options") or {}).get(ICC_OPTION) != "false":
        raise RuntimeError("network %s is not a bridge with inter-container traffic off" % name)


def logs_tail(container_id):
    try:
        status, raw = call("GET", "/containers/%s/logs?stdout=1&stderr=1&tail=50" % q(container_id))
    except Exception as error:
        return "(logs unreadable: %s)" % error
    # without a TTY the stream is multiplexed: 8-byte frame headers before each chunk
    chunks, at = [], 0
    while at + 8 <= len(raw) and raw[at] in (0, 1, 2) and raw[at + 1:at + 4] == b"\0\0\0":
        size = int.from_bytes(raw[at + 4:at + 8], "big")
        chunks.append(raw[at + 8:at + 8 + size])
        at += 8 + size
    return (b"".join(chunks) if chunks else raw).decode("utf-8", "replace")[-2000:]


def wait_running(container_id):
    deadline = time.monotonic() + RUNNING_CHECK_SEC
    while True:
        status, raw = call("GET", "/containers/%s/json" % q(container_id))
        if status == 404:
            raise StepFailed("running", "container is gone", vanished=True)
        state = (json.loads(raw).get("State") or {}) if status == 200 else {}
        if state.get("Running"):
            return
        if time.monotonic() >= deadline:
            seen = {key: state.get(key) for key in ("Status", "ExitCode", "OOMKilled", "Error")}
            raise StepFailed("running", "container is not running", state=seen, logs=logs_tail(container_id))
        time.sleep(0.2)


def remove_recorded(record):
    # both names were absent before this attempt, so whatever answers to them now is its own
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    state = record.state
    try:
        gone = True
        container = state.get("container_id") or state.get("container")
        if container:
            gone = call("DELETE", "/containers/%s?force=1&v=1" % q(container))[0] in (204, 404)
        if state.get("volume"):
            gone = call("DELETE", "/volumes/" + q(state["volume"]))[0] in (204, 404) and gone
    except Exception:
        gone = False
    record.save(outcome="cleaned" if gone else "cleanup_failed")
    return gone


def create(request):
    try:
        lock, record_path = hold_attempt_lock(request["container_name"], 0)
    except Exception as error:
        emit(step="failed", failed_step="lock", error=str(error), cleaned=True)
        return 1
    record = Record(record_path, {"attempt": request["attempt"], "outcome": None})
    container_name, volume_name = request["container_name"], request["volume"]["Name"]
    try:
        timed("network", lambda: require_isolated_bridge(request["network"]))
        if not is_absent("/volumes/" + q(volume_name)):
            raise StepFailed("volume_exists", "volume %s is already on the host" % volume_name)
        if not is_absent("/containers/%s/json" % q(container_name)):
            raise StepFailed("container_exists", "container %s is already on the host" % container_name)
        record.save(volume=volume_name)
        timed("volume_created", lambda: answer(
            "POST", "/volumes/create", request["volume"], request["volume_timeout_s"]
        ))
        record.save(container=container_name)
        container_id = timed("container_created", lambda: answer(
            "POST", "/containers/create?name=" + q(container_name), request["container"]
        )["Id"])
        record.save(container_id=container_id)
        timed("started", lambda: answer("POST", "/containers/%s/start" % q(container_id)))
        timed("running", lambda: (wait_running(container_id), record.save(outcome="running")))
    except StepFailed as failure:
        cleaned = remove_recorded(record)
        try:
            emit(step="failed", failed_step=failure.step, error=str(failure)[-2000:], cleaned=cleaned, **failure.details)
        except ReplyLost:
            pass
        return 1
    except ReplyLost:
        remove_recorded(record)
        return 3
    return 0


def settle(request):
    # after a lost reply: wait out the create attempt, then adopt its running pod or remove its own
    lock, record_path = hold_attempt_lock(request["container_name"], SETTLE_WAIT_SEC)
    try:
        with open(record_path) as handle:
            state = json.load(handle)
    except (OSError, ValueError):
        state = {}
    if state.get("attempt") != request["attempt"]:
        emit(step="settled", result="gone")
        return 0
    if state.get("outcome") == "running":
        status, raw = call("GET", "/containers/%s/json" % q(state["container_id"]))
        if status == 200 and (json.loads(raw).get("State") or {}).get("Running"):
            emit(step="settled", result="adopt")
            return 0
    gone = remove_recorded(Record(record_path, state))
    emit(step="settled", result="gone" if gone else "not_gone")
    return 0


def main():
    signal.signal(signal.SIGTERM, on_sigterm)
    request = json.load(sys.stdin)
    action = request.get("action") if isinstance(request, dict) else None
    if action == "create" and set(request) == CREATE_FIELDS:
        return create(request)
    if action == "settle" and set(request) == SETTLE_FIELDS:
        return settle(request)
    emit(step="failed", failed_step="request", error="unexpected request fields", cleaned=True)
    return 2


sys.exit(main())
'''

ONE_EXEC_CREATE_COMMAND = (
    f"timeout -k 5 {ONE_EXEC_CREATE_HOST_DEADLINE_SEC} python3 -c {shlex.quote(ONE_EXEC_CREATE_SCRIPT)}"
)


@dataclass(slots=True)
class OneExecCreateReply:
    """What the exec reported. `failure` is the script's `failed` line; neither a failure nor a
    `running` step means the script did not answer as designed (no python3, killed, garbage)."""

    step_ms: dict[str, int] = field(default_factory=dict)
    failure: dict | None = None
    exit_status: int | None = None
    wall_ms: int = 0
    stderr_tail: str = ""
    # the reply was lost, and the settle found this attempt's container running
    adopted: bool = False

    @property
    def ran_to_running(self) -> bool:
        return self.adopted or (self.failure is None and RUNNING_STEP in self.step_ms and self.exit_status == 0)

    @property
    def left_host_as_found(self) -> bool:
        # the script removed all it made and said so, or the shell could not start python3 at all
        return bool(self.failure and self.failure.get("cleaned")) or (
            self.exit_status in (126, 127) and not self.step_ms
        )


def build_one_exec_create_request(
    *,
    run_spec: ContainerRunSpec,
    volume_name: str,
    volume_driver: str | None,
    volume_driver_opts: dict[str, str] | None,
    volume_timeout_s: int,
) -> dict:
    return {
        "action": "create",
        # names this attempt's record on the host, so a settle never touches another attempt's
        "attempt": secrets.token_hex(8),
        "network": run_spec.network,
        # the body docker-py's `create_volume` sends
        "volume": {"Name": volume_name, "Driver": volume_driver, "DriverOpts": volume_driver_opts},
        "volume_timeout_s": volume_timeout_s,
        "container_name": run_spec.name,
        "container": build_container_create_body(run_spec),
    }


async def run_one_exec_create(
    ssh_client: asyncssh.SSHClientConnection,
    request: dict,
    on_step: Callable[[str], Awaitable[None]],
    log_extra: dict,
) -> OneExecCreateReply:
    # the script's step lines as they arrive; a dropped session or the deadline ends the reply early
    reply = OneExecCreateReply()
    started = time.monotonic()

    async def converse() -> None:
        async with await ssh_client.create_process(ONE_EXEC_CREATE_COMMAND) as process:
            process.stdin.write(json.dumps(request))
            process.stdin.write_eof()
            async for line in process.stdout:
                try:
                    fields = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(fields, dict):
                    continue
                if fields.get("step") == FAILED_LINE:
                    reply.failure = fields
                elif isinstance(fields.get("step"), str) and isinstance(fields.get("ms"), int):
                    reply.step_ms[fields["step"]] = fields["ms"]
                    await on_step(fields["step"])
            reply.stderr_tail = (await process.stderr.read())[-500:]
            reply.exit_status = (await process.wait()).exit_status

    try:
        await asyncio.wait_for(converse(), timeout=ONE_EXEC_CREATE_CONNECTOR_DEADLINE_SEC)
    except (asyncio.TimeoutError, asyncssh.Error, OSError) as exc:
        reply.stderr_tail = f"{type(exc).__name__}: {exc}"
    reply.wall_ms = round((time.monotonic() - started) * 1000)
    logger.info(
        _m(
            "ONE_EXEC_CREATE_REPLY",
            extra=get_extra_info({
                **log_extra,
                "host_step_ms": reply.step_ms,
                "wall_ms": reply.wall_ms,
                "exit_status": reply.exit_status,
                "failed_step": reply.failure.get("failed_step") if reply.failure else None,
                "error": reply.failure.get("error") if reply.failure else None,
                "cleaned": reply.failure.get("cleaned") if reply.failure else None,
                "stderr_tail": reply.stderr_tail,
            }),
        )
    )
    return reply


async def settle_one_exec_create(
    ssh_client: asyncssh.SSHClientConnection, create_request: dict, log_extra: dict
) -> str | None:
    """After a lost or unclear reply: wait until the create attempt has exited on the host, then
    `adopt` (its container runs) or remove what its record says it made — `gone` once dockerd
    confirms, `not_gone` otherwise. None when the settle itself could not run or answer."""
    request = {
        "action": "settle",
        "attempt": create_request["attempt"],
        "container_name": create_request["container_name"],
    }
    try:
        completed = await asyncio.wait_for(
            ssh_client.run(ONE_EXEC_CREATE_COMMAND, input=json.dumps(request), check=False),
            timeout=ONE_EXEC_SETTLE_CONNECTOR_DEADLINE_SEC,
        )
        result = json.loads(str(completed.stdout).strip().splitlines()[-1]).get("result")
    except (asyncio.TimeoutError, asyncssh.Error, OSError, ValueError, IndexError, AttributeError) as exc:
        result, completed = None, exc
    logger.warning(
        _m(
            "ONE_EXEC_CREATE_SETTLED",
            extra=get_extra_info({**log_extra, "result": result, "detail": str(completed)[-500:]}),
        )
    )
    return result if result in ("adopt", "gone", "not_gone") else None


def one_exec_profiler_rows(reply: OneExecCreateReply) -> list[ProfilerStep]:
    """Today's three rows with the host's own durations. The volume row also carries the exec's
    start and its one round trip, so the profile still sums to the connector's wall time."""
    run_ms = sum(reply.step_ms.get(step, 0) for step in (NETWORK_STEP, CONTAINER_CREATED_STEP, STARTED_STEP))
    running_ms = reply.step_ms.get(RUNNING_STEP, 0)
    return [
        ProfilerStep(
            name=ProfilerStepName.DOCKER_VOLUME_CREATION,
            duration=max(reply.wall_ms - run_ms - running_ms, 0),
        ),
        ProfilerStep(name=ProfilerStepName.DOCKER_RUN, duration=run_ms),
        ProfilerStep(name=ProfilerStepName.CONTAINER_RUNNING_CHECK, duration=running_ms),
    ]
