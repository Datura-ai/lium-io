"""DAH-4663: a renter locks (or resets) the graphics clocks of their own pod's GPUs.

Run over the validator's SSH session into the executor (the host's `docker` and `nvidia-smi`, as in
services/gpu_reset.py), the PEARL power-cap pattern of services/gpu_power_limit.py:

    prechecks (change nothing) -> record in Redis -> nvidia-smi -i <uuid> -pm 1 -> clamp to the GPU's
        supported clocks -> nvidia-smi -i <uuid> -lgc min,max -> read the clocks back

Safety invariants, each pinned by a test:
- the GPUs touched are a subset of BOTH the container's DeviceRequests GPUs and the backend's GPUs;
- `-lgc` / `-rgc` never run without `-i`, and the power limit (`-pl`) is never touched;
- every lock is recorded (`gpu_clock_lock:<uuid>`, `gpu_clock_lock_pod:<pod_id>:<miner_hotkey>:<executor_id>`) before
  it is set, so the pod's delete and the safety net before any create can `-rgc` it;
- one clock change per GPU at a time (`_holding`): an owner read, its `-rgc`/`-lgc` and its record write run
  under the GPU's lock, so a late delete of one pod never resets or forgets another pod's lock.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import shlex
import time
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from payload_models.payloads import GpuClockOutcome, GpuResetErrorCode
from services.gpu_reset import (
    DOCKER,
    HOST_GPUS_COMMAND,
    GpuResetRefused,
    _run,
    _tail,
    parse_host_gpus,
    pod_gpus,
)

if TYPE_CHECKING:
    from services.redis_service import RedisService
    from services.task.runner import SSHCommandRunner

logger = logging.getLogger(__name__)

CLOCK_LOCK_KEY_PREFIX = "gpu_clock_lock:"
# each nvidia-smi of a reset (-rgc, -rac, -rmc) is killed on the host past this, so every GPU answers in time
PER_RESET_TIMEOUT_SECONDS = 15
_APP_MEM_FLAGS = ("-rac", "-rmc")
# one GPU's three resets run in a row, all GPUs at once; with a margin the call answers before the runner gives up
_RESET_CALL_TIMEOUT_SECONDS = (1 + len(_APP_MEM_FLAGS)) * PER_RESET_TIMEOUT_SECONDS + 10
# 3 = "not supported": a GPU without application or memory clock control
_APP_MEM_RESET_OK_EXITS = {"0", "3"}
# 6 = NVML "not found" (no such clock setting); only when the same GPU answered its -rgc, so the GPU itself was found.
# Ada GeForce answers it to -rac/-rmc on every GPU (application clocks exist on Maxwell GeForce only, nvidia-smi docs)
_APP_MEM_NOT_FOUND_EXIT = "6"
# 137: the host's timeout killed that nvidia-smi, so it gave no verdict
_KILLED_EXIT = "137"
CLOCK_LOCK_POD_KEY_PREFIX = "gpu_clock_lock_pod:"


def _rgc_failure(code: str) -> str:
    return f"nvidia-smi -rgc failed (exit {code})"


# a completed -rgc refusal by a GPU whose clocks cannot be changed from here: 3 "not supported", 4 "insufficient
# permissions" (a passthrough VM), as -lgc answers for the lock (_record_and_lock): no clock lock to undo
_NO_CLOCK_CONTROL = {_rgc_failure("3"), _rgc_failure("4")}
# no "<uuid> <exit>" line for the GPU: the SSH call timed out or lost its answer (the runner answers -1, no output)
_NO_ANSWER = "nvidia-smi -rgc gave no answer"
# one per GPU uuid this validator ever changed; a few thousand at most, never freed
_GPU_LOCKS: dict[str, asyncio.Lock] = {}


@asynccontextmanager
async def _holding(uuids: list[str]):
    """Hold the clock lock of each GPU (in uuid order, so two holders never deadlock). The per-pod lock does not cover
    this: a delete of pod A and a lock of pod B on the same GPU are different pods (review)."""
    async with AsyncExitStack() as stack:
        for uuid in sorted(set(uuids)):
            await stack.enter_async_context(_GPU_LOCKS.setdefault(uuid, asyncio.Lock()))
        yield


@dataclass
class GpuClocksReport:
    ok: bool
    error_code: GpuResetErrorCode | None = None
    gpus: list[GpuClockOutcome] = field(default_factory=list)
    detail: str | None = None


def _lock_key(uuid: str) -> str:
    return f"{CLOCK_LOCK_KEY_PREFIX}{uuid}"


def _pod_key(pod_id: str, miner_hotkey: str, executor_id: str) -> str:
    # DAH-6540 (review): one index per pod on one executor of one miner (an executor id is unique per miner only), the
    # scope of its pod GPU operation lock, so a moved pod's delete and its lock on the new node never share an index
    return f"{CLOCK_LOCK_POD_KEY_PREFIX}{pod_id}:{miner_hotkey}:{executor_id}"


def _owned_by(record: dict, pod_id: str, miner_hotkey: str, executor_id: str) -> bool:
    # shortcut: a record written before the hotkey was recorded has none and matches any miner; drop it with the
    # legacy index once its 30-day rentals are gone
    return (
        record.get("pod_id") == pod_id
        and record.get("executor_id") == executor_id
        and record.get("miner_hotkey", miner_hotkey) == miner_hotkey
    )


def _legacy_pod_key(pod_id: str) -> str:
    # shortcut: the per-pod index written before DAH-6540; read at teardown only, drop once its 30-day rentals are gone
    return f"{CLOCK_LOCK_POD_KEY_PREFIX}{pod_id}"


def build_clock_command(
    action: Literal["lock", "reset"],
    uuid: str,
    *,
    pod_gpu_uuids: set[str] | None = None,
    min_mhz: int | None = None,
    max_mhz: int | None = None,
) -> str:
    """The one place `-lgc` / `-rgc` are spelled: always with `-i <one GPU>`, only on one of the pod's GPUs."""
    if not uuid.startswith("GPU-"):
        raise ValueError(f"refusing a clock change without -i <GPU uuid>: {uuid!r}")
    if pod_gpu_uuids is not None and uuid not in pod_gpu_uuids:
        raise ValueError("refusing to change the clocks of a GPU outside the pod")
    if action == "reset":
        return f"nvidia-smi -i {shlex.quote(uuid)} -rgc"
    return f"nvidia-smi -i {shlex.quote(uuid)} -lgc {int(min_mhz)},{int(max_mhz)}"


def _reset_command(uuids: list[str], pod_gpu_uuids: set[str] | None, results: str) -> str:
    """One job per GPU, all GPUs at once: -rgc, then -rac (application clocks: -ac, or any process once the host ran
    -acp UNRESTRICTED), then -rmc (locked memory clocks: -lmc), each echoing "<uuid> <exit>" (-rgc) or
    "<uuid> <flag> <exit>", so a create never hands a renter clocks the host or the last pod lowered. Every nvidia-smi
    is the direct child of its own SIGKILL timeout, so the call answers within three resets whatever the GPU count:
    the -rgc runs used to go one GPU after another with no bound of their own, and on an 8x H200 node they and the
    -rac/-rmc batch outlasted the runner's budget, so the call gave no answer at all and no GPU had a verdict (prod
    25 Nov). The kill is per nvidia-smi and nothing wraps the batch: uutils timeout (Ubuntu 26.04) signals only its
    direct child and GNU timeout gives each inner timeout its own process group, so an outer kill left resets running
    (lio#2362). `wait` returns only once every reset has exited. A killed reset echoes 137. Each line is appended to
    the host file `results` the moment its reset ends and the file is printed at the end, so if the call overruns the
    runner's budget (its answer lost), the GPUs that answered are read back from it and only the others are unknown;
    the file stays unless the print reached the caller."""
    per_reset = f"timeout -s KILL {PER_RESET_TIMEOUT_SECONDS:g}"
    quoted = shlex.quote(results)
    jobs = []
    for uuid in uuids:
        rgc = build_clock_command("reset", uuid, pod_gpu_uuids=pod_gpu_uuids)
        steps = [f"{per_reset} {rgc} >/dev/null 2>&1; echo {shlex.quote(uuid)} $? >> {quoted}"] + [
            f"{per_reset} nvidia-smi -i {shlex.quote(uuid)} {flag} >/dev/null 2>&1; "
            f"echo {shlex.quote(uuid)} {flag} $? >> {quoted}"
            for flag in _APP_MEM_FLAGS
        ]
        jobs.append("( " + "; ".join(steps) + " ) &")
    return f"sh -c {shlex.quote(' '.join(jobs) + f' wait; cat {quoted} && rm -f {quoted}')}"


def _app_mem_outcome(code: str, rgc_exit: str | None) -> str:
    if code in _APP_MEM_RESET_OK_EXITS:
        return "ok" if code == "0" else "unsupported"
    if code == _APP_MEM_NOT_FOUND_EXIT and rgc_exit not in (None, _KILLED_EXIT, _APP_MEM_NOT_FOUND_EXIT):
        return "not_found"
    return "no_answer" if code in ("unknown", _KILLED_EXIT) else "failed"


def _log_reset_outcome(
    purpose: str, pod_id: str | None, uuids: list[str], lines: list[list[str]], failures: dict[str, str | None], result
) -> None:
    """One line per reset call, success too, so the resets can be counted; plus one warning per -rac/-rmc that failed
    or gave no answer (never changes a GPU's -rgc verdict or its lock record)."""
    rgc_exits = {p[0]: p[1] for p in lines if len(p) == 2}
    exits = {(p[0], p[1]): p[2] for p in lines if len(p) >= 3}
    counts = dict.fromkeys(("ok", "unsupported", "not_found", "failed", "no_answer"), 0)
    for uuid in uuids:
        for flag in _APP_MEM_FLAGS:
            code = exits.get((uuid, flag), "unknown")
            outcome = _app_mem_outcome(code, rgc_exits.get(uuid))
            counts[outcome] += 1
            if outcome in ("failed", "no_answer"):
                logger.warning("gpu_clock_action=rac_rmc_failed gpu=%s flag=%s exit=%s", uuid, flag, code)
    rgc = list(failures.values())
    logger.info(
        "gpu_clock_action=reset_outcome purpose=%s pod=%s gpus=%d rgc_ok=%d rgc_failed=%d rgc_no_answer=%d "
        "app_mem_ok=%d app_mem_unsupported=%d app_mem_not_found=%d app_mem_failed=%d app_mem_no_answer=%d "
        "call_exit=%s call_error=%s duration_ms=%s",
        purpose, pod_id or "-", len(uuids), rgc.count(None), sum(1 for f in rgc if f not in (None, _NO_ANSWER)),
        rgc.count(_NO_ANSWER), counts["ok"], counts["unsupported"], counts["not_found"], counts["failed"],
        counts["no_answer"], getattr(result, "exit_code", "-"), getattr(result, "error_type", None) or "-",
        getattr(result, "duration_ms", "-"),
    )


def clamp_range(min_mhz: int | None, max_mhz: int | None, supported: list[int]) -> tuple[int, int]:
    low, high = min(supported), max(supported)
    lo = low if min_mhz is None else max(low, min(high, min_mhz))
    hi = high if max_mhz is None else max(low, min(high, max_mhz))
    return min(lo, hi), hi  # a min above the max locks at the max


def _parse_mhz(text: str) -> list[int]:
    values = []
    for line in text.splitlines():
        try:
            values.append(round(float(line.strip().split()[0])))
        except (ValueError, IndexError):
            continue
    return values


async def _pod_gpus_to_lock(
    runner: SSHCommandRunner,
    *,
    container_name: str,
    pod_gpu_uuids: list[str],
    gpu_indices: list[int] | None,
    in_tee: bool,
) -> tuple[list[str], dict[str, int]]:
    """Everything that can refuse, before anything on the host changes: the pod's GPUs to touch."""
    if in_tee:
        raise GpuResetRefused(GpuResetErrorCode.ClockLockNotSupported, "clock locking is not supported on TEE nodes")
    inspected = await _run(runner, f"{DOCKER} inspect --format '{{{{json .}}}}' {shlex.quote(container_name)}")
    if inspected.exit_code != 0:
        raise GpuResetRefused(GpuResetErrorCode.ContainerNotFound, f"container {container_name} not found")
    try:
        host_config = json.loads(inspected.stdout).get("HostConfig") or {}
    except (ValueError, AttributeError) as exc:
        raise GpuResetRefused(GpuResetErrorCode.ContainerNotFound, f"unreadable docker inspect: {exc}") from exc
    host = await _run(runner, HOST_GPUS_COMMAND)
    host_gpus = parse_host_gpus(host.stdout or "")
    if host.exit_code != 0 or not host_gpus:
        raise GpuResetRefused(GpuResetErrorCode.ClockLockNotSupported, "nvidia-smi listed no GPU on the host")
    host_by_uuid = {gpu.uuid: gpu for gpu in host_gpus}
    # the pod's numbering, shared with gpu_reset; only the backend's GPUs of the pod are ever touched
    gpus = pod_gpus(host_config.get("DeviceRequests"), host_gpus, pod_gpu_uuids, gpu_indices)
    targets, index_by_uuid = gpus.requested, gpus.index_by_uuid
    if not set(targets) <= set(pod_gpu_uuids):
        raise GpuResetRefused(GpuResetErrorCode.GpuNotInContainer, "a GPU to lock is not one of the pod's GPUs")
    if any(host_by_uuid[uuid].mig_enabled for uuid in targets):
        raise GpuResetRefused(GpuResetErrorCode.MigEnabled, "MIG is enabled on a GPU")
    return targets, index_by_uuid


async def _forget(redis: RedisService, uuids: list[str]) -> None:
    """Drop the lock records of these GPUs and their entries in the pod indices."""
    for uuid in uuids:
        raw = await redis.get(_lock_key(uuid))
        await redis.delete(_lock_key(uuid))
        record = json.loads(raw) if raw else {}
        if not record.get("pod_id") or not record.get("executor_id") or not record.get("miner_hotkey"):
            continue
        key = _pod_key(record["pod_id"], record["miner_hotkey"], record["executor_id"])
        raw_index = await redis.get(key)
        left = [u for u in (json.loads(raw_index) if raw_index else []) if u != uuid]
        if left:
            await redis.set(key, json.dumps(left))
        else:
            await redis.delete(key)


async def _reset_gpu_clocks(
    runner: SSHCommandRunner,
    redis: RedisService,
    uuids: list[str],
    pod_gpu_uuids: set[str] | None = None,
    *,
    purpose: str,
    pod_id: str | None = None,
    forget: bool = True,
) -> dict[str, str | None]:
    """`-rgc`, `-rac`, `-rmc` on every GPU, the GPUs in parallel, in one SSH call (a create runs this on every GPU);
    the record goes only once its -rgc succeeded (forget=False: the caller drops it). uuid -> failure (None = reset)."""
    if not uuids:
        _log_reset_outcome(purpose, pod_id, [], [], {}, None)
        return {}
    results = f"/tmp/lium-gpu-clock-reset-{secrets.token_hex(8)}"
    result = await _run(runner, _reset_command(uuids, pod_gpu_uuids, results), timeout=_RESET_CALL_TIMEOUT_SECONDS)
    stdout = result.stdout or ""
    if result.exit_code != 0:
        # the answer was lost (budget overrun, channel closed): what the GPUs wrote so far; a reset still running
        # after this read writes a line nobody reads (a few bytes left in /tmp), its GPU counted as no answer
        stdout = (await _run(runner, f"cat {results} 2>/dev/null; rm -f {results}")).stdout or ""
    lines = [line.split() for line in stdout.splitlines()]
    # the -rgc lines (a -rac/-rmc line has 3 fields); a killed -rgc gave no verdict, as a lost answer
    exits = {p[0]: p[1] for p in lines if len(p) == 2 and p[1] != _KILLED_EXIT}
    failures: dict[str, str | None] = {
        uuid: None if exits.get(uuid) == "0" else _rgc_failure(exits[uuid]) if uuid in exits else _NO_ANSWER
        for uuid in uuids
    }
    _log_reset_outcome(purpose, pod_id, uuids, lines, failures, result)
    if forget:
        await _forget(redis, [uuid for uuid, failure in failures.items() if failure is None])
    return failures


async def lock_pod_gpu_clocks(
    runner: SSHCommandRunner,
    redis: RedisService,
    *,
    container_name: str,
    pod_id: str,
    miner_hotkey: str,
    executor_id: str,
    pod_gpu_uuids: list[str],
    gpu_indices: list[int] | None,
    min_mhz: int | None,
    max_mhz: int | None,
    reset: bool,
    in_tee: bool = False,
) -> GpuClocksReport:
    try:
        targets, index_by_uuid = await _pod_gpus_to_lock(
            runner, container_name=container_name, pod_gpu_uuids=pod_gpu_uuids, gpu_indices=gpu_indices, in_tee=in_tee
        )
    except GpuResetRefused as refused:
        return GpuClocksReport(ok=False, error_code=refused.code, detail=refused.detail)
    pod_gpu_uuids = set(targets)

    if reset:
        async with _holding(targets):
            failures = await _reset_gpu_clocks(runner, redis, targets, pod_gpu_uuids, purpose="lock_reset", pod_id=pod_id)
        gpus = [
            GpuClockOutcome(index=index_by_uuid[u], uuid=u, result="failed" if f else "reset", detail=f)
            for u, f in failures.items()
        ]
        ok = all(f is None for f in failures.values())
        return GpuClocksReport(ok=ok, error_code=None if ok else GpuResetErrorCode.ClockLockNotSupported, gpus=gpus)

    # the pod's index first, so its delete resets every GPU this request may lock (a GPU never locked costs one -rgc)
    raw_index = await redis.get(_pod_key(pod_id, miner_hotkey, executor_id))
    pod_uuids = list(dict.fromkeys([*(json.loads(raw_index) if raw_index else []), *targets]))
    await redis.set(_pod_key(pod_id, miner_hotkey, executor_id), json.dumps(pod_uuids))

    gpus: list[GpuClockOutcome] = []
    for uuid in targets:
        try:
            gpus.append(await _lock_one_gpu(
                runner, redis, uuid, index_by_uuid[uuid], pod_gpu_uuids, min_mhz, max_mhz, pod_id=pod_id,
                miner_hotkey=miner_hotkey, executor_id=executor_id,
            ))
        except Exception:
            # Redis failed mid-request (review): answer the GPUs already locked, stop before the rest. A GPU whose
            # record was written keeps it, so the pod's delete still resets it
            gpus += [
                GpuClockOutcome(
                    index=index_by_uuid[u], uuid=u, result="failed",
                    detail="the validator could not record the lock" if u == uuid else "not attempted",
                )
                for u in targets[len(gpus):]
            ]
            return GpuClocksReport(
                ok=False, error_code=GpuResetErrorCode.ExecutorUnreachable,
                detail="the validator could not record the clock lock", gpus=gpus,
            )
    # a refused -lgc whose reset failed keeps its record: the pod's delete and the next create reset it again
    ok = all(gpu.result == "locked" for gpu in gpus)
    return GpuClocksReport(ok=ok, error_code=None if ok else GpuResetErrorCode.ClockLockNotSupported, gpus=gpus)


async def _lock_one_gpu(
    runner: SSHCommandRunner,
    redis: RedisService,
    uuid: str,
    index: int,
    pod_gpu_uuids: set[str],
    min_mhz: int | None,
    max_mhz: int | None,
    *,
    pod_id: str,
    miner_hotkey: str,
    executor_id: str,
) -> GpuClockOutcome:
    """-pm 1, clamp to the GPU's supported clocks, record the lock, -lgc, read the clock back."""
    outcome = GpuClockOutcome(index=index, uuid=uuid, result="failed")
    # best-effort, as in gpu_power_limit: without persistence the driver drops the lock on an idle GPU.
    # Intended: -rgc leaves persistence on (it only keeps the driver loaded; the provider's settings are unchanged)
    await _run(runner, f"nvidia-smi -i {shlex.quote(uuid)} -pm 1")
    supported = await _run(
        runner, f"nvidia-smi -i {shlex.quote(uuid)} --query-supported-clocks=graphics --format=csv,noheader,nounits"
    )
    supported_mhz = _parse_mhz(supported.stdout or "") if supported.exit_code == 0 else []
    if not supported_mhz:
        outcome.detail = "the GPU reports no supported graphics clocks"
        return outcome
    low, high = clamp_range(min_mhz, max_mhz, supported_mhz)
    async with _holding([uuid]):
        return await _record_and_lock(
            runner, redis, outcome, uuid, pod_gpu_uuids, low, high, pod_id, miner_hotkey, executor_id
        )


async def _record_and_lock(
    runner: SSHCommandRunner,
    redis: RedisService,
    outcome: GpuClockOutcome,
    uuid: str,
    pod_gpu_uuids: set[str],
    low: int,
    high: int,
    pod_id: str,
    miner_hotkey: str,
    executor_id: str,
) -> GpuClockOutcome:
    # record first, as the power cap does: never change a GPU without a stored way back. Only right before the
    # -lgc, so a GPU this request never changes gets no record a create would then refuse on (review F-46)
    record = {"gpu_uuid": uuid, "pod_id": pod_id, "miner_hotkey": miner_hotkey, "executor_id": executor_id,
              "locked_at": time.time()}
    previous = await redis.get(_lock_key(uuid))
    # no expiry (gateway review): a record goes only once an -rgc succeeded, so a GPU whose reset keeps failing, maybe
    # still locked, stays refused rather than handed on
    await redis.set(_lock_key(uuid), json.dumps(record))
    locked = await _run(runner, build_clock_command("lock", uuid, pod_gpu_uuids=pod_gpu_uuids, min_mhz=low, max_mhz=high))
    if locked.exit_code > 0:
        # a completed refusal ("not supported" 3, "insufficient permissions" 4, ...) changed no clocks: the record goes
        # back to what it was, so a GPU without clock control (that refuses -rgc too) never carries a record a create
        # would refuse on (review)
        if previous:
            await redis.set(_lock_key(uuid), previous)
        else:
            await redis.delete(_lock_key(uuid))
        outcome.detail = _tail(locked.stdout or locked.stderr) or f"nvidia-smi -lgc failed (exit {locked.exit_code})"
        return outcome
    if locked.exit_code != 0:
        # -1, the runner's timeout or lost answer: an applied -lgc may hide behind it, so reset the GPU, and its record
        # goes only once that reset succeeded; else it stays, so a later create refuses rather than hand it on locked
        reset_failure = (await _reset_gpu_clocks(runner, redis, [uuid], pod_gpu_uuids, purpose="lock_unanswered"))[uuid]
        outcome.detail = "nvidia-smi -lgc did not answer"
        outcome.detail += "; clocks reset to default" if reset_failure is None else f"; {reset_failure}"
        return outcome
    readback = await _run(
        runner, f"nvidia-smi -i {shlex.quote(uuid)} --query-gpu=clocks.gr --format=csv,noheader,nounits"
    )
    current = _parse_mhz(readback.stdout or "") if readback.exit_code == 0 else []
    outcome.result, outcome.applied_min_mhz, outcome.applied_max_mhz = "locked", low, high
    outcome.detail = f"graphics clock now {current[0]} MHz" if current else "graphics clock unreadable"
    return outcome


async def reset_pod_clock_locks(
    runner: SSHCommandRunner, redis: RedisService, pod_id: str, miner_hotkey: str, executor_id: str
) -> int:
    """Teardown on one executor: `-rgc` every GPU the pod locked there; the reset count. The caller makes it
    best-effort."""
    pod_key = _pod_key(pod_id, miner_hotkey, executor_id)
    raw_index, raw_legacy = await redis.get(pod_key), await redis.get(_legacy_pod_key(pod_id))
    if not raw_index and not raw_legacy:
        return 0  # the pod locked nothing: no SSH
    # only GPUs whose record still names this pod on this executor (review): a stale index left by a failed cleanup,
    # or the pod's lock on the node it moved to, must not be reset or dropped; an unrecorded GPU is reset by the next
    # create's safety net
    indexed = list(dict.fromkeys(str(u) for raw in (raw_index, raw_legacy) if raw for u in json.loads(raw)))
    # hold only the GPUs this pod owns (review): the legacy index is per pod only, so holding all of it made the old
    # node's delete wait on the new node's GPU lock
    records = {uuid: await _record(redis, uuid) for uuid in indexed}
    candidates = [uuid for uuid in indexed if _owned_by(records[uuid], pod_id, miner_hotkey, executor_id)]
    if any("miner_hotkey" not in records[uuid] for uuid in candidates):
        candidates = await _on_this_host(runner, candidates, records)
    async with _holding(candidates):  # re-read under the lock: no other pod's lock between the owner read and the reset
        owned = [u for u in candidates if _owned_by(await _record(redis, u), pod_id, miner_hotkey, executor_id)]
        failures = await _reset_gpu_clocks(runner, redis, owned, purpose="teardown", pod_id=pod_id)
    await redis.delete(pod_key)  # a GPU whose reset failed keeps its own record for the next create
    if raw_legacy:
        left = [u for u in json.loads(raw_legacy) if u not in owned]
        await (redis.set(_legacy_pod_key(pod_id), json.dumps(left)) if left else redis.delete(_legacy_pod_key(pod_id)))
    return sum(1 for failure in failures.values() if failure is None)


async def _on_this_host(runner: SSHCommandRunner, candidates: list[str], records: dict[str, dict]) -> list[str]:
    """A legacy record has no miner hotkey, so it cannot tell two miners' same executor id apart (review): keep it
    only for a GPU on the addressed host. An unread host keeps every legacy record for the next create's reset."""
    try:
        host = await _run(runner, HOST_GPUS_COMMAND)
        on_host = {gpu.uuid for gpu in parse_host_gpus(host.stdout or "")} if host.exit_code == 0 else set()
    except Exception:
        on_host = set()
    return [uuid for uuid in candidates if "miner_hotkey" in records[uuid] or uuid in on_host]


async def _record(redis: RedisService, uuid: str) -> dict:
    raw = await redis.get(_lock_key(uuid))
    return json.loads(raw) if raw else {}


async def reset_leftover_clock_locks(
    runner: SSHCommandRunner,
    redis: RedisService,
    gpu_uuids: list[str] | None,
    keep_pod_id: str | None = None,
    expected: dict[str, str] | None = None,
    purpose: str = "check",
) -> list[str]:
    """Safety net before a create: `-rgc` every GPU of the new pod (None/empty = every host GPU), with or
    without a lock record, so neither a lost record nor a lock set outside Lium hands a renter locked clocks (as
    raise_low_power_limits_to_default does for power). Answers the GPUs whose reset failed or gave no answer (records
    stay): the create must not start on them, so a GPU whose clocks cannot be confirmed reset fails closed. A GPU with no record that refuses -rgc as having no clock control here (exit 3
    or 4: nobody locked it from here) is not counted; any other failure is, record or not. While Redis cannot say
    which GPUs have a record, every failed reset counts. A GPU whose record names `keep_pod_id` (the pod's own edit or reboot) keeps its lock and
    record (review). `expected` (uuid -> the record read before the GPU lock) resets only a GPU whose record is
    still that one: a new renter's lock taken meanwhile is left alone (review F-83)."""
    if not gpu_uuids:
        host = await _run(runner, HOST_GPUS_COMMAND)
        gpu_uuids = [gpu.uuid for gpu in parse_host_gpus(host.stdout or "")]
    async with _holding(gpu_uuids):
        if expected is not None:
            try:
                gpu_uuids = [uuid for uuid in gpu_uuids if await redis.get(_lock_key(uuid)) == expected.get(uuid)]
            except Exception:
                return []  # unknown: touch nothing, the caller judges again next cycle
            if not gpu_uuids:
                return []
        return await _reset_leftovers(runner, redis, gpu_uuids, keep_pod_id, purpose)


async def _reset_leftovers(
    runner: SSHCommandRunner, redis: RedisService, gpu_uuids: list[str], keep_pod_id: str | None, purpose: str
) -> list[str]:
    try:
        raws = {uuid: await redis.get(_lock_key(uuid)) for uuid in gpu_uuids}
        recorded = {uuid for uuid, raw in raws.items() if isinstance(raw, (str, bytes))}
        own = {uuid for uuid in recorded if keep_pod_id and json.loads(raws[uuid]).get("pod_id") == keep_pod_id}
    except Exception:
        recorded, own = set(gpu_uuids), set()  # unknown (review): every failed reset counts, never a silent pass
    gpu_uuids = [uuid for uuid in gpu_uuids if uuid not in own]
    try:
        failures = await _reset_gpu_clocks(runner, redis, gpu_uuids, purpose=purpose, pod_id=keep_pod_id, forget=False)
    except Exception:
        failures = dict.fromkeys(gpu_uuids, _NO_ANSWER)  # the SSH call failed: no GPU answered
    try:
        await _forget(redis, [uuid for uuid, failure in failures.items() if failure is None])
    except Exception:
        pass  # the GPUs are reset; a record left over costs the next create one more -rgc (review)
    # fail closed: a GPU with no verdict (its -rgc killed on the host, or its line lost) may still hold lowered clocks,
    # record or not; only a completed exit 3/4 on an unrecorded GPU (no clock control from here) is not counted
    refused = [
        uuid for uuid in gpu_uuids
        if failures[uuid] is not None and (uuid in recorded or failures[uuid] not in _NO_CLOCK_CONTROL)
    ]
    for uuid in (u for u in gpu_uuids if failures[u] is not None):
        logger.warning(
            "gpu_clock_action=rgc_failed gpu=%s recorded=%s failure=%s",
            uuid, str(uuid in recorded).lower(), failures[uuid],
        )
    return refused


async def clock_locked_gpu_uuids(redis: RedisService, gpu_uuids: list[str]) -> set[str]:
    """The GPUs among these with a lock record (a Redis error answers none: the check judges as before)."""
    try:
        # only a real stored record counts (a mock or an odd client answer never skips the check)
        return {uuid for uuid in gpu_uuids if isinstance(await redis.get(_lock_key(uuid)), (str, bytes))}
    except Exception:
        return set()
