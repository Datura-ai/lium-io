"""DAH-4663: a renter locks (or resets) the graphics clocks of their own pod's GPUs.

Run over the validator's SSH session into the executor (the host's `docker` and `nvidia-smi`, as in
services/gpu_reset.py), the PEARL power-cap pattern of services/gpu_power_limit.py:

    prechecks (change nothing) -> record in Redis -> nvidia-smi -i <uuid> -pm 1 -> clamp to the GPU's
        supported clocks -> nvidia-smi -i <uuid> -lgc min,max -> read the clocks back

Safety invariants, each pinned by a test:
- the GPUs touched are a subset of BOTH the container's DeviceRequests GPUs and the backend's GPUs;
- `-lgc` / `-rgc` never run without `-i`, and the power limit (`-pl`) is never touched;
- every lock is recorded (`gpu_clock_lock:<uuid>`, `gpu_clock_lock_pod:<pod_id>`) before it is set,
  so the pod's delete and the safety net before any create can `-rgc` it.
"""

from __future__ import annotations

import json
import shlex
import time
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

CLOCK_LOCK_KEY_PREFIX = "gpu_clock_lock:"
CLOCK_LOCK_POD_KEY_PREFIX = "gpu_clock_lock_pod:"


@dataclass
class GpuClocksReport:
    ok: bool
    error_code: GpuResetErrorCode | None = None
    gpus: list[GpuClockOutcome] = field(default_factory=list)
    detail: str | None = None


def _lock_key(uuid: str) -> str:
    return f"{CLOCK_LOCK_KEY_PREFIX}{uuid}"


def _pod_key(pod_id: str) -> str:
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
    if "sysbox" in str(host_config.get("Runtime") or ""):
        raise GpuResetRefused(GpuResetErrorCode.ClockLockNotSupported, "clock locking is not supported on sysbox pods")

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
        pod_id = json.loads(raw).get("pod_id") if raw else None
        if not pod_id:
            continue
        raw_index = await redis.get(_pod_key(pod_id))
        left = [u for u in (json.loads(raw_index) if raw_index else []) if u != uuid]
        if left:
            await redis.set(_pod_key(pod_id), json.dumps(left))
        else:
            await redis.delete(_pod_key(pod_id))


async def _reset_gpus(
    runner: SSHCommandRunner,
    redis: RedisService,
    uuids: list[str],
    pod_gpu_uuids: set[str] | None = None,
    *,
    forget: bool = True,
) -> dict[str, str | None]:
    """`-rgc` each GPU in one SSH call (a create runs this on every GPU); the record goes only once its reset
    succeeded (forget=False: the caller drops it). uuid -> failure (None = reset)."""
    if not uuids:
        return {}
    # each reset echoes "<uuid> <exit>", so one GPU's failure never hides another's result
    cmd = "; ".join(
        f"{build_clock_command('reset', uuid, pod_gpu_uuids=pod_gpu_uuids)} >/dev/null 2>&1; echo {shlex.quote(uuid)} $?"
        for uuid in uuids
    )
    result = await _run(runner, cmd)
    exits = dict(line.split()[:2] for line in (result.stdout or "").splitlines() if len(line.split()) >= 2)
    failures: dict[str, str | None] = {
        uuid: None if exits.get(uuid) == "0" else f"nvidia-smi -rgc failed (exit {exits.get(uuid, 'unknown')})"
        for uuid in uuids
    }
    if forget:
        await _forget(redis, [uuid for uuid, failure in failures.items() if failure is None])
    return failures


async def lock_pod_gpu_clocks(
    runner: SSHCommandRunner,
    redis: RedisService,
    *,
    container_name: str,
    pod_id: str,
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
        failures = await _reset_gpus(runner, redis, targets, pod_gpu_uuids)
        gpus = [
            GpuClockOutcome(index=index_by_uuid[u], uuid=u, result="failed" if f else "reset", detail=f)
            for u, f in failures.items()
        ]
        ok = all(f is None for f in failures.values())
        return GpuClocksReport(ok=ok, error_code=None if ok else GpuResetErrorCode.ClockLockNotSupported, gpus=gpus)

    # the pod's index first, so its delete resets every GPU this request may lock (a GPU never locked costs one -rgc)
    raw_index = await redis.get(_pod_key(pod_id))
    pod_uuids = list(dict.fromkeys([*(json.loads(raw_index) if raw_index else []), *targets]))
    await redis.set(_pod_key(pod_id), json.dumps(pod_uuids))

    gpus = [
        await _lock_one_gpu(
            runner, redis, uuid, index_by_uuid[uuid], pod_gpu_uuids, min_mhz, max_mhz, pod_id=pod_id,
            executor_id=executor_id,
        )
        for uuid in targets
    ]
    # a refused -lgc leaves the record it found; the pod's delete still runs -rgc on every GPU in its index
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
    # record first, as the power cap does: never change a GPU without a stored way back. Only right before the
    # -lgc, so a GPU this request never changes gets no record a create would then refuse on (review F-46)
    record = {"gpu_uuid": uuid, "pod_id": pod_id, "executor_id": executor_id, "locked_at": time.time()}
    previous = await redis.get(_lock_key(uuid))
    await redis.set(_lock_key(uuid), json.dumps(record))
    locked = await _run(runner, build_clock_command("lock", uuid, pod_gpu_uuids=pod_gpu_uuids, min_mhz=low, max_mhz=high))
    if locked.exit_code != 0:
        # a refused -lgc changed nothing: back to the record before it (review: else a GPU that also refuses -rgc
        # would carry a record for clocks never locked, and every later create on it would be refused)
        if previous:
            await redis.set(_lock_key(uuid), previous)
        else:
            await redis.delete(_lock_key(uuid))
        outcome.detail = _tail(locked.stdout or locked.stderr) or "nvidia-smi -lgc failed"
        return outcome
    readback = await _run(
        runner, f"nvidia-smi -i {shlex.quote(uuid)} --query-gpu=clocks.gr --format=csv,noheader,nounits"
    )
    current = _parse_mhz(readback.stdout or "") if readback.exit_code == 0 else []
    outcome.result, outcome.applied_min_mhz, outcome.applied_max_mhz = "locked", low, high
    outcome.detail = f"graphics clock now {current[0]} MHz" if current else "graphics clock unreadable"
    return outcome


async def reset_pod_clock_locks(runner: SSHCommandRunner, redis: RedisService, pod_id: str) -> int:
    """Teardown: `-rgc` every GPU the pod locked; the reset count. The caller makes it best-effort."""
    raw_index = await redis.get(_pod_key(pod_id))
    if not raw_index:
        return 0  # the pod locked nothing: one Redis read, no SSH
    failures = await _reset_gpus(runner, redis, [str(u) for u in json.loads(raw_index)])
    await redis.delete(_pod_key(pod_id))  # a GPU whose reset failed keeps its own record for the next create
    return sum(1 for failure in failures.values() if failure is None)


async def reset_leftover_clock_locks(
    runner: SSHCommandRunner, redis: RedisService, gpu_uuids: list[str] | None
) -> list[str]:
    """Safety net before a create: `-rgc` every GPU of the new pod (None/empty = every host GPU), with or
    without a lock record, so a lost record never hands a renter another's locked clocks (as
    raise_low_power_limits_to_default does for power). Answers the GPUs with a lock record whose reset failed
    (their records stay): the create must not start on them. A failed reset of a GPU with no record (one that
    does not support -rgc, never locked here) is not counted; while Redis cannot say which GPUs have a record,
    every failed reset counts."""
    if not gpu_uuids:
        host = await _run(runner, HOST_GPUS_COMMAND)
        gpu_uuids = [gpu.uuid for gpu in parse_host_gpus(host.stdout or "")]
    try:
        recorded = {uuid for uuid in gpu_uuids if isinstance(await redis.get(_lock_key(uuid)), (str, bytes))}
    except Exception:
        recorded = set(gpu_uuids)  # unknown (review): every failed reset counts, never a silent pass
    try:
        failures = await _reset_gpus(runner, redis, gpu_uuids, forget=False)
    except Exception:
        if not recorded:
            raise
        failures = dict.fromkeys(gpu_uuids, "the reset did not run")
    try:
        await _forget(redis, [uuid for uuid, failure in failures.items() if failure is None])
    except Exception:
        pass  # the GPUs are reset; a record left over costs the next create one more -rgc (review)
    return [uuid for uuid in gpu_uuids if uuid in recorded and failures[uuid] is not None]


async def clock_locked_gpu_uuids(redis: RedisService, gpu_uuids: list[str]) -> set[str]:
    """The GPUs among these with a lock record (a Redis error answers none: the check judges as before)."""
    try:
        # only a real stored record counts (a mock or an odd client answer never skips the check)
        return {uuid for uuid in gpu_uuids if isinstance(await redis.get(_lock_key(uuid)), (str, bytes))}
    except Exception:
        return set()
