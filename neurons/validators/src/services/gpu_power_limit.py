"""DAH-2356: cap GPU power (watts) for the Lium PEARL default-job (FILLER) container.

The backend (lium-io-backend) computes a per-GPU target and sends it as
``ContainerCreateRequest.gpu_power_limits`` (GpuPowerLimit: gpu_uuid + watts). We apply it host-side
with ``nvidia-smi -pl``, clamped to each GPU's hardware ``[min, max]``, before the filler starts.

Redis state (all keys written only by this validator):
- ``gpu_power_restore:<gpu_uuid>`` — one frozen ``GpuPowerRestoreRecord`` per capped GPU holding the
  pre-cap limit. Never overwritten (a leftover record from a failed restore holds the TRUE original
  limit; overwriting on a re-cap would ratchet the "original" down forever). Deleted only after the
  recorded limit has been successfully restored with ``nvidia-smi -pl``.
- ``gpu_power_restore_pod:<pod_id>`` — the gpu_uuids a filler pod capped, so its delete can restore
  exactly its own GPUs without an SSH enumeration and without sweeping a replacement filler's fresh
  records.

A record that outlives its filler (delete failed, SSH broke mid-teardown) is repaired by two safety
nets, so a reduced limit can never stick:
- ``GpuPowerLimitCheck`` (validator): a below-floor GPU whose record this validator wrote is our own
  stale cap — no penalty, and records past ``STALE_CAP_GRACE_SECONDS`` are restored.
- ``create_container`` (validator connector): before starting any container WITHOUT a cap of its
  own, leftover records for its GPUs are restored, so customers never inherit a reduced limit.

**Every set is verified** (live-repro on H100, 2026-07-13): with persistence mode off the driver
unloads once the GPU goes idle and silently reverts ``-pl`` — nvidia-smi still exits 0 ("All done").
So each set first enables persistence mode (``-pm 1``, best-effort: keeps the driver loaded so the
limit survives) and then READS BACK ``power.limit``; a mismatch counts as a failed set. The readback
is the hard gate — a cap that cannot be observed on the GPU does not exist.
**Apply is fail-closed for the filler**: ``apply_filler_gpu_power_limits`` returns ``False`` if the
cap could not be fully applied, after undoing whatever it already capped or stored; the caller then
REFUSES to start the PEARL filler (running the miner uncapped defeats the whole point).
**Restore stays best-effort**: teardown must never be blocked by a power-limit hiccup; a record
whose restore failed is kept and retried by the safety nets.
Every power-limit change is logged via ``_m`` (so the fields reach the JSON/Loki output) with
executor_id, gpu_uuid, watts before/after, and status.
"""

from __future__ import annotations

import json
import logging
import shlex
import time
from dataclasses import dataclass
from typing import Literal

import asyncssh
from payload_models.payloads import GpuPowerLimit
from pydantic import BaseModel, ValidationError
from services.redis_service import RedisService

from core.utils import _m, get_extra_info

logger = logging.getLogger(__name__)

_POWER_STATE_CMD = (
    "nvidia-smi --query-gpu=uuid,power.limit,power.default_limit,power.min_limit,power.max_limit "
    "--format=csv,noheader,nounits"
)
# Bound each nvidia-smi call so a hung driver can't stall filler deploy/undeploy (PoC #1120).
_NVIDIA_SMI_TIMEOUT_SECONDS = 30
_RESTORE_KEY_PREFIX = "gpu_power_restore:"
_POD_INDEX_KEY_PREFIX = "gpu_power_restore_pod:"
# A record younger than this may belong to a filler the check's backend snapshot doesn't report yet
# (backend marks owner="lium" from STARTING, but the snapshot is taken at cycle start) — the check
# must not uncap it. Comfortably above the backend's 15-minute STARTING/STOPPING transitional window.
STALE_CAP_GRACE_SECONDS = 30 * 60
# Floor enforced by GpuPowerLimitCheck. Doubles as the raise trigger at rental start: a limit
# below this can't be a legitimate miner setting (the check would zero-score it anyway).
MIN_POWER_LIMIT_RATIO = 0.9


class GpuPowerRestoreRecord(BaseModel):
    """One GPU's pre-cap power limit, frozen in Redis until successfully restored."""

    gpu_uuid: str
    watts: int
    pod_id: str
    executor_id: str
    capped_at: float  # unix seconds; lets the check age-gate its restore against STALE_CAP_GRACE_SECONDS


class GpuPowerRestoreReadResult(BaseModel):
    """Records found in Redis plus whether any read errored.

    ``read_failed`` lets the check tell "no record exists" (genuine miner violation → penalize)
    from "Redis didn't answer" (our own outage → skip the penalty this cycle). A transient Redis
    error must never zero-score an innocent miner whose cap we set ourselves."""

    records: list[GpuPowerRestoreRecord]
    read_failed: bool


@dataclass(frozen=True)
class GpuPowerState:
    # Only the current limit is required (needed to record the pre-cap value). The others are
    # optional: some GPUs report "[N/A]"/"[Not Supported]" and must NOT drop the whole GPU.
    current_watts: int
    min_watts: int | None
    max_watts: int | None
    default_watts: int | None = None


@dataclass(frozen=True)
class GpuPowerReadback:
    """What one GPU reports right after a set: the limit that stuck, and whether persistence mode
    is on. ``persistence_enabled=None`` means nvidia-smi did not report the field."""

    watts: int | None
    persistence_enabled: bool | None


@dataclass(frozen=True)
class PowerLimitSetOutcome:
    """Outcome of one verified set: why it failed (None = success) and the persistence verdict the
    readback saw. A successful set with persistence off can still revert on its own later."""

    failure: str | None
    persistence_enabled: bool | None


# nvidia-smi's own labels; anything else means the GPU did not report the field.
_PERSISTENCE_BY_LABEL: dict[str, bool] = {"enabled": True, "disabled": False}


def _restore_key(gpu_uuid: str) -> str:
    return f"{_RESTORE_KEY_PREFIX}{gpu_uuid}"


def _pod_index_key(pod_id: str) -> str:
    return f"{_POD_INDEX_KEY_PREFIX}{pod_id}"


def _watts_or_none(raw: str) -> int | None:
    # nvidia-smi prints "[N/A]" / "[Not Supported]" for fields a GPU doesn't expose.
    try:
        return round(float(raw))
    except ValueError:
        return None


def _parse_power_state_csv(stdout: str) -> dict[str, GpuPowerState]:
    state_by_uuid: dict[str, GpuPowerState] = {}
    for line in stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            continue
        uuid, current_raw, default_raw, min_raw, max_raw = fields
        current_watts = _watts_or_none(current_raw)
        if not uuid or current_watts is None:
            continue  # without the current limit we can't record a pre-cap value to restore later
        state_by_uuid[uuid] = GpuPowerState(
            current_watts=current_watts,
            min_watts=_watts_or_none(min_raw),
            max_watts=_watts_or_none(max_raw),
            default_watts=_watts_or_none(default_raw),
        )
    return state_by_uuid


def _parse_power_readback_csv(stdout: str) -> GpuPowerReadback:
    # Unreported persistence ("[N/A]", missing column) is not the same as persistence being off.
    watts_raw, _, persistence_raw = stdout.partition(",")
    return GpuPowerReadback(
        watts=_watts_or_none(watts_raw.strip()),
        persistence_enabled=_PERSISTENCE_BY_LABEL.get(persistence_raw.strip().lower()),
    )


def _clamp_watts(target_watts: int, state: GpuPowerState) -> int:
    # Clamp only by the bounds the GPU actually reports (a "[N/A]" bound is skipped, not treated as 0).
    watts = target_watts
    if state.max_watts is not None:
        watts = min(watts, state.max_watts)
    if state.min_watts is not None:
        watts = max(watts, state.min_watts)
    return watts


def _log(level: int, message: str, fields: dict[str, object], log_extra: dict[str, object] | None) -> None:
    logger.log(level, _m(message, extra=get_extra_info({**(log_extra or {}), **fields})))


async def _query_power_state(ssh: asyncssh.SSHClientConnection) -> dict[str, GpuPowerState]:
    result = await ssh.run(_POWER_STATE_CMD, timeout=_NVIDIA_SMI_TIMEOUT_SECONDS)
    if result.exit_status != 0:
        raise RuntimeError(
            f"nvidia-smi power-state query failed: exit_status={result.exit_status}, "
            f"stderr={result.stderr!r}"
        )
    return _parse_power_state_csv(result.stdout)


async def _enable_persistence_mode(
    ssh: asyncssh.SSHClientConnection,
    uuid: str,
    log_extra: dict[str, object] | None,
) -> None:
    """Enable persistence mode so a set limit survives the driver unloading on an idle GPU.

    Best-effort: the readback verify after ``-pl`` is the hard gate, not this. Never raises.
    """
    failure: str | None = None
    try:
        result = await ssh.run(
            f"nvidia-smi -i {shlex.quote(uuid)} -pm 1", timeout=_NVIDIA_SMI_TIMEOUT_SECONDS
        )
        if result.exit_status != 0:
            failure = f"exit={result.exit_status}, stderr={result.stderr!r}"
    except Exception as exc:
        failure = str(exc)
    if failure is not None:
        _log(
            logging.WARNING,
            f"gpu power limit: enabling persistence mode for {uuid} failed ({failure}); "
            f"the set may not stick — readback verify will decide",
            {"gpu_uuid": uuid},
            log_extra,
        )


async def _read_back_power_state(ssh: asyncssh.SSHClientConnection, uuid: str) -> GpuPowerReadback:
    """Read one GPU's power.limit and persistence mode. Both come from the same nvidia-smi query, so
    verifying persistence costs no extra round trip. Unreadable -> all-None (never raises)."""
    readback_command = (
        f"nvidia-smi -i {shlex.quote(uuid)} --query-gpu=power.limit,persistence_mode "
        f"--format=csv,noheader,nounits"
    )
    try:
        result = await ssh.run(readback_command, timeout=_NVIDIA_SMI_TIMEOUT_SECONDS)
    except Exception:
        return GpuPowerReadback(watts=None, persistence_enabled=None)
    if result.exit_status != 0:
        return GpuPowerReadback(watts=None, persistence_enabled=None)
    return _parse_power_readback_csv(str(result.stdout))


async def _set_power_limit(
    ssh: asyncssh.SSHClientConnection,
    uuid: str,
    watts: int,
) -> PowerLimitSetOutcome:
    """Set one GPU's power limit and VERIFY it stuck (never raises). nvidia-smi can report success
    while the limit silently reverts (persistence mode off, driver unloads) — only the readback
    proves the cap exists."""
    try:
        result = await ssh.run(
            f"nvidia-smi -i {shlex.quote(uuid)} -pl {watts}", timeout=_NVIDIA_SMI_TIMEOUT_SECONDS
        )
    except Exception as exc:
        return PowerLimitSetOutcome(failure=f"nvidia-smi -pl errored: {exc}", persistence_enabled=None)
    if result.exit_status != 0:
        return PowerLimitSetOutcome(
            failure=f"nvidia-smi -pl failed: exit={result.exit_status}, stderr={result.stderr!r}",
            persistence_enabled=None,
        )
    readback = await _read_back_power_state(ssh, uuid)
    if readback.watts is None:
        return PowerLimitSetOutcome(
            failure="nvidia-smi -pl reported success but the limit could not be read back for verification",
            persistence_enabled=readback.persistence_enabled,
        )
    if readback.watts != watts:
        return PowerLimitSetOutcome(
            failure=(
                f"nvidia-smi -pl reported success but the limit did not stick: "
                f"readback {readback.watts}W != target {watts}W (persistence mode unavailable?)"
            ),
            persistence_enabled=readback.persistence_enabled,
        )
    return PowerLimitSetOutcome(failure=None, persistence_enabled=readback.persistence_enabled)


async def _set_and_log_power_limit(
    ssh: asyncssh.SSHClientConnection,
    action: Literal["cap", "restore", "raise"],
    executor_id: str,
    gpu_uuid: str,
    watts_before: int | None,
    watts_after: int,
    log_extra: dict[str, object] | None,
) -> bool:
    # Reviewer contract (PR #1115): every PL change is logged with executor, GPU, before/after, status.
    await _enable_persistence_mode(ssh, gpu_uuid, log_extra)
    set_outcome = await _set_power_limit(ssh, gpu_uuid, watts_after)
    failure = set_outcome.failure
    status = "ok" if failure is None else "failed"
    message = f"gpu power limit {action} {status}: executor={executor_id} gpu={gpu_uuid} watts {watts_before} -> {watts_after}"
    if failure is not None:
        message = f"{message} ({failure})"
    # DAH-2702: with persistence off the driver unloads on an idle GPU and the stock limit returns,
    # which is how a cap reverts untouched. Only a cap is at risk — restore/raise set the limit back
    # UP, where an unload lands anyway. Never fail-closed: refusing the filler would drop PEARL from
    # every host that cannot hold persistence mode.
    cap_can_revert = failure is None and action == "cap" and set_outcome.persistence_enabled is False
    if cap_can_revert:
        message = f"{message} (persistence mode is off after -pm 1; this cap can revert on its own)"
    if failure is not None:
        level = logging.ERROR
    elif cap_can_revert:
        level = logging.WARNING
    else:
        level = logging.INFO
    _log(
        level,
        message,
        {
            "gpu_power_action": action,
            "executor_uuid": executor_id,
            "gpu_uuid": gpu_uuid,
            "watts_before": watts_before,
            "watts_after": watts_after,
            "status": status,
            "persistence_enabled": set_outcome.persistence_enabled,
        },
        log_extra,
    )
    return failure is None


async def _ensure_restore_record(
    redis: RedisService,
    gpu_uuid: str,
    pre_cap_watts: int,
    pod_id: str,
    executor_id: str,
    log_extra: dict[str, object] | None,
) -> bool:
    key = _restore_key(gpu_uuid)
    try:
        existing_raw: str | bytes | None = await redis.get(key)
    except Exception as exc:
        _log(logging.ERROR, f"gpu power cap: redis read failed for {key}: {exc}", {"gpu_uuid": gpu_uuid}, log_extra)
        return False
    if existing_raw:
        # Frozen invariant: a leftover record (earlier restore failed) holds the TRUE original limit.
        _log(
            logging.WARNING,
            f"gpu power cap: keeping frozen pre-cap record for {gpu_uuid} (an earlier restore failed); "
            f"current limit {pre_cap_watts}W is NOT recorded",
            {"gpu_uuid": gpu_uuid},
            log_extra,
        )
        return True
    record = GpuPowerRestoreRecord(
        gpu_uuid=gpu_uuid, watts=pre_cap_watts, pod_id=pod_id, executor_id=executor_id, capped_at=time.time()
    )
    try:
        await redis.set(key, record.model_dump_json())
        return True
    except Exception as exc:
        _log(logging.ERROR, f"gpu power cap: could not persist pre-cap record for {gpu_uuid}: {exc}", {"gpu_uuid": gpu_uuid}, log_extra)
        return False


async def read_gpu_power_restore_records(
    redis: RedisService,
    gpu_uuids: list[str],
    log_extra: dict[str, object] | None = None,
) -> GpuPowerRestoreReadResult:
    """Read the frozen restore records for these GPUs. Pure read: a corrupt record is logged and
    skipped (it can't grant a check pass and can't be restored; cleanup is manual); a failed Redis
    read is logged and reported via ``read_failed``."""
    records: list[GpuPowerRestoreRecord] = []
    read_failed = False
    for gpu_uuid in gpu_uuids:
        key = _restore_key(gpu_uuid)
        try:
            raw: str | bytes | None = await redis.get(key)
        except Exception as exc:
            _log(logging.ERROR, f"gpu power restore: redis read failed for {key}: {exc}", {"gpu_uuid": gpu_uuid}, log_extra)
            read_failed = True
            continue
        if not raw:
            continue
        try:
            records.append(GpuPowerRestoreRecord.model_validate_json(raw))
        except (ValidationError, TypeError, ValueError):
            _log(logging.ERROR, f"gpu power restore: ignoring corrupt record {raw!r} for {gpu_uuid}", {"gpu_uuid": gpu_uuid}, log_extra)
    return GpuPowerRestoreReadResult(records=records, read_failed=read_failed)


async def _delete_pod_index(redis: RedisService, pod_id: str, log_extra: dict[str, object] | None) -> None:
    try:
        await redis.delete(_pod_index_key(pod_id))
    except Exception as exc:
        _log(logging.ERROR, f"gpu power restore: could not clear pod index for {pod_id}: {exc}", {}, log_extra)


async def _restore_records(
    ssh: asyncssh.SSHClientConnection,
    redis: RedisService,
    records: list[GpuPowerRestoreRecord],
    state_by_uuid: dict[str, GpuPowerState],
    log_extra: dict[str, object] | None,
) -> int:
    """Apply each record with ``nvidia-smi -pl``; delete a record ONLY after its restore succeeded
    (a failed restore keeps it for the safety nets to retry). Returns the restored count."""
    restored = 0
    for record in records:
        state = state_by_uuid.get(record.gpu_uuid)
        watts_before = state.current_watts if state else None
        changed = await _set_and_log_power_limit(
            ssh, "restore", record.executor_id, record.gpu_uuid, watts_before, record.watts, log_extra
        )
        if not changed:
            continue
        try:
            await redis.delete(_restore_key(record.gpu_uuid))
            restored += 1
        except Exception as exc:
            _log(
                logging.ERROR,
                f"gpu power restore: restored {record.gpu_uuid} but could not clear its record: {exc}; "
                f"a duplicate restore may follow",
                {"gpu_uuid": record.gpu_uuid},
                log_extra,
            )
    return restored


async def restore_tracked_gpu_power_limits(
    ssh: asyncssh.SSHClientConnection,
    redis: RedisService,
    gpu_uuids: list[str],
    log_extra: dict[str, object] | None = None,
) -> int:
    """Restore the frozen pre-cap limit of every tracked GPU among ``gpu_uuids``.

    Best-effort (logs, never raises); returns the number of records restored and cleared.
    """
    read_result = await read_gpu_power_restore_records(redis, gpu_uuids, log_extra)
    if not read_result.records:
        return 0
    try:
        state_by_uuid = await _query_power_state(ssh)  # before-values for the change log only
    except Exception as exc:
        _log(logging.WARNING, f"gpu power restore: state query failed: {exc}; restoring without before-values", {}, log_extra)
        state_by_uuid = {}
    return await _restore_records(ssh, redis, read_result.records, state_by_uuid, log_extra)


async def restore_all_host_gpu_power_limits(
    ssh: asyncssh.SSHClientConnection,
    redis: RedisService,
    log_extra: dict[str, object] | None = None,
) -> int:
    """Enumerate the host's GPUs over SSH and restore every tracked one — for whole-node containers
    whose payload names no gpu_uuids. Best-effort; returns the restored count."""
    try:
        state_by_uuid = await _query_power_state(ssh)
    except Exception as exc:
        _log(logging.ERROR, f"gpu power restore: state query failed: {exc}; will retry later", {}, log_extra)
        return 0
    read_result = await read_gpu_power_restore_records(redis, list(state_by_uuid), log_extra)
    if not read_result.records:
        return 0
    return await _restore_records(ssh, redis, read_result.records, state_by_uuid, log_extra)


async def raise_low_power_limits_to_default(
    ssh: asyncssh.SSHClientConnection,
    executor_id: str,
    gpu_uuids: list[str] | None,
    log_extra: dict[str, object] | None = None,
) -> int:
    """State-free last-resort net for rental start: lift every GPU sitting below
    ``MIN_POWER_LIMIT_RATIO`` x its default limit back to the default.

    Needs no Redis — the GPU itself reports both limits. Covers a lost/corrupt restore record:
    the exact pre-cap value is gone, but the default equals the post-reboot state, so a customer
    never starts on a below-floor GPU no matter what happened to our state. GPUs between the
    floor and the default are left alone (a miner may legitimately run there). ``gpu_uuids=None``
    means every host GPU. Best-effort (never raises); returns the raised count.
    """
    try:
        state_by_uuid = await _query_power_state(ssh)
    except Exception as exc:
        _log(logging.ERROR, f"gpu power raise: state query failed: {exc}; leaving limits as-is", {}, log_extra)
        return 0
    target_uuids: list[str] = gpu_uuids if gpu_uuids else list(state_by_uuid)
    raised = 0
    for gpu_uuid in target_uuids:
        state = state_by_uuid.get(gpu_uuid)
        if state is None or state.default_watts is None:
            continue
        if state.current_watts >= MIN_POWER_LIMIT_RATIO * state.default_watts:
            continue
        lifted = await _set_and_log_power_limit(
            ssh, "raise", executor_id, gpu_uuid, state.current_watts, state.default_watts, log_extra
        )
        if lifted:
            raised += 1
    return raised


async def restore_filler_pod_gpu_power_limits(
    ssh: asyncssh.SSHClientConnection,
    redis: RedisService,
    pod_id: str,
    log_extra: dict[str, object] | None = None,
) -> int:
    """Restore exactly the GPUs this filler pod capped (its pod index), then drop the index.

    Touching only the pod's own records means a replacement filler's fresh caps on the same host are
    never swept, and a filler we never capped costs one Redis read — no SSH. Best-effort: a record
    whose restore failed is kept for the safety nets; the index is dropped either way (the per-GPU
    records, not the index, are the source of truth). Returns the restored count.
    """
    index_key = _pod_index_key(pod_id)
    try:
        raw_index: str | bytes | None = await redis.get(index_key)
    except Exception as exc:
        _log(logging.ERROR, f"gpu power restore: redis read failed for {index_key}: {exc}; will retry later", {}, log_extra)
        return 0
    if not raw_index:
        return 0
    try:
        capped_uuids: list[str] = [str(uuid) for uuid in json.loads(raw_index)]
    except (ValueError, TypeError):
        _log(logging.ERROR, f"gpu power restore: dropping corrupt pod index {raw_index!r} for {pod_id}", {}, log_extra)
        await _delete_pod_index(redis, pod_id, log_extra)
        return 0
    restored = await restore_tracked_gpu_power_limits(ssh, redis, capped_uuids, log_extra)
    await _delete_pod_index(redis, pod_id, log_extra)
    return restored


async def _undo_partial_apply(
    ssh: asyncssh.SSHClientConnection,
    redis: RedisService,
    gpu_uuids: list[str],
    pod_id: str,
    log_extra: dict[str, object] | None,
) -> None:
    # A failed apply must not leave GPUs capped or state keys behind: with the filler never starting,
    # the backend keeps reporting owner="lium", which blocks the check-side restore indefinitely.
    await restore_tracked_gpu_power_limits(ssh, redis, gpu_uuids, log_extra)
    await _delete_pod_index(redis, pod_id, log_extra)


async def apply_filler_gpu_power_limits(
    ssh: asyncssh.SSHClientConnection,
    gpu_power_limits: list[GpuPowerLimit],
    redis: RedisService,
    pod_id: str,
    executor_id: str,
    log_extra: dict[str, object] | None = None,
) -> bool:
    """Record each GPU's pre-cap limit (frozen), then cap it at the target watts (clamped to hw [min, max]).

    Fail-closed: returns True only when EVERY requested GPU was capped; on failure any partial work
    (records, pod index, already-capped GPUs) is undone first. False means the caller must NOT start
    the filler. Every failure path logs an error so it is diagnosable/alertable — no silent skips.
    """
    try:
        state_by_uuid = await _query_power_state(ssh)
    except Exception as exc:
        _log(logging.ERROR, f"gpu power cap: state query failed: {exc}; refusing to start filler uncapped", {}, log_extra)
        return False
    missing = [target.gpu_uuid for target in gpu_power_limits if target.gpu_uuid not in state_by_uuid]
    if missing:
        _log(
            logging.ERROR,
            f"gpu power cap: requested GPU(s) {missing} not in nvidia-smi output {sorted(state_by_uuid)}; "
            f"refusing to start filler uncapped",
            {},
            log_extra,
        )
        return False
    target_uuids = [target.gpu_uuid for target in gpu_power_limits]
    # Persist every restore record BEFORE lowering anything: never cap a GPU without a stored way back.
    for target in gpu_power_limits:
        pre_cap_watts = state_by_uuid[target.gpu_uuid].current_watts
        if not await _ensure_restore_record(redis, target.gpu_uuid, pre_cap_watts, pod_id, executor_id, log_extra):
            await _undo_partial_apply(ssh, redis, target_uuids, pod_id, log_extra)
            return False
    try:
        await redis.set(_pod_index_key(pod_id), json.dumps(target_uuids))
    except Exception as exc:
        _log(logging.ERROR, f"gpu power cap: could not persist pod index for {pod_id}: {exc}", {}, log_extra)
        await _undo_partial_apply(ssh, redis, target_uuids, pod_id, log_extra)
        return False
    all_set = True
    for target in gpu_power_limits:
        state = state_by_uuid[target.gpu_uuid]
        target_watts = _clamp_watts(target.watts, state)
        if not await _set_and_log_power_limit(
            ssh, "cap", executor_id, target.gpu_uuid, state.current_watts, target_watts, log_extra
        ):
            all_set = False
    if not all_set:
        await _undo_partial_apply(ssh, redis, target_uuids, pod_id, log_extra)
    return all_set
