from __future__ import annotations

import logging
import time
from dataclasses import replace

from core.config import settings
from core.utils import _m, get_extra_info
from protocol.vc_protocol.compute_requests import RentedExecutorsResponse

from ..messages import IdleDiskWriteMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context
from .container_download_speed import download_speed_command
from .rented_throughput import (
    _DISK_SCRIPT,
    _SNAPSHOT,
    DISK_TIMEOUT_SECONDS,
    parse_busy,
    parse_disk,
    phase_offset_seconds,
)
from .task_time import task_seconds_left

logger = logging.getLogger(__name__)

INTERVAL_SECONDS = 24 * 3600
# the 2 s busy window before the write
SNAPSHOT_TIMEOUT_SECONDS = 30
# container start, the O_DIRECT write and the buffered retry
WRITE_TIMEOUT_SECONDS = 2 * DISK_TIMEOUT_SECONDS + 30
# the backend's rented re-read: one un-retried read with a 30 s timeout
_RENTED_REREAD_TIMEOUT_SECONDS = 30
# Redis, scoring and the result save after this check
_AFTER_CHECK_SLACK_SECONDS = 60
_RESERVE_SECONDS = (
    SNAPSHOT_TIMEOUT_SECONDS + _RENTED_REREAD_TIMEOUT_SECONDS + WRITE_TIMEOUT_SECONDS + _AFTER_CHECK_SLACK_SECONDS
)
_REDIS_PREFIX = "idle_disk_write"
# DAH-5385: the last reading and when it was taken (epoch seconds), sent in the specs every cycle (the backend keeps
# only the latest specs) for filler start
SPECS_KEY = "idle_disk_write_mb_per_s"
SPECS_MEASURED_AT_KEY = "idle_disk_write_measured_at"
_READING_PREFIX = "idle_disk_write_reading"
# from the write only: a node that cannot be measured again stops sending its figure after this
_READING_TTL_SECONDS = 3 * INTERVAL_SECONDS


class IdleDiskWriteCheck:
    """DAH-5320: the disk a renter's container writes to, on idle and filler nodes. A renter's pod on a node crawled
    because the docker root disk wrote 12.8 MB/s with fsync; RentedThroughputCheck only reads rented nodes.

    Once a day per node at a per-node phase, last in the pipeline (after VerifyX, the matmul, rental verification
    and the rental probe, so it never takes their time): unless the disks under the docker root already move
    RENTED_THROUGHPUT_DISK_BUSY_MB_PER_S, a throwaway container writes RentedThroughputCheck's DISK_PROBE_BYTES with
    fsync into its writable layer. Logs IDLE_DISK_WRITE with the MB/s, the card, the provider and a shadow verdict
    (fail under RENTED_THROUGHPUT_DISK_MIN_MB_PER_S). Always passes: IDLE_DISK_WRITE_ENFORCE_ENABLED is read by nothing.
    A node running a filler counts as busy from IDLE_DISK_WRITE_FILLER_BUSY_MB_PER_S (sectors read plus written): a
    filler moving its disk under the rented busy line would otherwise lower the reading. DAH-5385: every cycle that
    reaches this check sends the last reading to the backend as specs[SPECS_KEY]; filler start keeps the big fillers
    off a slow disk. A node the backend still rents is never measured: TenantEnforcementCheck halts most such cycles
    earlier, and its passing returns (a rental of another hotkey or address, a stale pod) reach _skip_reason's re-read
    of the rented list, so the backend holds the last reading it had.
    """

    check_id = "executor.validate.idle_disk_write"
    fatal = False

    async def run(self, ctx: Context) -> CheckResult:
        if not settings.IDLE_DISK_WRITE_CHECK_ENABLED:
            return self._result(ctx, {"reason": "IDLE_DISK_WRITE_CHECK_ENABLED is off"}, None)
        what: dict = {}
        try:
            what["reason"] = await self._measure(ctx, what)
        except Exception:  # noqa: BLE001 — shadow only: nothing it hits may change the result
            logger.warning(_m("Idle disk write: probe failed", extra=get_extra_info(ctx.default_extra)))
            what["reason"] = "probe failed"
        return self._result(ctx, what, what.pop("reading", None) or await _last_reading(ctx))

    async def _measure(self, ctx: Context, what: dict) -> str:
        key = f"{_REDIS_PREFIX}:{ctx.miner_hotkey}:{ctx.executor.uuid}"
        skip = await self._skip_reason(ctx, key, what)
        if skip is not None:
            return skip
        await ctx.services.redis.set(key, str(time.time()), ex=2 * INTERVAL_SECONDS)
        result = await ctx.runner.run(
            download_speed_command(bool(ctx.state.sysbox_runtime), _DISK_SCRIPT),
            timeout=WRITE_TIMEOUT_SECONDS,
            retryable=False,
        )
        disk = parse_disk(result.stdout or "")
        verdict = None
        if disk.mb_per_s is not None:
            # sent this cycle even if Redis then fails to keep it
            what["reading"] = (disk.mb_per_s, time.time())
            verdict = "fail" if disk.mb_per_s < settings.RENTED_THROUGHPUT_DISK_MIN_MB_PER_S else "pass"
        gpu_details = ctx.state.gpu_details or ctx.state.specs.get("gpu", {}).get("details", [])
        logger.info(
            _m(
                "IDLE_DISK_WRITE",
                extra=get_extra_info(
                    {
                        **ctx.default_extra,
                        "mb_per_s": disk.mb_per_s,
                        "bytes": disk.bytes,
                        "seconds": disk.seconds,
                        "error": disk.error,
                        "disk_min_mb_per_s": settings.RENTED_THROUGHPUT_DISK_MIN_MB_PER_S,
                        "gpu_model": ctx.state.gpu_model or (gpu_details[0].get("name") if gpu_details else None),
                        "miner_hotkey": ctx.miner_hotkey,
                        "filler": _has_filler(ctx.state.rented_data, ctx),
                        "shadow_verdict": verdict,
                        "enforced": False,
                    }
                ),
            )
        )
        if disk.mb_per_s is not None:
            try:
                await ctx.services.redis.set(
                    _reading_key(ctx), f"{disk.mb_per_s}:{what['reading'][1]}", ex=_READING_TTL_SECONDS
                )
            except Exception:  # noqa: BLE001 — the next cycles send nothing until the next probe; this one still does
                logger.warning(_m("Idle disk write: reading not stored", extra=get_extra_info(ctx.default_extra)))
        return f"measured: {verdict or 'no verdict'}"

    async def _skip_reason(self, ctx: Context, key: str, what: dict) -> str | None:
        """Why not now: not due, too little task time, or the node not idle. None = write now."""
        redis = ctx.services.redis
        raw = await redis.get(key)
        now = time.time()
        if raw is None:
            # first sight: due at the node's phase within the next day, so a deploy doesn't write on the fleet at once
            await redis.set(key, str(now - _node_fraction_of_day(ctx) * INTERVAL_SECONDS), ex=2 * INTERVAL_SECONDS)
            return "first sight"
        if now < float(raw) + INTERVAL_SECONDS:
            return "not due"
        if task_seconds_left(ctx) < _RESERVE_SECONDS:
            return "no time left"

        # first, so a busy node (never stamped, so due every cycle) costs one 2 s read and no backend list
        snapshot = await ctx.runner.run(_SNAPSHOT, timeout=SNAPSHOT_TIMEOUT_SECONDS, retryable=False)
        _, disk_busy = parse_busy(snapshot.stdout or "")
        what["disk_busy_mb_per_s"] = disk_busy
        if disk_busy is None:
            # the docker root's disks are not resolvable (btrfs/zfs, no host view): no later cycle reads them either
            await redis.set(key, str(now), ex=2 * INTERVAL_SECONDS)
            return "docker root disk not resolvable"
        busy = f"disk busy ({disk_busy:.0f} MB/s)"
        if disk_busy >= settings.RENTED_THROUGHPUT_DISK_BUSY_MB_PER_S:
            return busy
        filler_busy = disk_busy >= settings.IDLE_DISK_WRITE_FILLER_BUSY_MB_PER_S
        if filler_busy and _has_filler(ctx.state.rented_data, ctx):
            return busy
        fresh = await ctx.services.backend.get_rented_executors_now()
        if fresh is None:
            return "rented state unknown"
        # a filler started since the cycle-start snapshot (capability.py's race): its download is not the disk's speed
        if filler_busy and _has_filler(fresh, ctx):
            return busy
        rented_now = fresh.executors.get(ctx.executor.uuid)
        if rented_now is not None and rented_now.pods:
            return "rented"
        if await redis.renting_in_progress(ctx.miner_hotkey, ctx.executor.uuid):
            return "rent in progress"
        return None

    def _result(self, ctx: Context, what: dict, reading: tuple[float, float] | None) -> CheckResult:
        if reading is not None:
            what["mb_per_s_sent"], what["measured_at"] = reading
        event = render_message(Msg.LOGGED, ctx=ctx, check_id=self.check_id, what=what)
        if reading is None:
            return CheckResult(passed=True, event=event)
        specs = {**ctx.state.specs, SPECS_KEY: reading[0], SPECS_MEASURED_AT_KEY: reading[1]}
        return CheckResult(passed=True, event=event, updates={"state": replace(ctx.state, specs=specs)})


def _reading_key(ctx: Context) -> str:
    return f"{_READING_PREFIX}:{ctx.miner_hotkey}:{ctx.executor.uuid}"


async def _last_reading(ctx: Context) -> tuple[float, float] | None:
    """The stored (MB/s, measured_at); not refreshed here, so it expires _READING_TTL_SECONDS after its probe."""
    try:
        raw = await ctx.services.redis.get(_reading_key(ctx))
        if raw is None:
            return None
        mb_per_s, measured_at = str(raw if isinstance(raw, str) else raw.decode()).split(":")
        return float(mb_per_s), float(measured_at)
    except Exception:  # noqa: BLE001 — no reading is sent as none: the backend then allows every filler
        return None


def _node_fraction_of_day(ctx: Context) -> float:
    """The node's fixed place in the day, from the same hash as RentedThroughputCheck's phase."""
    return phase_offset_seconds(ctx.executor.uuid) / (settings.RENTED_THROUGHPUT_INTERVAL_HOURS * 3600)


def _has_filler(rented_data: RentedExecutorsResponse | None, ctx: Context) -> bool:
    return bool(rented_data and rented_data.get_filler_containers(ctx.executor.uuid))
