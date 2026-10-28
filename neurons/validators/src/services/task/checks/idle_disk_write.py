from __future__ import annotations

import logging
import time

from core.config import settings
from core.utils import _m, get_extra_info

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

logger = logging.getLogger(__name__)

INTERVAL_SECONDS = 24 * 3600
# the 2 s busy window before the write
SNAPSHOT_TIMEOUT_SECONDS = 30
# container start, the O_DIRECT write and the buffered retry
WRITE_TIMEOUT_SECONDS = 2 * DISK_TIMEOUT_SECONDS + 30
# the backend's rented re-read: one un-retried read with a 30 s timeout
_RENTED_REREAD_TIMEOUT_SECONDS = 30
# the executor task is cancelled at JOB_TIME_OUT - 120 (miner_service)
_TASK_TIMEOUT_MARGIN_SECONDS = 120
_REDIS_PREFIX = "idle_disk_write"


class IdleDiskWriteCheck:
    """DAH-5320: the disk a renter's container writes to, on idle and filler nodes. A renter's pod on a node crawled
    because the docker root disk wrote 12.8 MB/s with fsync; RentedThroughputCheck only reads rented nodes.

    Once a day per node at a per-node phase, last in the pipeline (after VerifyX, the matmul, rental verification
    and the rental probe, so it never takes their time): unless the disks under the docker root already move
    RENTED_THROUGHPUT_DISK_BUSY_MB_PER_S, a throwaway container writes RentedThroughputCheck's DISK_PROBE_BYTES with
    fsync into its writable layer. Logs IDLE_DISK_WRITE with the MB/s, the card, the provider and a shadow verdict
    (fail under RENTED_THROUGHPUT_DISK_MIN_MB_PER_S). Always passes: IDLE_DISK_WRITE_ENFORCE_ENABLED is read by nothing.
    """

    check_id = "executor.validate.idle_disk_write"
    fatal = False

    async def run(self, ctx: Context) -> CheckResult:
        if not settings.IDLE_DISK_WRITE_CHECK_ENABLED:
            return self._result(ctx, "IDLE_DISK_WRITE_CHECK_ENABLED is off")
        rented_data = ctx.state.rented_data
        rented_executor = rented_data.executors.get(ctx.executor.uuid) if rented_data else None
        if rented_executor is not None and rented_executor.pods:
            return self._result(ctx, "rented")
        try:
            return self._result(ctx, await self._measure(ctx))
        except Exception:  # noqa: BLE001 — shadow only: nothing it hits may change the result
            logger.warning(_m("Idle disk write: probe failed", extra=get_extra_info(ctx.default_extra)))
            return self._result(ctx, "probe failed")

    async def _measure(self, ctx: Context) -> str:
        key = f"{_REDIS_PREFIX}:{ctx.miner_hotkey}:{ctx.executor.uuid}"
        redis = ctx.services.redis
        raw = await redis.get(key)
        now = time.time()
        if raw is None:
            # first sight: due at the node's phase within the next day, so a deploy doesn't write on the fleet at once
            await redis.set(key, str(now - _phase_fraction(ctx) * INTERVAL_SECONDS), ex=2 * INTERVAL_SECONDS)
            return "first sight"
        if now < float(raw) + INTERVAL_SECONDS:
            return "not due"
        reserve = SNAPSHOT_TIMEOUT_SECONDS + _RENTED_REREAD_TIMEOUT_SECONDS + WRITE_TIMEOUT_SECONDS
        left = settings.JOB_TIME_OUT - _TASK_TIMEOUT_MARGIN_SECONDS - (time.monotonic() - ctx.started_at_monotonic)
        if left < reserve:
            return "no time left"

        # first, so a busy node (never stamped, so due every cycle) costs one 2 s read and no backend list
        snapshot = await ctx.runner.run(_SNAPSHOT, timeout=SNAPSHOT_TIMEOUT_SECONDS, retryable=False)
        _, disk_busy = parse_busy(snapshot.stdout or "")
        if disk_busy is None or disk_busy >= settings.RENTED_THROUGHPUT_DISK_BUSY_MB_PER_S:
            return f"disk busy ({disk_busy} MB/s)"
        fresh = await ctx.services.backend.get_rented_executors_now()
        if fresh is None:
            return "rented state unknown"
        rented_now = fresh.executors.get(ctx.executor.uuid)
        if rented_now is not None and rented_now.pods:
            return "rented"
        if await redis.renting_in_progress(ctx.miner_hotkey, ctx.executor.uuid):
            return "rent in progress"

        await redis.set(key, str(now), ex=2 * INTERVAL_SECONDS)
        result = await ctx.runner.run(
            download_speed_command(bool(ctx.state.sysbox_runtime), _DISK_SCRIPT),
            timeout=WRITE_TIMEOUT_SECONDS,
            retryable=False,
        )
        disk = parse_disk(result.stdout or "")
        verdict = None
        if disk.mb_per_s is not None:
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
                        "other_disk_mb_per_s": disk_busy,
                        "disk_min_mb_per_s": settings.RENTED_THROUGHPUT_DISK_MIN_MB_PER_S,
                        "gpu_model": ctx.state.gpu_model or (gpu_details[0].get("name") if gpu_details else None),
                        "miner_hotkey": ctx.miner_hotkey,
                        "filler": bool(_fillers(ctx)),
                        "shadow_verdict": verdict,
                        "enforced": False,
                    }
                ),
            )
        )
        return f"measured: {verdict or 'no verdict'}"

    def _result(self, ctx: Context, reason: str) -> CheckResult:
        what = {"reason": reason}
        return CheckResult(passed=True, event=render_message(Msg.LOGGED, ctx=ctx, check_id=self.check_id, what=what))


def _phase_fraction(ctx: Context) -> float:
    """The node's place in the day, from the same hash as RentedThroughputCheck's phase."""
    return phase_offset_seconds(ctx.executor.uuid) / (settings.RENTED_THROUGHPUT_INTERVAL_HOURS * 3600)


def _fillers(ctx: Context) -> list:
    rented_data = ctx.state.rented_data
    return rented_data.get_filler_containers(ctx.executor.uuid) if rented_data else []
