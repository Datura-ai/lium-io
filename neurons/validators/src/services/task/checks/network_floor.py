"""NETWORK_FLOOR (shadow): a bandwidth floor that grows with the node's GPU count.

VerifyXCheck holds every node to one download floor whatever its size, and a rented node keeps its last
EMA because VerifyX does not run beside a rental. This check reads the stored VerifyX download and upload
EMAs against a floor scaled by GPU count and logs one `network_floor_shadow` line. It never changes the
score and never fails the cycle; the existing gate is untouched.
"""

from __future__ import annotations

import logging
from typing import Any

from core.config import settings
from core.utils import _m, get_extra_info

from ..messages import NetworkFloorMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context

logger = logging.getLogger(__name__)

SHADOW_LOG_KEY = "network_floor_shadow"


def scaled_download_floor_mbps(gpu_count: int) -> float:
    """MIN at 1 GPU up to MAX at NETWORK_FLOOR_FULL_GPU_COUNT GPUs, linear per GPU, flat above."""
    low = settings.NETWORK_FLOOR_MIN_DOWNLOAD_MBPS
    high = settings.NETWORK_FLOOR_MAX_DOWNLOAD_MBPS
    full = settings.NETWORK_FLOOR_FULL_GPU_COUNT
    gpus = min(max(gpu_count, 1), full)
    return round(low + (high - low) * (gpus - 1) / (full - 1), 1)


def _reading(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return None
    return float(value)


class NetworkFloorShadowCheck:
    """Shadow GPU-count-scaled bandwidth floor; non-fatal, never touches the score."""

    check_id = "network.validate.scaled_floor"
    fatal = False

    async def run(self, ctx: Context) -> CheckResult:
        if not settings.NETWORK_FLOOR_SCALED_SHADOW_ENABLED:
            return CheckResult(
                passed=True, event=render_message(Msg.DISABLED, ctx=ctx, check_id=self.check_id)
            )

        rented_data = ctx.state.rented_data
        ema = rented_data.network_ema.get(ctx.executor.uuid) if rented_data else None
        download = _reading(ema.ema_verifyx_download_speed) if ema else None
        upload = _reading(ema.ema_verifyx_upload_speed) if ema else None
        gpu_count = ctx.state.gpu_count
        if gpu_count is None:
            gpu_count = ((ctx.state.specs or {}).get("gpu") or {}).get("count") or 0
        rented_executor = rented_data.executors.get(ctx.executor.uuid) if rented_data else None

        what: dict[str, Any] = {
            "executor_uuid": ctx.executor.uuid,
            "shadow": True,
            "gpu_count": gpu_count,
            "rented": bool(rented_executor and rented_executor.pods),
            "ema_download_mbps": download,
            "ema_upload_mbps": upload,
        }
        if download is None or not gpu_count:
            what["verdict"] = "no_reading"
            self._log(ctx, what, logging.INFO)
            return CheckResult(
                passed=True,
                event=render_message(Msg.NO_READING, ctx=ctx, check_id=self.check_id, what=what),
            )

        download_floor = scaled_download_floor_mbps(gpu_count)
        upload_floor = round(download_floor * settings.NETWORK_FLOOR_UPLOAD_RATIO, 1)
        below = []
        if download < download_floor:
            below.append("download")
        if upload is not None and upload < upload_floor:
            below.append("upload")
        what.update(
            download_floor_mbps=download_floor,
            upload_floor_mbps=upload_floor,
            below=below,
            verdict="below_floor" if below else "ok",
        )
        if below:
            self._log(ctx, what, logging.WARNING)
            return CheckResult(
                passed=True,
                event=render_message(Msg.SHADOW_BELOW, ctx=ctx, check_id=self.check_id, what=what),
            )
        self._log(ctx, what, logging.INFO)
        return CheckResult(
            passed=True,
            event=render_message(Msg.SHADOW_OK, ctx=ctx, check_id=self.check_id, what=what),
        )

    @staticmethod
    def _log(ctx: Context, what: dict[str, Any], level: int) -> None:
        logger.log(level, _m(SHADOW_LOG_KEY, extra=get_extra_info({**ctx.default_extra, **what})))
