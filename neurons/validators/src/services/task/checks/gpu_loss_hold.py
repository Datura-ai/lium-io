"""GPU_LOST — a node whose GPU dropped out is hidden and takes no new rentals until the card stays back.

A card that falls out of NVML ('Unknown Error', lost, fallen off the bus) can read clean on the next scrape and drop
out again an hour later, while a renter's container on it keeps failing. The fatal GPU checks after the scrape
(GpuModelValidCheck's DETAILS_MISMATCH, GpuFingerprintCheck) judge only the scrape in front of them, so such a node
was listed again on its first clean scrape, and a rented one passed the rented check with score 1 in between.

Every cycle this check judges the scrape with `judge_rented_gpus` (the rental's GPU count, the driver's count, the
anchored UUID set, the NVML error code; an idle node counts as renting one GPU). A fault puts the node on a hold of
GPU_LOSS_HOLD_SECONDS, renewed by every faulty cycle; weak evidence (`confirm_first`) starts the hold only when the
cycle before also showed a fault. Every cycle in the hold carries a GPU_LOST availability entry: the backend hides
the node and refuses new rentals, and leaves `active`, billing and a rental already on the node alone. The first
cycle after the hold ran out carries none, and the node is listed again.

The check's own verdict: a cycle that shows the fault goes on (the fatal GPU checks after it judge it as before,
verification reset included), except an NVML loss code over a full card list, which no later check fails: that is
fatal here on an idle node. A clean cycle inside the hold fails: fatally on an idle node; not under a live
rental, as BannedGpuCheck, so the renter's pod keeps being verified and billed.

The hold is keyed by miner hotkey and executor UUID (the UUID is the miner's word, so one miner cannot hold another's
node). A dry run reads it and writes nothing. A Redis that cannot be read is no hold: the verdict never rests on Redis
being up.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from services.redis_service import GPU_ANCHOR_KEY

from core.config import settings
from core.utils import _m, get_extra_info

from ..availability import ReachSource, ReachTarget
from ..messages import GpuLossHoldMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context
from .gpu_fingerprint import split_uuids
from .rented_gpu_drop import (
    FAULT_NVML_ERROR,
    GPU_DROP_REDIS_ERRORS,
    GpuDrop,
    _listed_uuids,
    _rented_gpu_count,
    judge_rented_gpus,
)

logger = logging.getLogger(__name__)

GPU_LOSS_HOLD_KEY_PREFIX = "gpu_loss_hold"
# A weak first sighting waits this long for the next cycle (15 min apart) to confirm it.
GPU_LOSS_CONFIRM_SECONDS = 45 * 60


def _key(ctx: Context) -> str:
    return f"{GPU_LOSS_HOLD_KEY_PREFIX}:{ctx.miner_hotkey}:{ctx.executor.uuid}"


@dataclass(frozen=True)
class DropMark:
    """The hold as kept in Redis; stored as {"held", "since", "last_seen"}."""

    is_held: bool
    since: str
    last_seen: str

    @classmethod
    def load(cls, raw: object) -> DropMark | None:
        loaded = json.loads(raw) if raw else None
        if not isinstance(loaded, dict) or not loaded.get("last_seen"):
            return None
        last_seen = loaded["last_seen"]
        return cls(bool(loaded.get("held")), loaded.get("since") or last_seen, last_seen)

    def dump(self) -> str:
        return json.dumps({"held": self.is_held, "since": self.since, "last_seen": self.last_seen})


def rented_pods(ctx: Context) -> list:
    """This miner's pods on the executor; a rental under another hotkey is not this node's."""
    rented_data = ctx.state.rented_data
    rented = rented_data.executors.get(ctx.executor.uuid) if rented_data else None
    if rented is None or rented.miner_hotkey != ctx.miner_hotkey:
        return []
    return list(rented.pods)


def judge_scrape(ctx: Context, pods: list) -> GpuDrop | None:
    """The GPU fault this cycle's scrape shows, judged as RentedGpuDropCheck judges a rented node."""
    specs = ctx.state.specs or {}
    gpu_details = list(ctx.state.gpu_details or [])
    nvml_count = ctx.state.gpu_count
    if nvml_count is None:
        nvml_count = (specs.get("gpu") or {}).get("count", 0) or 0
    return judge_rented_gpus(
        rented_gpu_count=_rented_gpu_count(pods),
        anchor_uuids=split_uuids((ctx.verified or {}).get(GPU_ANCHOR_KEY) or ""),
        nvml_count=nvml_count,
        listed_uuids=_listed_uuids(gpu_details),
        listed_count=len(gpu_details),
        scrape_error=specs.get("gpu_scrape_error"),
    )


class GpuLossHoldCheck:
    """Hide a node whose GPU dropped out, until its cards have read clean for GPU_LOSS_HOLD_SECONDS.

    See the module docstring for the rule. Runs right after the scrape, before the fatal GPU checks and the
    rented short-circuit, so rented and idle nodes are judged alike.
    """

    check_id = "gpu.validate.loss_hold"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        pods = rented_pods(ctx)
        drop = judge_scrape(ctx, pods)
        key = _key(ctx)
        mark: DropMark | None = None
        try:
            mark = DropMark.load(await ctx.services.redis.get(key))
        except (*GPU_DROP_REDIS_ERRORS, TypeError, ValueError):
            self._log_redis_unavailable(ctx, "read")
        now = datetime.now(UTC)

        if drop is not None:
            mark = DropMark(
                is_held=mark is not None or not drop.confirm_first,
                since=mark.since if mark else now.isoformat(),
                last_seen=now.isoformat(),
            )
            await self._save(
                ctx,
                key,
                mark,
                settings.GPU_LOSS_HOLD_SECONDS if mark.is_held else GPU_LOSS_CONFIRM_SECONDS,
            )
            what = {
                "faults": drop.faults,
                "missing_uuids": drop.missing_uuids,
                "nvml_error_code": drop.nvml_error_code,
                "expected_gpu_count": drop.expected,
                "visible_gpu_count": drop.visible,
            }
            if not mark.is_held:
                event = render_message(Msg.PENDING, ctx=ctx, check_id=self.check_id, what=what)
                return CheckResult(passed=True, event=event)
            # the fatal GPU checks after this one judge the faulty scrape itself, as before; an NVML
            # loss code over a full card list fails none of them, so on an idle node it is fatal here
            nvml_only = drop.faults == [FAULT_NVML_ERROR]
            return self._lost(ctx, mark, what, fatal=nvml_only and not pods)

        if mark is None:
            return CheckResult(
                passed=True, event=render_message(Msg.OK, ctx=ctx, check_id=self.check_id)
            )
        if not mark.is_held:
            # a weak sighting the next cycle did not confirm
            await self._forget(ctx, key)
            return CheckResult(
                passed=True, event=render_message(Msg.OK, ctx=ctx, check_id=self.check_id)
            )
        return self._lost(ctx, mark, {"faults": []}, fatal=not pods)

    def _lost(self, ctx: Context, mark: DropMark, what: dict, *, fatal: bool) -> CheckResult:
        last_seen = datetime.fromisoformat(mark.last_seen)
        event = render_message(
            Msg.LOST,
            ctx=ctx,
            check_id=self.check_id,
            what={
                "reach_source": str(ReachSource.VALIDATOR),
                "reach_target": str(ReachTarget.GPU),
                "executor_uuid": ctx.executor.uuid,
                **what,
                "since": mark.since,
                "last_seen": mark.last_seen,
                "hold_until": (
                    last_seen + timedelta(seconds=settings.GPU_LOSS_HOLD_SECONDS)
                ).isoformat(),
            },
        )
        return CheckResult(passed=False, event=event, fatal=fatal)

    async def _save(self, ctx: Context, key: str, mark: DropMark, ttl: int) -> None:
        # A dry run shares Redis with the live validator, so it leaves the live hold alone.
        if settings.DRY_RUN:
            return
        try:
            await ctx.services.redis.set(key, mark.dump(), ex=ttl)
        except GPU_DROP_REDIS_ERRORS:
            self._log_redis_unavailable(ctx, "write")

    async def _forget(self, ctx: Context, key: str) -> None:
        if settings.DRY_RUN:
            return
        try:
            await ctx.services.redis.delete(key)
        except GPU_DROP_REDIS_ERRORS:
            self._log_redis_unavailable(ctx, "delete")

    @staticmethod
    def _log_redis_unavailable(ctx: Context, on: str) -> None:
        logger.warning(
            _m(
                "GPU_LOSS_HOLD_REDIS_UNAVAILABLE",
                extra=get_extra_info({**ctx.default_extra, "on": on}),
            ),
            exc_info=True,
        )
