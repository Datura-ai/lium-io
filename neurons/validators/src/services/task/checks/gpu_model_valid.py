from __future__ import annotations

import json
import logging
import time
from typing import Any

from core.config import settings
from core.utils import _m, get_extra_info

from ..messages import GpuModelMessages as Msg, render_message
from ..models import ValidationEvent
from ..pipeline import CheckResult, Context

logger = logging.getLogger(__name__)

# P125: consecutive cycles a RENTED node reported fewer GPUs than it advertises. One JSON string per
# executor, written only while the fault stands; a clean scrape or an idle cycle deletes it, and the
# key expires on its own after _FAULT_STREAK_TTL_SECONDS so a node this validator stopped scraping
# (deregistered, skipped, validator restarted) does not resume an old count months later.
_REDIS_FAULT_PREFIX = "rented_gpu_fault"
_FAULT_STREAK_TTL_SECONDS = 4 * 15 * 60  # four cycles at the ~15 min cadence


class GpuModelValidCheck:
    """Gate validation on supported GPU SKUs and healthy scrape output.

    This mirrors the legacy guard that rejected unknown models, zero counts, or mismatched
    detail lists. It prevents us from handing out scores when the scrape clearly failed or
    when a miner advertises off-policy hardware.

    P125: on a RENTED node a short GPU list is a card off the bus under a renter's pod (F-1361:
    8 advertised, 1 enumerated, the renter deleted a $1,023 rental himself and the node stayed
    listed). After RENTED_GPU_FAULT_CYCLES such cycles in a row the result is RENTED_NODE_GPU_FAULT
    and the verified job is cleared, which is the path the rental probe (DAH-3436) uses to delist a
    node; the backend also notifies the renter and stamps the rental. Below the threshold, or on an
    idle node, the cycle scores 0 as it always did.
    """

    check_id = "gpu.validate.model"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        gpu_model_rates = ctx.config.gpu_model_rates
        if not gpu_model_rates:
            event = render_message(
                Msg.POLICY_MISSING,
                ctx=ctx,
                check_id=self.check_id,
            )
            return CheckResult(passed=False, event=event)

        gpu_model_rates_map = gpu_model_rates
        specs = ctx.state.specs
        gpu_count = ctx.state.gpu_count
        if gpu_count is None:
            gpu_count = specs.get("gpu", {}).get("count", 0)
        gpu_details = ctx.state.gpu_details
        if not gpu_details:
            gpu_details = specs.get("gpu", {}).get("details", [])

        gpu_model = None
        if gpu_count > 0 and len(gpu_details) > 0:
            gpu_model = gpu_details[0].get("name", None)

        if gpu_model not in gpu_model_rates_map:
            supported_models = list(gpu_model_rates_map.keys())
            event = render_message(
                Msg.MODEL_UNSUPPORTED,
                ctx=ctx,
                check_id=self.check_id,
                what={
                    "gpu_model": gpu_model,
                    "gpu_count": gpu_count,
                    "supported_models": supported_models,
                },
                remediation=(
                    "Use a supported GPU model. Supported models: "
                    f"{', '.join(supported_models[:5])}{'...' if len(supported_models) > 5 else ''}"
                ),
            )
            if gpu_model is None:
                # no card to name a model from (count 0 or an empty list): the same missing-GPU shape
                return await self._rented_fault_or(ctx, event, gpu_count=gpu_count, details_len=len(gpu_details))
            return CheckResult(passed=False, event=event)

        if gpu_count == 0:
            event = render_message(
                Msg.COUNT_ZERO,
                ctx=ctx,
                check_id=self.check_id,
                what={"gpu_count": gpu_count},
            )
            return CheckResult(passed=False, event=event)

        if len(gpu_details) != gpu_count:
            event = render_message(
                Msg.DETAILS_MISMATCH,
                ctx=ctx,
                check_id=self.check_id,
                what={"gpu_count": gpu_count, "details_len": len(gpu_details)},
            )
            return await self._rented_fault_or(ctx, event, gpu_count=gpu_count, details_len=len(gpu_details))

        await _clear_fault(ctx)
        event = render_message(
            Msg.MODEL_OK,
            ctx=ctx,
            check_id=self.check_id,
            what={"gpu_model": gpu_model, "gpu_count": gpu_count},
        )
        return CheckResult(passed=True, event=event)

    async def _rented_fault_or(
        self, ctx: Context, event: ValidationEvent, *, gpu_count: int, details_len: int
    ) -> CheckResult:
        """The plain failure, or RENTED_NODE_GPU_FAULT once a rented node has failed enough cycles in a row."""
        plain = CheckResult(passed=False, event=event)
        pod_ids = _rented_pod_ids(ctx)
        if not pod_ids:
            # an idle node's short GPU list is the ordinary score-0 path; its count must not carry
            # into a later rental
            await _clear_fault(ctx)
            return plain
        if not settings.RENTED_GPU_FAULT_ENABLED:
            return plain

        fault = await _bump_fault(ctx)
        if fault is None or fault["count"] < settings.RENTED_GPU_FAULT_CYCLES:
            return plain

        what = {
            "gpu_count": gpu_count,
            "details_len": details_len,
            "consecutive_cycles": fault["count"],
            "first_seen_at": fault["first_seen_at"],
            "pod_ids": pod_ids,
            "plain_reason_code": event.reason_code,
        }
        fault_event = render_message(Msg.RENTED_NODE_GPU_FAULT, ctx=ctx, check_id=self.check_id, what=what)
        return CheckResult(
            passed=False,
            event=fault_event,
            updates={
                "clear_verified_job_info": True,
                "clear_verified_job_evidence": {
                    "reason_code": fault_event.reason_code,
                    "check_id": self.check_id,
                    **what,
                },
            },
        )


def _rented_pod_ids(ctx: Context) -> list[str]:
    rented_data = ctx.state.rented_data
    if rented_data is None:
        return []
    rented = rented_data.executors.get(ctx.executor.uuid)
    if rented is None:
        return []
    return [pod.pod_id for pod in rented.pods]


def _fault_key(ctx: Context) -> str:
    return f"{_REDIS_FAULT_PREFIX}:{ctx.executor.uuid}"


async def _bump_fault(ctx: Context) -> dict[str, Any] | None:
    """Count this cycle; None when Redis could not be read or written (logged; the cycle then
    reports the plain failure, so a Redis outage never delists a node)."""
    redis = ctx.services.redis
    if redis is None:
        return None
    try:
        raw = await redis.get(_fault_key(ctx))
        fault: dict[str, Any] = {"count": 0, "first_seen_at": time.time()}
        if raw is not None:
            text = raw.decode() if isinstance(raw, bytes) else str(raw)
            try:
                loaded = json.loads(text)
                fault = {
                    "count": int(loaded.get("count", 0)),
                    "first_seen_at": float(loaded.get("first_seen_at", fault["first_seen_at"])),
                }
            except (ValueError, TypeError, AttributeError):
                # an unreadable key starts the count over rather than delisting on a guess
                pass
        fault["count"] += 1
        await redis.set(_fault_key(ctx), json.dumps(fault), ex=_FAULT_STREAK_TTL_SECONDS)
        return fault
    except Exception:
        logger.warning(
            _m(
                "Rented node reported a GPU fault but the streak could not be counted in Redis; reporting the plain failure",
                extra=get_extra_info(ctx.default_extra),
            ),
            exc_info=True,
        )
        return None


async def _clear_fault(ctx: Context) -> None:
    redis = ctx.services.redis
    if redis is None:
        return
    try:
        await redis.delete(_fault_key(ctx))
    except Exception:
        logger.warning(
            _m(
                "GPU fault streak could not be cleared in Redis; the next rented fault may count one cycle high",
                extra=get_extra_info(ctx.default_extra),
            ),
            exc_info=True,
        )
