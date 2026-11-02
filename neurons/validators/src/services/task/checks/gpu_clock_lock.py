"""DAH-4663: a renter's GPU clock lock left behind on an idle node is reset here or takes the node out of listing.

A lock record (``gpu_clock_lock:<uuid>``) never expires: it goes only once an ``-rgc`` succeeded, and while it
stays every create on that GPU is refused (docker_service). Without this check a GPU whose reset keeps failing
would stay listed but unrentable. On a node that is not rented (after TenantEnforcementCheck), a record that
names no live pod of this executor and is older than the backend snapshot's lag is retried with ``-rgc``; a GPU
still locked after that fails the node (fatal, score 0) until a later cycle's reset succeeds.
"""

from __future__ import annotations

import json
import logging
import time

from services.gpu_clocks import CLOCK_LOCK_KEY_PREFIX, reset_leftover_clock_locks

from core.utils import _m, get_extra_info

from ..messages import GpuClockLockMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context

logger = logging.getLogger(__name__)

# A pod rented since the cycle's backend snapshot may already hold a lock; same lag as the filler power caps.
STALE_LOCK_GRACE_SECONDS = 30 * 60
# Stable log line for a Loki alert (as nvidia_devices.py's kernel_gpu_verdict lines).
STUCK_LOCK_LOG_MSG = "gpu_clock_lock: stuck lock record, GPU unrentable until its clock reset succeeds"


class GpuClockLockCheck:
    check_id = "gpu.validate.clock_lock"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        uuids = [d["uuid"] for d in ctx.state.gpu_details or [] if d.get("uuid")]
        rented = ctx.state.rented_data.executors.get(ctx.executor.uuid) if ctx.state.rented_data else None
        live_pods = {pod.pod_id for pod in rented.pods} if rented else set()
        stale: list[str] = []
        try:
            for uuid in uuids:
                raw = await ctx.services.redis.get(f"{CLOCK_LOCK_KEY_PREFIX}{uuid}")
                if not isinstance(raw, (str, bytes)):
                    continue
                record = json.loads(raw)
                if record.get("pod_id") not in live_pods and time.time() - record.get("locked_at", 0) >= STALE_LOCK_GRACE_SECONDS:
                    stale.append(uuid)
        except Exception:
            stale = []  # Redis down: the create path still refuses a recorded GPU; judged next cycle
        if not stale:
            return CheckResult(passed=True, event=render_message(Msg.OK, ctx=ctx, check_id=self.check_id))

        still_locked = await reset_leftover_clock_locks(ctx.runner, ctx.services.redis, stale)
        what = {"stale_gpu_uuids": stale, "still_locked_gpu_uuids": still_locked}
        if not still_locked:
            return CheckResult(passed=True, event=render_message(Msg.RESET, ctx=ctx, check_id=self.check_id, what=what))
        logger.error(_m(STUCK_LOCK_LOG_MSG, extra=get_extra_info({**ctx.default_extra, **what})))
        return CheckResult(
            passed=False,
            event=render_message(Msg.STUCK, ctx=ctx, check_id=self.check_id, what=what),
            updates={"score": 0.0, "job_score": 0.0, "score_warning": " WARNING: GPU clocks could not be reset"},
        )
