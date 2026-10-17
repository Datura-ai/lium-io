from __future__ import annotations

import shlex
from typing import Any

from core.config import settings

from ..messages import GpuMemoryHealthMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context
from .banned_provider import has_live_rental
from .gpu_fault_probe import MAX_FAULT_LINES, OUTPUT_TAIL_CHARS, PROBE_SOURCE, _cap, _parse_report

# one NVML snapshot in a fork with its own 10 s deadline (miner_jobs/gpu_fault_probe.py), plus SSH
TIMEOUT_SECONDS = 30


class GpuMemoryHealthCheck:
    """Flag a card whose memory NVML reports broken (DAH-3110): uncorrected ECC errors since the
    driver loaded, a pending or failed row remap, a required recovery action (what `nvidia-smi -q
    -d ECC,ROW_REMAPPER` shows). The fault probe sees these only as changes across its own run, and
    only on an idle node; this is one NVML read and no CUDA, so it runs before the rented
    short-circuit, for rented and idle executors alike.

    A flagged card fails the node while GPU_MEMORY_HEALTH_ENFORCEMENT_ENABLED (on by default) like
    a GPU ban: verified job info is cleared, so it is not rented again until the host resets or
    replaces the card, and the failure is fatal only when no renter's pod is running (a live rental
    keeps being verified and billed). A node NVML could not read passes.
    """

    check_id = "gpu.validate.memory_health"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        command = f"{shlex.quote(ctx.executor.python_path)} -I - --nvml-only"
        run = await ctx.runner.run(
            command, timeout=TIMEOUT_SECONDS, retryable=False, stdin_text=PROBE_SOURCE
        )
        report = _parse_report(run.stdout) or {}
        what: dict[str, Any] = {"executor_uuid": ctx.executor.uuid}
        status = report.get("status")
        if status == "flagged":
            flags = _cap(report.get("flags"))
            what["flags"] = flags[:MAX_FAULT_LINES] if isinstance(flags, list) else []
            if settings.GPU_MEMORY_HEALTH_ENFORCEMENT_ENABLED:
                event = render_message(Msg.FAILED, ctx=ctx, check_id=self.check_id, what=what)
                return CheckResult(
                    passed=False,
                    event=event,
                    updates={"clear_verified_job_info": True},
                    fatal=not has_live_rental(ctx),
                )
            message = Msg.FLAGGED
        elif status == "ok":
            message = Msg.OK
        else:
            # no NVML binding, NVML that would not start, SSH: not a reading of the cards
            nvml = report.get("nvml") if isinstance(report.get("nvml"), dict) else {}
            what["error"] = _cap(nvml.get("error")) or run.error_message or "no report in output"
            what["stderr_tail"] = run.stderr[-OUTPUT_TAIL_CHARS:]
            message = Msg.UNKNOWN
        event = render_message(message, ctx=ctx, check_id=self.check_id, what=what)
        return CheckResult(passed=True, event=event)
