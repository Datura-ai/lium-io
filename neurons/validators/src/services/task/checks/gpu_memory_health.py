from __future__ import annotations

import re
import shlex
from typing import Any

from core.config import settings

from ..availability import ReachSource, ReachTarget
from ..messages import GpuMemoryHealthMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context, MemoryHealthRead
from .banned_provider import has_live_rental
from .gpu_fault_probe import OUTPUT_TAIL_CHARS, PROBE_SOURCE, _cap, _parse_report
from .post_rental_gpu_recheck import RecheckMarker, record_marker

# one NVML snapshot in a fork with its own 10 s deadline (miner_jobs/gpu_fault_probe.py), plus SSH
TIMEOUT_SECONDS = 30
# a flag names its card as "gpu <index> (<uuid>): ..." (miner_jobs/gpu_fault_probe.py nvml_health)
_FLAG_UUID = re.compile(r"^gpu \S+ \(([^()]+)\): ")


async def read_memory_health(ctx: Context) -> MemoryHealthRead:
    """The NVML-only probe's run and report, read once per cycle: RentedGpuDropCheck reads it first
    on a rented node (and keeps it in `ctx.state.memory_health`), and GpuMemoryHealthCheck reuses it."""
    if ctx.state.memory_health is not None:
        return ctx.state.memory_health
    command = f"{shlex.quote(ctx.executor.python_path)} -I - --nvml-only"
    run = await ctx.runner.run(
        command, timeout=TIMEOUT_SECONDS, retryable=False, stdin_text=PROBE_SOURCE
    )
    return MemoryHealthRead(run=run, report=_parse_report(run.stdout) or {})


def flagged_uuids(report: dict[str, Any]) -> list[str]:
    """The UUIDs of the cards a flagged report names broken."""
    if report.get("status") != "flagged" or not isinstance(report.get("flags"), list):
        return []
    found = (_FLAG_UUID.match(flag) for flag in report["flags"] if isinstance(flag, str))
    return sorted({match.group(1) for match in found if match and match.group(1) != "None"})


class GpuMemoryHealthCheck:
    """Flag a card whose memory NVML reports broken (DAH-3110): uncorrected ECC errors since the
    driver loaded, a failed row remap, or a pending remap / required recovery action on a card with
    uncorrectable remapped rows or uncorrected ECC (what `nvidia-smi -q -d ECC,ROW_REMAPPER` shows);
    a pending remap or recovery action from correctable errors alone is only a warning, sent as one
    so the provider resets the card. The fault probe sees these only as changes across its own run,
    and only on an idle node; this is one NVML read and no CUDA, so it runs before the rented
    short-circuit, for rented and idle executors alike.

    A flagged card fails the node while GPU_MEMORY_HEALTH_ENFORCEMENT_ENABLED (on by default) like
    a GPU ban: verified job info is cleared, so it is not rented again until the host resets or
    replaces the card, and the failure is fatal only when no renter's pod is running (a live rental
    keeps being verified and billed). DAH-4683: the failure is also a GPU_MEMORY_HEALTH_FAILED
    availability entry, so the backend hides the node and refuses new rentals while leaving a rental
    already on it alone: a split-rented node with one live pod otherwise kept its broken free card
    listed. A node NVML could not read passes. A card flagged under a live rental
    also owes a post-rental every-card recheck (post_rental_gpu_recheck.py).
    """

    check_id = "gpu.validate.memory_health"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        read = await read_memory_health(ctx)
        run, report = read.run, read.report
        what: dict[str, Any] = {"executor_uuid": ctx.executor.uuid}
        status = report.get("status")
        if status == "flagged":
            flags = _cap(report.get("flags"))
            what["flags"] = flags if isinstance(flags, list) else []
            if has_live_rental(ctx):
                # DAH-4482: the node owes an every-card recheck on its next unrented cycle
                await record_marker(
                    ctx, RecheckMarker(self.check_id, "flagged", flags=what["flags"])
                )
            if settings.GPU_MEMORY_HEALTH_ENFORCEMENT_ENABLED:
                what["reach_source"] = str(ReachSource.VALIDATOR)
                what["reach_target"] = str(ReachTarget.GPU)
                event = render_message(Msg.FAILED, ctx=ctx, check_id=self.check_id, what=what)
                return CheckResult(
                    passed=False,
                    event=event,
                    updates={"clear_verified_job_info": True},
                    fatal=not has_live_rental(ctx),
                )
            message = Msg.FLAGGED
        elif status == "ok":
            # a pending remap from correctable errors only: the card still works, but needs a reset
            warnings = _cap(report.get("warnings"))
            if isinstance(warnings, list) and warnings:
                what["warnings"] = warnings
                message = Msg.RESET_ADVISED
            else:
                message = Msg.OK
        else:
            # no NVML binding, NVML that would not start, SSH: not a reading of the cards
            nvml = report.get("nvml") if isinstance(report.get("nvml"), dict) else {}
            what["error"] = _cap(nvml.get("error")) or run.error_message or "no report in output"
            what["stderr_tail"] = run.stderr[-OUTPUT_TAIL_CHARS:]
            message = Msg.UNKNOWN
        event = render_message(message, ctx=ctx, check_id=self.check_id, what=what)
        return CheckResult(passed=True, event=event)
