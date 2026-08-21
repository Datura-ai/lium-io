from __future__ import annotations

import asyncssh

from core.config import settings

from ..messages import CpuTruthMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context

# Advertised cores may exceed the kernel-present population by this many before it is called a spoof.
# `present` (== the population lscpu counts in `CPU(s):`) and the advertised count are the SAME
# population, so an honest host — even one with cores offlined — matches exactly; this slack only
# absorbs a benign hotplug/measurement race. Observed spoofs advertise 4-8x their real cores (176
# vs 44), far outside any slack, so the margin never masks a real lie. The predicate is provisional:
# the emitted RAW counts let ops recalibrate from a week of fleet data before flipping enforcement.
CPU_TRUTH_MARGIN_CORES = 4
# Reading one sysfs file is instant on a live host; without a bound a wedged host would hold the
# check open for the executor's whole validation budget. A timeout lands in the *_unknown path
# (asyncssh raises the builtin TimeoutError, itself an OSError), so the host is never failed for it.
CPU_TRUTH_READ_TIMEOUT_SECONDS = 15


class CpuTruthCheck:
    """Compare the advertised CPU(s) count against the kernel-present population.

    The scrape reports `CPU(s):` from a `lscpu` a userspace wrapper can rewrite. This check reads,
    over ``ctx.ssh``, the kernel-present CPU population (`/sys/devices/system/cpu/present`, the same
    population lscpu counts) — one SSH read, the like-with-like comparison.

    Non-fatal and observe-only: shadow always passes, and only CPU_TRUTH_ENFORCEMENT_ENABLED lets a
    genuine mismatch fail and zero the score. Never fails on an inability to measure (SSH error, unreadable population,
    missing advertised count) — those return passed=True with a distinct *_unknown reason.
    """

    check_id = "host.validate.cpu_truth"
    fatal = False

    async def run(self, ctx: Context) -> CheckResult:
        # compare advertised CPU(s) against the kernel-present population over ssh
        if not settings.CPU_TRUTH_CHECK_ENABLED:
            event = render_message(Msg.SKIPPED, ctx=ctx, check_id=self.check_id)
            return CheckResult(passed=True, event=event)

        advertised = advertised_cpu_count(ctx)
        if advertised is None:
            return self._unknown(ctx, reason="advertised CPU count missing from specs")

        try:
            present = await self._read_present(ctx.ssh)
        except (asyncssh.Error, OSError) as exc:
            return self._unknown(ctx, reason="ssh transport error", details={"error": repr(exc)})

        if present is None:
            return self._unknown(
                ctx,
                reason="kernel-present CPU population unreadable",
                details={"advertised": advertised},
            )

        what = {
            "executor_uuid": ctx.executor.uuid,
            "advertised": advertised,
            "present": present,
            "margin_cores": CPU_TRUTH_MARGIN_CORES,
        }
        enforce = settings.CPU_TRUTH_ENFORCEMENT_ENABLED
        if advertised - present > CPU_TRUTH_MARGIN_CORES:
            event = render_message(
                Msg.CPU_MISMATCH,
                ctx=ctx,
                check_id=self.check_id,
                severity=None if enforce else "warning",
                impact=None if enforce else "Shadow observation only: score was NOT changed",
                what=what,
            )
            # The check is non-fatal, so passed=False alone changes nothing in the pipeline, and
            # zeroing `score` here would be overwritten by ScoreCheck further down. The verdict
            # travels as a context flag that calculate_scores gates on, like the TDX one.
            # DAH-2742: no clear_verified_job_info. The kernel-present read can race a
            # measurement window inside CPU_TRUTH_MARGIN_CORES, so an instant flip on a benign
            # race is not warranted; a real understatement fails every cycle.
            updates = {"cpu_truth_passed": False} if enforce else {}
            return CheckResult(passed=not enforce, event=event, updates=updates)

        event = render_message(Msg.CPU_OK, ctx=ctx, check_id=self.check_id, what=what)
        return CheckResult(passed=True, event=event)

    async def _read_present(self, ssh) -> int | None:
        # kernel-present CPU population — the authoritative like-with-like comparison
        res = await ssh.run(
            "cat /sys/devices/system/cpu/present", timeout=CPU_TRUTH_READ_TIMEOUT_SECONDS
        )
        if getattr(res, "exit_status", 1) != 0:
            return None
        return _count_cpu_list((getattr(res, "stdout", "") or "").strip())

    def _unknown(
        self, ctx: Context, *, reason: str, details: dict | None = None
    ) -> CheckResult:
        event = render_message(
            Msg.CPU_UNKNOWN,
            ctx=ctx,
            check_id=self.check_id,
            what={"executor_uuid": ctx.executor.uuid, "reason": reason, **(details or {})},
        )
        return CheckResult(passed=True, event=event)


def advertised_cpu_count(ctx: Context) -> int | None:
    # the core count the miner reported this cycle, or None when the scrape produced no usable value
    cpu = (ctx.state.specs or {}).get("cpu") or {}
    try:
        count = int(cpu.get("count"))
    except (TypeError, ValueError):
        return None
    return count if count > 0 else None


def _count_cpu_list(spec: str) -> int | None:
    # count entries in a Linux CPU-list string, e.g. "0-127" -> 128, "0-19,40-59" -> 40, "0" -> 1
    if not spec:
        return None
    total = 0
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            low, _, high = part.partition("-")
            try:
                total += int(high) - int(low) + 1
            except ValueError:
                return None
        else:
            try:
                int(part)
            except ValueError:
                return None
            total += 1
    return total or None
