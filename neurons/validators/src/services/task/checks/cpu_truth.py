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


class CpuTruthCheck:
    """Corroborate the advertised CPU(s) count against sources the lscpu wrapper does not author.

    The scrape reports `CPU(s):` from a `lscpu` a userspace wrapper can rewrite. This check reads,
    over ``ctx.ssh``, the kernel-present CPU population (`/sys/devices/system/cpu/present`, the same
    population lscpu counts), the online logical CPUs (`/proc/cpuinfo`), and docker's NCPU. The
    present count is the like-with-like comparison; `/proc/cpuinfo` and docker NCPU resolve through
    the host too (bind-mount / PATH-wrap reachable) so they are recorded as corroborating only.

    Non-fatal and observe-only: shadow always passes, and only CPU_TRUTH_ENFORCEMENT_ENABLED lets a
    genuine mismatch fail. Never fails on an inability to measure (SSH error, unreadable population,
    missing advertised count) — those return passed=True with a distinct *_unknown reason.
    """

    check_id = "host.validate.cpu_truth"
    fatal = False

    async def run(self, ctx: Context) -> CheckResult:
        # compare advertised CPU(s) against the kernel-present population and docker NCPU over ssh
        if not settings.CPU_TRUTH_CHECK_ENABLED:
            event = render_message(Msg.SKIPPED, ctx=ctx, check_id=self.check_id)
            return CheckResult(passed=True, event=event)

        advertised = self._advertised_count(ctx)
        if advertised is None:
            return self._unknown(ctx, reason="advertised CPU count missing from specs")

        try:
            online = await self._read_online(ctx.ssh)
            present = await self._read_present(ctx.ssh)
            ncpu = await self._read_ncpu(ctx.ssh)
        except (asyncssh.Error, OSError) as exc:
            return self._unknown(ctx, reason="ssh transport error", details={"error": repr(exc)})

        if present is None:
            return self._unknown(
                ctx,
                reason="kernel-present CPU population unreadable",
                details={"advertised": advertised, "online": online, "ncpu": ncpu},
            )

        what = {
            "executor_uuid": ctx.executor.uuid,
            "advertised": advertised,
            "online": online,
            "present": present,
            "ncpu": ncpu,
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
            return CheckResult(passed=not enforce, event=event)

        event = render_message(Msg.CPU_OK, ctx=ctx, check_id=self.check_id, what=what)
        return CheckResult(passed=True, event=event)

    def _advertised_count(self, ctx: Context) -> int | None:
        cpu = (ctx.state.specs or {}).get("cpu") or {}
        try:
            count = int(cpu.get("count"))
        except (TypeError, ValueError):
            return None
        return count if count > 0 else None

    async def _read_online(self, ssh) -> int | None:
        # online logical CPUs — corroborating, resolves through the host like the wrapper does
        return _parse_int(await ssh.run("grep -c ^processor /proc/cpuinfo"))

    async def _read_present(self, ssh) -> int | None:
        # kernel-present CPU population — the authoritative like-with-like comparison
        res = await ssh.run("cat /sys/devices/system/cpu/present")
        if getattr(res, "exit_status", 1) != 0:
            return None
        return _count_cpu_list((getattr(res, "stdout", "") or "").strip())

    async def _read_ncpu(self, ssh) -> int | None:
        # docker's view of host CPUs by ABSOLUTE path (matches the scrape's docker invocation)
        return _parse_int(await ssh.run("/usr/bin/docker info --format '{{.NCPU}}'"))

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


def _parse_int(res) -> int | None:
    if getattr(res, "exit_status", 1) != 0:
        return None
    try:
        return int((getattr(res, "stdout", "") or "").strip())
    except (TypeError, ValueError):
        return None


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
