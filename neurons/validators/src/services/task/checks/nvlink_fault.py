from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

from core.config import settings

from ..messages import NvlinkFaultMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context
from .capability import get_filler_only_container

# Stdlib-only, shipped over stdin like the pinned-memory probe: nothing to upload or update.
PROBE_SOURCE_PATH = Path(__file__).resolve().parents[3] / "miner_jobs" / "nvlink_multicast_probe.py"
PROBE_SOURCE = PROBE_SOURCE_PATH.read_text()
PROBE_JSON_MARKER = "NVLINK_MULTICAST_PROBE_JSON:"
# above the probe's own 45 s budget; only catches a stalled SSH channel (scored UNKNOWN)
PROBE_TIMEOUT_SECONDS = 75
MAX_CHARS = 300


class NvlinkFaultCheck:
    """Fail an all-NVLink node whose GPUs cannot bind NVLink multicast memory (sticky NVLink fault).

    An 8x B300 (23 Oct) took an uncorrectable NVLink error; afterwards every link still read active and
    every single-GPU probe passed, but NCCL could not bind NVLS multicast memory on any GPU, so it was
    listed and unusable for multi-GPU work. The probe (miner_jobs/nvlink_multicast_probe.py) does that
    bind. Only boards the scrape reports as NVLink between every GPU pair are probed; a GPU without
    multicast support, a driver without the API, or any error other than the NVLink-fault codes passes
    with NVLINK_UNKNOWN. Runs after the rented halt, so a rented node is never probed or failed here.
    A fault fails the node only with NVLINK_FAULT_ENFORCEMENT_ENABLED; off, it is logged and the node passes.
    """

    check_id = "gpu.validate.nvlink_fault"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        interconnect = (ctx.state.specs or {}).get("interconnect")
        if not (isinstance(interconnect, dict) and interconnect.get("nvlink") is True):
            return CheckResult(
                passed=True, event=render_message(Msg.SKIPPED, ctx=ctx, check_id=self.check_id)
            )
        # the probe opens a context on every GPU beside the filler's own NCCL, like the fault probe it skips
        filler_container = get_filler_only_container(ctx)
        if filler_container:
            return CheckResult(
                passed=True,
                event=render_message(
                    Msg.FILLER_SKIPPED,
                    ctx=ctx,
                    check_id=self.check_id,
                    what={"filler_container": filler_container},
                ),
            )

        run = await ctx.runner.run(
            f"{shlex.quote(ctx.executor.python_path)} -I -",
            timeout=PROBE_TIMEOUT_SECONDS,
            retryable=False,
            stdin_text=PROBE_SOURCE,
        )
        report = _parse_report(run.stdout)
        what: dict[str, Any] = {"duration_ms": run.duration_ms, "exit_code": run.exit_code}
        if report is None:
            what["error"] = (run.error_message or "no probe report in output")[:MAX_CHARS]
            what["stderr_tail"] = run.stderr[-MAX_CHARS:]
            return CheckResult(
                passed=True,
                event=render_message(Msg.UNKNOWN, ctx=ctx, check_id=self.check_id, what=what),
            )
        # the report is executor-controlled: copy only bounded scalars
        what["probe"] = {
            key: (value[:MAX_CHARS] if isinstance(value, str) else value)
            for key, value in report.items()
            if key in ("status", "error", "gpus", "size_mb")
            and isinstance(value, (str, int, float))
        }
        status = report.get("status")
        if status == "fault":
            enforce = settings.NVLINK_FAULT_ENFORCEMENT_ENABLED
            event = render_message(
                Msg.PROBE_FAILED,
                ctx=ctx,
                check_id=self.check_id,
                severity=None if enforce else "warning",
                impact=None if enforce else "Shadow observation only: score was NOT changed",
                what=what,
            )
            return CheckResult(passed=not enforce, event=event)
        template = Msg.PROBE_OK if status == "ok" else Msg.UNKNOWN
        return CheckResult(
            passed=True, event=render_message(template, ctx=ctx, check_id=self.check_id, what=what)
        )


def _parse_report(stdout: str) -> dict[str, Any] | None:
    # the last marker line wins; stdout is executor-controlled
    for line in reversed(stdout.splitlines()):
        if line.startswith(PROBE_JSON_MARKER):
            try:
                parsed = json.loads(line[len(PROBE_JSON_MARKER) :])
            except (ValueError, TypeError):
                return None
            return parsed if isinstance(parsed, dict) else None
    return None
