from __future__ import annotations

import json
import shlex
from dataclasses import replace
from pathlib import Path
from typing import Any

from core.config import settings

from ..messages import PinnedMemoryProbeMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context
from .capability import get_filler_only_container, lium_workload_live_now

# Stdlib-only, shipped over stdin like the kernel-fault probe (DAH-3035): nothing to upload or update.
PROBE_SOURCE_PATH = Path(__file__).resolve().parents[3] / "miner_jobs" / "pinned_memory_probe.py"
PROBE_SOURCE = PROBE_SOURCE_PATH.read_text()
PROBE_JSON_MARKER = "PINNED_MEMORY_PROBE_JSON:"
# The probe bounds itself: 20 s to count the devices, `nvidia-smi conf-compute` capped at 10 s, one 30 s
# budget for every card's setup and copy together. This cap sits above that sum and only catches an
# interpreter that never printed or an SSH channel that stalled (scored UNKNOWN, not a fault).
PROBE_TIMEOUT_SECONDS = 90
OUTPUT_TAIL_CHARS = 800
# the report is executor-controlled: bound what is copied into the event
MAX_FAULT_LINES = 16
MAX_FAULT_CHARS = 300
MAX_GPUS = 16
# what the probe reports for a card whose copy never finished inside the budget
HUNG_COPY_FAULT_SUFFIX = ": worker hung during copy"


class PinnedMemoryProbeCheck:
    """Round-trip a pinned host buffer through every GPU and fail the node if the data does not come back (F-1599).

    A renter of an 8x H200 node (13 Oct) found the GPUs could not read pinned host memory: the executor
    ran in a TDX guest whose GPUs were not in confidential-computing mode, so DMA into page-locked guest
    memory returned garbage. The matmul and every other check passed; PyTorch's pin_memory()
    + non_blocking copies, which every DataLoader uses, did not. The probe (miner_jobs/pinned_memory_probe.py)
    allocates 64 MB with cuMemAllocHost on each card, copies it H2D and back through the async copy engines and
    compares it on the host; it also reports the cpuinfo tdx_guest flag and `nvidia-smi conf-compute -f` as evidence.

    PINNED_MEMORY_PROBE_CHECK_ENABLED (default on, a few-ms copy per card) runs it; a fault fails the (fatal)
    check, which keeps the node unrentable. Wrong data, a copy error, a copy that hangs or a
    driver error naming a broken GPU (BROKEN_GPU_CODES) counts as a fault. A probe that could not run (no libcuda, a
    context or allocation refused, a memlock limit) passes with PINNED_MEMORY_PROBE_UNKNOWN.

    A Lium filler holds the cards, so the probe skips a filler-only node the way the kernel-fault probe does,
    and a copy that only hung is re-checked against the backend's live workload the way CapabilityCheck does:
    on 19 Oct a copy hung on an honest 8x B200 beside our own filler.
    """

    check_id = "gpu.validate.pinned_memory"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        if not settings.PINNED_MEMORY_PROBE_CHECK_ENABLED:
            return CheckResult(
                passed=True, event=render_message(Msg.DISABLED, ctx=ctx, check_id=self.check_id)
            )

        filler_container = get_filler_only_container(ctx)
        if filler_container:
            event = render_message(
                Msg.FILLER_SKIPPED,
                ctx=ctx,
                check_id=self.check_id,
                what={"filler_container": filler_container},
            )
            return CheckResult(passed=True, event=event)

        command = f"{shlex.quote(ctx.executor.python_path)} -I -"
        run = await ctx.runner.run(
            command, timeout=PROBE_TIMEOUT_SECONDS, retryable=False, stdin_text=PROBE_SOURCE
        )
        report = _parse_report(run.stdout)
        what: dict[str, Any] = {
            "executor_uuid": ctx.executor.uuid,
            "duration_ms": run.duration_ms,
            "exit_code": run.exit_code,
        }

        if run.error_type == "timeout" or report is None:
            # no verdict came back: a stalled channel or an interpreter that could not run the probe.
            # A card that hangs its copy is reported by the probe itself, so this is not a GPU fault.
            what.update(
                error=(
                    f"probe did not finish in {PROBE_TIMEOUT_SECONDS}s (SSH timeout, no verdict)"
                    if run.error_type == "timeout"
                    else run.error_message or "no probe report in output"
                ),
                stdout_tail=run.stdout[-OUTPUT_TAIL_CHARS:],
                stderr_tail=run.stderr[-OUTPUT_TAIL_CHARS:],
            )
            return CheckResult(
                passed=True,
                event=render_message(Msg.UNKNOWN, ctx=ctx, check_id=self.check_id, what=what),
            )

        what["probe"] = _summary(report)
        status = report.get("status")
        if status == "ok":
            return CheckResult(
                passed=True,
                event=render_message(Msg.PROBE_OK, ctx=ctx, check_id=self.check_id, what=what),
            )
        if status == "fault":
            faults = _cap(report.get("faults"))
            error = (
                "; ".join(str(fault) for fault in faults if fault is not None)
                if isinstance(faults, list)
                else ""
            )
            what["error"] = (error or "fault")[: MAX_FAULT_LINES * MAX_FAULT_CHARS]
            # wrong data is never waived; only a copy that hung may have lost the card to a Lium workload
            if _only_hung_copies(faults):
                lium_workload = await lium_workload_live_now(ctx)
                if lium_workload is not None:
                    event = render_message(
                        Msg.WORKLOAD_SKIPPED,
                        ctx=ctx,
                        check_id=self.check_id,
                        what={
                            **what,
                            "workload": lium_workload.kind,
                            "containers": list(lium_workload.container_names),
                        },
                    )
                    return CheckResult(
                        passed=True,
                        event=event,
                        updates={"state": replace(ctx.state, rented_data=lium_workload.snapshot)},
                    )
            event = render_message(Msg.PROBE_FAILED, ctx=ctx, check_id=self.check_id, what=what)
            return CheckResult(passed=False, event=event)
        what["error"] = _cap(report.get("error")) or f"probe status {_cap(status)!r}"
        return CheckResult(
            passed=True,
            event=render_message(Msg.UNKNOWN, ctx=ctx, check_id=self.check_id, what=what),
        )


def _parse_report(stdout: str) -> dict[str, Any] | None:
    # the last marker line wins; the probe prints exactly one, but stdout is executor-controlled
    for line in reversed(stdout.splitlines()):
        if line.startswith(PROBE_JSON_MARKER):
            try:
                parsed = json.loads(line[len(PROBE_JSON_MARKER) :])
            except (ValueError, TypeError):
                return None
            return parsed if isinstance(parsed, dict) else None
    return None


def _only_hung_copies(faults: Any) -> bool:
    return (
        isinstance(faults, list)
        and bool(faults)
        and all(isinstance(fault, str) and fault.endswith(HUNG_COPY_FAULT_SUFFIX) for fault in faults)
    )


def _cap(value: Any) -> Any:
    """Bound anything copied out of the executor-written report: strings cut, lists cut and capped
    element-wise, numbers and None kept, dicts and everything else dropped."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:MAX_FAULT_CHARS]
    if isinstance(value, list):
        return [
            _cap(item) if isinstance(item, (str, bool, int, float)) or item is None else None
            for item in value[:MAX_FAULT_LINES]
        ]
    return None


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    devices = []
    raw_devices = report.get("devices")
    for device in raw_devices[:MAX_GPUS] if isinstance(raw_devices, list) else []:
        if isinstance(device, dict):
            devices.append(
                {
                    key: _cap(device.get(key))
                    for key in ("index", "name", "status", "error", "size_mb", "copy_ms")
                    if key in device
                }
            )
    tee = report.get("tee")
    return {
        "status": _cap(report.get("status")),
        "elapsed_s": _cap(report.get("elapsed_s")),
        "devices": devices,
        "faults": _cap(report.get("faults")),
        "tee": (
            {
                key: _cap(tee.get(key))
                for key in (
                    "dev_tdx_guest",
                    "cpuinfo_tdx_guest",
                    "cpuinfo_error",
                    "gpu_cc_mode",
                    "gpu_cc_mode_error",
                )
                if key in tee
            }
            if isinstance(tee, dict)
            else None
        ),
    }
