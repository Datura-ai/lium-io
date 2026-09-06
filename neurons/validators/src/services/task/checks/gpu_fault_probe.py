from __future__ import annotations

import json
import shlex
from pathlib import Path

from core.config import settings

from ..messages import GpuFaultProbeMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context
from .capability import _get_filler_only_container

# The probe runs on the executor's own interpreter, standard library only, and ships over stdin the way
# the scrape source does (DAH-2794): nothing to upload, nothing the miner has to update.
PROBE_SOURCE_PATH = Path(__file__).resolve().parents[3] / "miner_jobs" / "gpu_fault_probe.py"
PROBE_SOURCE = PROBE_SOURCE_PATH.read_text()
PROBE_JSON_MARKER = "GPU_FAULT_PROBE_JSON:"
# Measured on a healthy 1x RTX 4090 (driver 580): 5.3 s in-script, 6.6 s over SSH, 7 rounds on a 2 GB
# working set. The probe's own per-GPU deadline is --seconds + 30 s; this cap only catches an interpreter
# that never printed — a wedged card that hangs the driver is a fault, not a slow host.
PROBE_TIMEOUT_SECONDS = 90
PROBE_SECONDS = 4
OUTPUT_TAIL_CHARS = 800


class GpuFaultProbeCheck:
    """Run the kernel-fault probe on the executor's GPUs after the matmul has passed (DAH-3035).

    A cuBLAS matmul reads and writes memory sequentially and verifies one number; a card whose memory
    subsystem faults under indexed access renders nothing (Blender: "Illegal address in CUDA queue" on
    both OptiX and CUDA, 6 Sep) yet passes it and stays listed. The probe (miner_jobs/gpu_fault_probe.py)
    gathers, scatters, atomics and pointer-chases through a random permutation of a ~2 GB working set,
    round-trips pinned memory through the copy engines, verifies every result on the device, and reads
    uncorrected ECC / remapped rows / recovery action from NVML before and after.

    Shadow-first like every score-zeroing gate: GPU_FAULT_PROBE_CHECK_ENABLED runs it and emits the verdict,
    GPU_FAULT_PROBE_ENFORCEMENT_ENABLED lets a fault fail the (fatal) check. A probe that could not run
    (no libcuda, cuInit, JIT) passes with GPU_FAULT_PROBE_UNKNOWN: inability to measure is not a fault.
    """

    check_id = "gpu.validate.fault_probe"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        if not settings.GPU_FAULT_PROBE_CHECK_ENABLED:
            return CheckResult(
                passed=True, event=render_message(Msg.DISABLED, ctx=ctx, check_id=self.check_id)
            )

        filler_container = _get_filler_only_container(ctx)
        if filler_container:
            event = render_message(
                Msg.FILLER_SKIPPED,
                ctx=ctx,
                check_id=self.check_id,
                what={"filler_container": filler_container},
            )
            return CheckResult(passed=True, event=event)

        command = f"{shlex.quote(ctx.executor.python_path)} -I - --seconds {PROBE_SECONDS}"
        run = await ctx.runner.run(
            command, timeout=PROBE_TIMEOUT_SECONDS, retryable=False, stdin_text=PROBE_SOURCE
        )
        report = _parse_report(run.stdout)
        what = {
            "executor_uuid": ctx.executor.uuid,
            "duration_ms": run.duration_ms,
            "exit_code": run.exit_code,
        }

        if run.error_type == "timeout":
            return self._fault(
                ctx, what, error=f"probe did not finish in {PROBE_TIMEOUT_SECONDS}s", report=None
            )
        if report is None:
            # the interpreter never printed a verdict: an SSH error, a python that could not run stdlib
            # code, or a crash in the parent process. Not something the GPU is blamed for.
            what.update(
                error=run.error_message or "no probe report in output",
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
            return self._fault(
                ctx, what, error="; ".join(report.get("faults") or []) or "fault", report=report
            )
        what["error"] = report.get("error") or f"probe status {status!r}"
        return CheckResult(
            passed=True,
            event=render_message(Msg.UNKNOWN, ctx=ctx, check_id=self.check_id, what=what),
        )

    def _fault(self, ctx: Context, what: dict, *, error: str, report: dict | None) -> CheckResult:
        enforce = settings.GPU_FAULT_PROBE_ENFORCEMENT_ENABLED
        what["error"] = error
        if report is not None:
            what["xid"] = report.get("xid")
        event = render_message(
            Msg.PROBE_FAILED,
            ctx=ctx,
            check_id=self.check_id,
            severity=None if enforce else "warning",
            impact=None if enforce else "Shadow observation only: score was NOT changed",
            what=what,
        )
        return CheckResult(passed=not enforce, event=event)


def _parse_report(stdout: str) -> dict | None:
    # the last marker line wins; the probe prints exactly one, but stdout is executor-controlled
    for line in reversed(stdout.splitlines()):
        if line.startswith(PROBE_JSON_MARKER):
            try:
                parsed = json.loads(line[len(PROBE_JSON_MARKER) :])
            except (ValueError, TypeError):
                return None
            return parsed if isinstance(parsed, dict) else None
    return None


def _summary(report: dict) -> dict:
    # what the operator needs from the report without the per-round noise: verdict, timing, devices, NVML deltas
    devices = []
    for device in report.get("devices") or []:
        if not isinstance(device, dict):
            continue
        devices.append(
            {
                key: device.get(key)
                for key in (
                    "index",
                    "name",
                    "status",
                    "error",
                    "rounds",
                    "work_s",
                    "elapsed_s",
                    "working_set_mb",
                    "jit_ms",
                )
                if key in device
            }
        )
    return {
        "status": report.get("status"),
        "elapsed_s": report.get("elapsed_s"),
        "devices": devices,
        "faults": report.get("faults"),
        "nvml_after": report.get("nvml_after"),
    }
