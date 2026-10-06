from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

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
# working set. A hung card is caught INSIDE the probe: the workers share one wall-clock budget (--seconds
# + 30 s + 5 s per extra GPU) and are drained together, so every card gets the whole budget and a verdict —
# a hang in setup is an error (scored UNKNOWN), a hang in the kernels is a fault. This cap sits above the
# largest budget (8 GPUs: 4 + 65 s) plus the parent's bounded tail after the drain (one reap grace, two NVML
# snapshots in a fork with their own deadline, two dmesg reads: 47 s when everything hangs at once) —
# test_the_ssh_cap_sits_above_the_probes_own_largest_deadline adds it up — and only catches
# an interpreter that never printed or an SSH channel that stalled;
# the runner sets error_type "timeout" for any asyncio.TimeoutError around ssh.run, so that path cannot
# tell a transport stall from a hung host and is scored UNKNOWN, like the other no-report outcomes.
PROBE_TIMEOUT_SECONDS = 150
PROBE_SECONDS = 4
OUTPUT_TAIL_CHARS = 800
# the report is executor-controlled: bound what is copied into the event
MAX_FAULT_LINES = 16
MAX_FAULT_CHARS = 300
MAX_NVML_GPUS = 16
MAX_XID_LINES = 5
MAX_XID_CHARS = 200


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
        what: dict[str, Any] = {
            "executor_uuid": ctx.executor.uuid,
            "duration_ms": run.duration_ms,
            "exit_code": run.exit_code,
        }

        if run.error_type == "timeout":
            # no verdict came back: the SSH channel stalled or the interpreter hung outside its workers.
            # A card that hangs its worker is reported by the probe itself, so this is not a GPU fault.
            what.update(
                error=f"probe did not finish in {PROBE_TIMEOUT_SECONDS}s (SSH timeout, no verdict)",
                stdout_tail=run.stdout[-OUTPUT_TAIL_CHARS:],
                stderr_tail=run.stderr[-OUTPUT_TAIL_CHARS:],
            )
            return CheckResult(
                passed=True,
                event=render_message(Msg.UNKNOWN, ctx=ctx, check_id=self.check_id, what=what),
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
            faults = _cap(report.get("faults"))
            error = (
                "; ".join(str(fault) for fault in faults if fault is not None)
                if isinstance(faults, list)
                else ""
            )
            return self._fault_verdict(ctx, what, error=error or "fault", report=report)
        what["error"] = _cap(report.get("error")) or f"probe status {_cap(status)!r}"
        return CheckResult(
            passed=True,
            event=render_message(Msg.UNKNOWN, ctx=ctx, check_id=self.check_id, what=what),
        )

    def _fault_verdict(
        self, ctx: Context, what: dict[str, Any], *, error: str, report: dict[str, Any] | None
    ) -> CheckResult:
        enforce = settings.GPU_FAULT_PROBE_ENFORCEMENT_ENABLED
        what["error"] = error[: MAX_FAULT_LINES * MAX_FAULT_CHARS]
        if report is not None:
            what["xid"] = _capped_xid(report.get("xid"))
        event = render_message(
            Msg.PROBE_FAILED,
            ctx=ctx,
            check_id=self.check_id,
            severity=None if enforce else "warning",
            impact=None if enforce else "Shadow observation only: score was NOT changed",
            what=what,
        )
        return CheckResult(passed=not enforce, event=event)


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


def _capped_list(value: Any, limit: int) -> list[Any]:
    return value[:limit] if isinstance(value, list) else []


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    # what the operator needs from the report without the per-round noise: verdict, timing, devices, NVML
    # deltas — every scalar and list bounded by _cap, since the executor wrote the report
    devices = []
    for device in _capped_list(report.get("devices"), MAX_NVML_GPUS):
        if not isinstance(device, dict):
            continue
        devices.append(
            {
                key: _cap(device.get(key))
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
        "status": _cap(report.get("status")),
        "elapsed_s": _cap(report.get("elapsed_s")),
        "devices": devices,
        "faults": _cap(report.get("faults")),
        "nvml_after": _capped_nvml(report.get("nvml_after")),
    }


def _capped_nvml(snapshot: Any) -> dict[str, Any] | None:
    if not isinstance(snapshot, dict):
        return None
    gpus = []
    for gpu in _capped_list(snapshot.get("gpus"), MAX_NVML_GPUS):
        if isinstance(gpu, dict):
            gpus.append(
                {
                    key: _cap(gpu[key])
                    for key in (
                        "index",
                        "uuid",
                        "pci_bus_id",
                        "ecc_uncorrected",
                        "remapped_rows",
                        "recovery_action",
                    )
                    if key in gpu
                }
            )
    capped: dict[str, Any] = {"available": bool(snapshot.get("available")), "gpus": gpus}
    if snapshot.get("error") is not None:
        capped["error"] = str(snapshot["error"])[:MAX_FAULT_CHARS]
    return capped


def _capped_xid(xid: Any) -> dict[str, Any] | None:
    if not isinstance(xid, dict):
        return None
    capped: dict[str, Any] = {"available": bool(xid.get("available"))}
    if isinstance(xid.get("count"), int):
        capped["count"] = xid["count"]
    if isinstance(xid.get("last"), list):
        capped["last"] = [str(line)[:MAX_XID_CHARS] for line in xid["last"][-MAX_XID_LINES:]]
    if isinstance(xid.get("other_new"), list):
        # new Xid lines on other cards (or of application types): evidence, not this executor's fault
        capped["other_new"] = [
            str(line)[:MAX_XID_CHARS] for line in xid["other_new"][-MAX_XID_LINES:]
        ]
    if xid.get("error") is not None:
        capped["error"] = str(xid["error"])[:MAX_FAULT_CHARS]
    return capped
