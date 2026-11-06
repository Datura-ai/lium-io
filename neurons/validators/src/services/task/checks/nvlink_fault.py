from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

from services.gpu_spec_table import is_sxm_on_pcie_adapter, normalize_gpu_model

from ..messages import NvlinkFaultMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context

# Stdlib-only, shipped over stdin like the pinned-memory probe: nothing to upload or update.
PROBE_SOURCE_PATH = Path(__file__).resolve().parents[3] / "miner_jobs" / "nvlink_multicast_probe.py"
PROBE_SOURCE = PROBE_SOURCE_PATH.read_text()
PROBE_JSON_MARKER = "NVLINK_MULTICAST_PROBE_JSON:"
# above the probe's own 45 s budget; only catches a stalled SSH channel (scored UNKNOWN)
PROBE_TIMEOUT_SECONDS = 75
MAX_CHARS = 300
# These GPUs ship only as SXM modules on an NVLink baseboard, so a full 8-GPU board of them with no NVLink
# at all has its fabric down (fabric manager or NVSwitch), not a PCIe build. Fewer GPUs can be a VM slice.
NVLINK_BASEBOARD_GPUS = frozenset(
    {
        "NVIDIA H100 80GB HBM3",
        "NVIDIA H800 80GB HBM3",
        "NVIDIA H200",
        "NVIDIA B200",
        "NVIDIA B300 SXM6 AC",
        "NVIDIA B300 SXM6 PC",
        "NVIDIA A100-SXM4-80GB",
    }
)
BASEBOARD_GPU_COUNT = 8


class NvlinkFaultCheck:
    """Fail an all-NVLink node whose GPUs cannot bind NVLink multicast memory (sticky NVLink fault).

    An 8x B300 (23 Oct) took an uncorrectable NVLink error; afterwards every link still read active and
    every single-GPU probe passed, but NCCL could not bind NVLS multicast memory on any GPU, so it was
    listed and unusable for multi-GPU work. The probe (miner_jobs/nvlink_multicast_probe.py) does that
    bind. Only boards the scrape reports as NVLink between every GPU pair are probed; a GPU without
    multicast support, a driver without the API, or any error other than the NVLink-fault codes passes
    with NVLINK_UNKNOWN. Runs after the rented halt, so a rented node is never probed or failed here.
    It runs beside an active filler: on 8x H100 and 8x B300 (28 Oct) the probe passed beside an NVLS
    all-reduce filler, which kept exact results with one slower second per probe. A filler holding
    nearly all GPU memory can leave no room for the probe's context: NVLINK_UNKNOWN, the node passes.

    A full 8x board of SXM-only GPUs that reports no NVLink at all fails with NVLINK_MISSING: an 8x H100
    HBM3 (1 Nov) was listed with PCIe-only topology (fabric down) and a renter found no NVLink.
    H100 HBM3 GPUs with no NVLink at all are not failed: they are listed as H100 PCIe (DAH-5677,
    services.gpu_spec_table.is_sxm_on_pcie_adapter), the SXM-on-PCIe-adapter build renters get.
    """

    check_id = "gpu.validate.nvlink_fault"
    replays_after_session_drop = True
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        specs = ctx.state.specs or {}
        interconnect = specs.get("interconnect")
        if not (isinstance(interconnect, dict) and interconnect.get("nvlink") is True):
            missing_nvlink = _missing_nvlink_on_sxm_board(specs, interconnect)
            if missing_nvlink is not None:
                return CheckResult(
                    passed=False,
                    event=render_message(
                        Msg.MISSING, ctx=ctx, check_id=self.check_id, what=missing_nvlink
                    ),
                )
            return CheckResult(
                passed=True, event=render_message(Msg.SKIPPED, ctx=ctx, check_id=self.check_id)
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
            return CheckResult(
                passed=False,
                event=render_message(Msg.PROBE_FAILED, ctx=ctx, check_id=self.check_id, what=what),
            )
        template = Msg.PROBE_OK if status == "ok" else Msg.UNKNOWN
        return CheckResult(
            passed=True, event=render_message(template, ctx=ctx, check_id=self.check_id, what=what)
        )


def _missing_nvlink_on_sxm_board(specs: dict[str, Any], interconnect: Any) -> dict[str, Any] | None:
    # only an explicit nvlink False counts; an old executor without the interconnect scrape is skipped
    if not (isinstance(interconnect, dict) and interconnect.get("nvlink") is False):
        return None
    if is_sxm_on_pcie_adapter(specs):
        return None
    details = (specs.get("gpu") or {}).get("details") or []
    # specs keep the raw NVML name (e.g. "Tesla H100 80GB HBM3")
    names = {normalize_gpu_model(gpu.get("name")) for gpu in details if isinstance(gpu, dict)}
    if len(details) < BASEBOARD_GPU_COUNT or len(names) != 1 or not names <= NVLINK_BASEBOARD_GPUS:
        return None
    return {
        "gpu_name": next(iter(names)),
        "gpu_count": len(details),
        "nvlink_active_links": interconnect.get("nvlink_active_links"),
        "pcie_class": interconnect.get("pcie_class"),
    }


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
