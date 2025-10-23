from __future__ import annotations

from typing import Callable, Tuple

from core.utils import StructuredMessage

from ..models import build_msg
from ..pipeline import CheckResult, Context


class GpuUsageCheck:
    """Re-use the legacy GPU utilisation guard for both rented and idle states.

    In the old flow high utilisation outside the validator zeroed the job. We keep that
    policy so miners cannot collect rewards while GPUs are already occupied by host jobs
    or rogue containers.
    """

    check_id = "gpu.validate.usage"
    fatal = True

    def __init__(
        self,
        *,
        gpu_usage_checker: Callable[[list[dict], list[dict], dict, bool], Tuple[bool, StructuredMessage | None]],
        rented: bool = False,
    ):
        self.gpu_usage_checker = gpu_usage_checker
        self.rented = rented

    async def run(self, ctx: Context) -> CheckResult:
        if ctx.rented and not self.rented:
            event = build_msg(
                event="GPU usage validation skipped for rented executor",
                reason="GPU_USAGE_SKIPPED",
                severity="info",
                category="runtime",
                impact="Proceed",
                what={"rented": True},
                check_id=self.check_id,
                ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
            )
            return CheckResult(passed=True, event=event)

        gpu_details = ctx.state.gpu_details
        gpu_processes = ctx.state.gpu_processes

        ok, log_msg = self.gpu_usage_checker(
            gpu_details,
            gpu_processes,
            ctx.default_extra,
            self.rented,
        )

        if ok:
            event = build_msg(
                event="GPU usage within limits",
                reason="GPU_USAGE_OK",
            severity="info",
            category="runtime",
            impact="Proceed",
            what={"process_count": len(gpu_processes)},
            check_id=self.check_id,
            ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
        )
            return CheckResult(passed=True, event=event)

        message = log_msg.message if isinstance(log_msg, StructuredMessage) else str(log_msg)
        payload = log_msg.extra if isinstance(log_msg, StructuredMessage) else {}

        event = build_msg(
            event=message,
            reason=payload.get("reason_code", "GPU_USAGE_VIOLATION"),
            severity=payload.get("severity", "warning"),
            category="runtime",
            impact=payload.get("impact", "Score set to 0"),
            remediation=payload.get("remediation"),
            what=payload.get("what_we_saw", {}),
            check_id=self.check_id,
            ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
        )
        return CheckResult(passed=False, event=event)
