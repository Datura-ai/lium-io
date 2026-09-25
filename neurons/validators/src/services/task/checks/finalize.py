from __future__ import annotations

from services.const import BATCH_PORT_VERIFICATION_SIZE, MIN_PORT_COUNT, UNRENTED_MULTIPLIER

from ..messages import FinalizeMessages as Msg, render_message
from ..pipeline import CheckResult, Context
from .port_count import hidden_from_renters_text, port_count_below_listing_floor, port_floor_what


class FinalizeCheck:
    """Aggregate pipeline results into the structured log + success flag.

    This check produces the user-facing log message and translates score warnings into
    remediation guidance. Housing it in a dedicated check makes the terminal behaviour
    explicit and easy to review.
    """

    check_id = "pipeline.finalize"
    fatal = False

    async def run(self, ctx: Context) -> CheckResult:
        success = ctx.score > 0
        severity = "info" if success else "warning"

        if ctx.score_warning:
            remediation = ("No action needed." + ctx.score_warning) if success else ("Address issues:" + ctx.score_warning)
        else:
            remediation = "No action needed." if success else "Address issues."

        impact = f"Job score={ctx.job_score}, actual score={ctx.score}"
        what = {
            "gpu_model": ctx.state.gpu_model,
            "gpu_count": ctx.state.gpu_count if ctx.state.gpu_count is not None else 0,
            "contract_version": ctx.contract_version,
            "unrented_multiplier": UNRENTED_MULTIPLIER,
            "sysbox_runtime": ctx.state.sysbox_runtime,
        }
        # Only a run exempted from PortCountCheck by a pod that then proved stale gets here below the floor.
        port_count_below_floor = port_count_below_listing_floor(ctx.state)
        if port_count_below_floor is not None:
            impact = f"{hidden_from_renters_text(port_count_below_floor)}. {impact}"
            what["port_floor"] = port_floor_what(ctx.state, port_count_below_floor)
            port_floor_fix = (
                f"Only ports that answer among the lowest {BATCH_PORT_VERIFICATION_SIZE} free ports of the "
                f"declared range count: allow at least {MIN_PORT_COUNT} of them through the host firewall and "
                "any port forwarding, or declare only open ports; the next cycle probes them again."
            )
            remediation = f"{remediation} {port_floor_fix}" if ctx.score_warning else port_floor_fix

        event = render_message(
            Msg.COMPLETED,
            ctx=ctx,
            check_id=self.check_id,
            severity=severity,
            impact=impact,
            remediation=remediation,
            what=what,
        )

        return CheckResult(
            passed=True,
            event=event,
            updates={
                "success": success,
                "log_status": severity,
                "log_text": event.event,
            },
        )
