from __future__ import annotations

from dataclasses import replace
from dataclasses import replace

from ..models import build_msg
from ..pipeline import CheckResult, Context


class VerifyXCheck:
    """Run the optional VerifyX hardware probe and update specs with its findings.

    This preserves the legacy feature flag behaviour: when VerifyX is enabled we block on
    its success, otherwise the check becomes a no-op. Documenting it here lets us debate
    the value of the external probe independently from the rest of the pipeline.
    """

    check_id = "gpu.validate.verifyx"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        if not ctx.config.verifyx_enabled:
            event = build_msg(
                event="VerifyX validation skipped",
                reason="VERIFYX_DISABLED",
                severity="info",
                category="env",
                impact="Proceed",
                check_id=self.check_id,
                ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
            )
            return CheckResult(passed=True, event=event)
        verifyx_service = ctx.services.verifyx
        specs = ctx.state.specs
        if not specs:
            event = build_msg(
                event="VerifyX validation skipped (no specs)",
                reason="VERIFYX_NO_SPECS",
                severity="error",
                category="env",
                impact="Validation halted",
                remediation="Run the machine scrape before executing VerifyX.",
                check_id=self.check_id,
                ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
            )
            return CheckResult(passed=False, event=event)
        result = await verifyx_service.validate_verifyx_and_process_job(
            shell=ctx.services.shell,
            executor_info=ctx.executor,
            default_extra=ctx.default_extra,
            machine_spec=specs,
        )

        if result.data and result.data.get("success"):
            base_specs = ctx.state.specs
            updated_specs = dict(base_specs)
            updated_specs.update(
                {
                    "ram": result.data.get("ram", updated_specs.get("ram")),
                    "hard_disk": result.data.get("hard_disk", updated_specs.get("hard_disk")),
                    "network": result.data.get("network", updated_specs.get("network")),
                }
            )

            event = build_msg(
                event="VerifyX validation passed",
                reason="VERIFYX_OK",
                severity="info",
                category="env",
                impact="Proceed",
                what={"verifyx_success": True},
                check_id=self.check_id,
                ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
            )
            updated_state = replace(ctx.state, specs=updated_specs)

            return CheckResult(
                passed=True,
                event=event,
                updates={
                    "state": updated_state,
                },
            )

        errors = None
        if result.data:
            errors = result.data.get("errors")
        errors = errors or result.error or "Unknown errors"

        event = build_msg(
            event="VerifyX validation failed",
            reason="VERIFYX_FAILED",
            severity="error",
            category="env",
            impact="Score set to 0",
            remediation="Run VerifyX locally to debug network, disk, and RAM probes.",
            what={"errors": errors},
            check_id=self.check_id,
            ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
        )
        return CheckResult(passed=False, event=event)
