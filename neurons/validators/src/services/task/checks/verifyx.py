from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from typing import Any

from ..messages import VerifyXMessages as Msg, render_message
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
            event = render_message(
                Msg.DISABLED,
                ctx=ctx,
                check_id=self.check_id,
            )
            return CheckResult(passed=True, event=event)
        verifyx_service = ctx.services.verifyx
        specs = ctx.state.specs
        if not specs:
            event = render_message(
                Msg.NO_SPECS,
                ctx=ctx,
                check_id=self.check_id,
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
            sanitized = _to_iso(result.data)
            updated_specs = dict(base_specs)
            updated_specs.update(
                {
                    "ram": sanitized.get("ram", updated_specs.get("ram")),
                    "hard_disk": sanitized.get("hard_disk", updated_specs.get("hard_disk")),
                }
            )

            if "network" in sanitized and "download_speed" in sanitized["network"]:
                if "network" not in updated_specs:
                    updated_specs["network"] = {}
                updated_specs["network"]["download_speed"] = sanitized["network"]["download_speed"]

            event = render_message(
                Msg.VERIFY_SUCCESS,
                ctx=ctx,
                check_id=self.check_id,
                what={"verifyx_success": True},
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

        event = render_message(
            Msg.VERIFY_FAILED,
            ctx=ctx,
            check_id=self.check_id,
            what={"errors": errors},
        )
        return CheckResult(passed=False, event=event)


def _to_iso(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _to_iso(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_iso(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_iso(v) for v in value)
    return value
