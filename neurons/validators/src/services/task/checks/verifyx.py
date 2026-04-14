from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from typing import Any

from ..messages import VerifyXMessages as Msg, render_message
from ..pipeline import CheckResult, Context
from .network_ema import compute_ema


class VerifyXCheck:
    """Run the optional VerifyX hardware probe and update specs with its findings.

    This preserves the legacy feature flag behaviour: when VerifyX is enabled we block on
    its success, otherwise the check becomes a no-op. Documenting it here lets us debate
    the value of the external probe independently from the rest of the pipeline.
    """

    check_id = "gpu.validate.verifyx"
    fatal = True

    def _extract_errors(self, result) -> str | list | None:
        """Extract errors from result with clear priority: data.errors > result.error."""
        if result.data and result.data.get("errors"):
            return result.data["errors"]
        return result.error

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

        # Extract errors with clear priority: data.errors > result.error
        errors = self._extract_errors(result)

        if result.data and result.data.get("success"):
            base_specs = ctx.state.specs
            sanitized = _to_iso(result.data)
            verifyx_network = sanitized.get("network", {})
            speedtest_network = base_specs.get("network", {}) or {}
            updated_specs = dict(base_specs)
            updated_specs.update(
                {
                    "ram": sanitized.get("ram", updated_specs.get("ram")),
                }
            )

            # Store verifyx network measurements under their own keys (additive — does not overwrite speedtest values)
            if verifyx_network.get("download_speed") is not None:
                if "network" not in updated_specs:
                    updated_specs["network"] = {}
                updated_specs["network"]["verifyx_download_speed"] = verifyx_network.get("download_speed")
                prev_ema = (
                    ctx.state.rented_data.network_ema.get(ctx.executor.uuid)
                    if ctx.state.rented_data else None
                )
                updated_specs["network"]["ema_verifyx_download_speed"] = compute_ema(
                    prev_ema.ema_verifyx_download_speed if prev_ema else None,
                    verifyx_network.get("download_speed"),
                )

            # Update storage specs if storage is present
            if "hard_disk" in sanitized:
                updated_specs.update({
                    "hard_disk": sanitized.get("hard_disk", updated_specs.get("hard_disk")),
                })

            event = render_message(
                Msg.VERIFY_SUCCESS,
                ctx=ctx,
                check_id=self.check_id,
                what={
                    "verifyx_success": True,
                    "verifyx_network_success": verifyx_network.get("success"),
                    "network": speedtest_network,
                },
            )
            if errors:
                event.what_we_saw["errors"] = errors

            updated_state = replace(ctx.state, specs=updated_specs)

            return CheckResult(
                passed=True,
                event=event,
                updates={
                    "state": updated_state,
                },
            )

        # Ensure we have an error message for the failure case
        error_message = errors or "Unknown errors"

        diagnostics = getattr(result, "diagnostics", None) or {}
        template = _select_failure_template(diagnostics.get("failure_class"))

        what: dict[str, Any] = {"errors": error_message}
        if diagnostics:
            what["failure_class"] = diagnostics.get("failure_class")
            what["exit_status"] = diagnostics.get("exit_status")
            what["stdout_len"] = diagnostics.get("stdout_len")
            what["stderr_tail"] = diagnostics.get("stderr_tail")
            if diagnostics.get("transport_error") is not None:
                what["transport_error"] = diagnostics.get("transport_error")

        event = render_message(
            template,
            ctx=ctx,
            check_id=self.check_id,
            what=what,
        )
        return CheckResult(passed=False, event=event)


_FAILURE_TEMPLATE_BY_CLASS = {
    "SSH_TRANSPORT": Msg.VERIFY_FAILED_SSH_TRANSPORT,
    "EXECUTOR_CRASH": Msg.VERIFY_FAILED_EXECUTOR_CRASH,
    "EMPTY_RESPONSE": Msg.VERIFY_FAILED_EMPTY_RESPONSE,
    "CIPHER_REJECTED": Msg.VERIFY_FAILED_CIPHER_REJECTED,
}


def _select_failure_template(failure_class: str | None):
    if failure_class is None:
        return Msg.VERIFY_FAILED
    return _FAILURE_TEMPLATE_BY_CLASS.get(failure_class, Msg.VERIFY_FAILED)


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
