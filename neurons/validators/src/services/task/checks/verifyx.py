from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from typing import Any

from ..messages import VerifyXMessages as Msg, render_message
from ..pipeline import CheckResult, Context
from .network_ema import compute_ema

MIN_VERIFYX_EMA_DOWNLOAD_SPEED_MBPS = 5.0  # staging-only test branch, do not merge


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

        filler_container = _get_filler_only_container(ctx)
        if filler_container:
            updated_specs = _with_last_known_verifyx_ema(ctx)
            event = render_message(
                Msg.FILLER_SKIPPED,
                ctx=ctx,
                check_id=self.check_id,
                what={
                    "filler_container": filler_container,
                    "ema_verifyx_download_speed": (
                        (updated_specs.get("network") or {}).get("ema_verifyx_download_speed")
                    ),
                },
            )
            return CheckResult(
                passed=True,
                event=event,
                updates={"state": replace(ctx.state, specs=updated_specs)},
            )

        result = await verifyx_service.validate_verifyx_and_process_job(
            shell=ctx.services.shell,
            executor_info=ctx.executor,
            default_extra=ctx.default_extra,
            machine_spec=specs,
        )

        # Extract errors with clear priority: data.errors > result.error
        errors = self._extract_errors(result)

        prev_ema = (
            ctx.state.rented_data.network_ema.get(ctx.executor.uuid)
            if ctx.state.rented_data else None
        )

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

            # Always compute verifyx network EMA — use 0.0 when network measurement failed
            # so EMA decays toward 0 on repeated failures, eventually triggering exclusion
            if "network" not in updated_specs:
                updated_specs["network"] = {}
            download_speed = verifyx_network.get("download_speed")
            if download_speed is not None:
                updated_specs["network"]["verifyx_download_speed"] = download_speed
            updated_specs["network"]["ema_verifyx_download_speed"] = compute_ema(
                prev_ema.ema_verifyx_download_speed if prev_ema else None,
                download_speed if download_speed is not None else 0.0,
            )
            upload_speed = verifyx_network.get("upload_speed")
            if upload_speed is not None:
                updated_specs["network"]["verifyx_upload_speed"] = upload_speed
            updated_specs["network"]["ema_verifyx_upload_speed"] = compute_ema(
                prev_ema.ema_verifyx_upload_speed if prev_ema else None,
                upload_speed if upload_speed is not None else 0.0,
            )

            # Update storage specs if storage is present. Merged rather than replaced: VerifyX
            # measures only its own df-derived fields, so overwriting the dict wholesale would drop
            # the scrape's docker usage breakdown on every node where VerifyX runs.
            if "hard_disk" in sanitized:
                verifyx_hard_disk = sanitized.get("hard_disk")
                scraped_hard_disk = updated_specs.get("hard_disk")
                if isinstance(verifyx_hard_disk, dict) and isinstance(scraped_hard_disk, dict):
                    updated_specs["hard_disk"] = {**scraped_hard_disk, **verifyx_hard_disk}
                else:
                    updated_specs["hard_disk"] = verifyx_hard_disk

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

            ema_download = updated_specs["network"]["ema_verifyx_download_speed"]
            if ema_download < MIN_VERIFYX_EMA_DOWNLOAD_SPEED_MBPS:
                slow_event = render_message(
                    Msg.VERIFY_FAILED_NETWORK_SPEED_TOO_SLOW,
                    ctx=ctx,
                    check_id=self.check_id,
                    what={
                        "ema_verifyx_download_speed": ema_download,
                        "min_download_speed_mbps": MIN_VERIFYX_EMA_DOWNLOAD_SPEED_MBPS,
                    },
                )
                return CheckResult(passed=False, event=slow_event, updates={"state": updated_state})

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
        template = _FAILURE_TEMPLATE_BY_CLASS.get(
            diagnostics.get("failure_class"), Msg.VERIFY_FAILED
        )
        what: dict[str, Any] = {"errors": error_message, **diagnostics}

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


def _get_filler_only_container(ctx: Context) -> str | None:
    rented_data = ctx.state.rented_data
    if not rented_data:
        return None

    filler_container = rented_data.get_filler_container(ctx.executor.uuid)
    rented_executor = rented_data.executors.get(ctx.executor.uuid)
    has_customer_rental = bool(rented_executor and rented_executor.pods)
    return filler_container if filler_container and not has_customer_rental else None


def _with_last_known_verifyx_ema(ctx: Context) -> dict:
    specs = dict(ctx.state.specs or {})
    rented_data = ctx.state.rented_data
    network_ema = rented_data.network_ema.get(ctx.executor.uuid) if rented_data else None
    if not network_ema or network_ema.ema_verifyx_download_speed is None:
        return specs

    network = dict(specs.get("network") or {})
    network.setdefault("ema_verifyx_download_speed", network_ema.ema_verifyx_download_speed)
    if network_ema.ema_verifyx_upload_speed is not None:
        network.setdefault("ema_verifyx_upload_speed", network_ema.ema_verifyx_upload_speed)
    specs["network"] = network
    return specs


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
