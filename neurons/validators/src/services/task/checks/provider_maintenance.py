"""DAH-4709: a node inside its provider maintenance window fails PROVIDER_MAINTENANCE (score 0).

The window is compared with the cycle start (the clock of job_batch_id), half-open
[starting_at, expiring_at). Verification is kept: the node is not re-verified after the window.
A rented node is judged inside TenantEnforcementCheck, after its pod states are read; this
check fails an idle node before any probe runs on a machine the provider is working on.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from protocol.vc_protocol.compute_requests import MaintenanceWindow

from ..messages import ProviderMaintenanceMessages as Msg, render_message
from ..pipeline import CheckResult, Context

JOB_BATCH_ID_FORMAT = "%Y-%m-%d %H:%M:%S"


def _utc(moment: datetime) -> datetime:
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


def cycle_started_at(job_batch_id: str | None) -> datetime:
    """The cycle start, read from job_batch_id as the settlement does; the clock when it does not parse."""
    try:
        return datetime.strptime(job_batch_id or "", JOB_BATCH_ID_FORMAT).replace(tzinfo=UTC)
    except ValueError:
        return datetime.now(UTC)


def maintenance_window_at_cycle_start(ctx: Context) -> MaintenanceWindow | None:
    """The executor's maintenance window when this cycle started inside it, else None."""
    rented_data = ctx.state.rented_data
    window = (
        rented_data.maintenance_windows_by_executor.get(str(ctx.executor.uuid))
        if rented_data
        else None
    )
    if window is None:
        return None
    started = cycle_started_at(ctx.config.job_batch_id)
    return window if _utc(window.starting_at) <= started < _utc(window.expiring_at) else None


def maintenance_failure(
    ctx: Context,
    check_id: str,
    window: MaintenanceWindow,
    extra: dict[str, Any] | None = None,
    **updates: Any,
) -> CheckResult:
    # no clear_verified_job_info: the node keeps its verification through the window
    event = render_message(
        Msg.IN_WINDOW,
        ctx=ctx,
        check_id=check_id,
        what={
            "executor_uuid": ctx.executor.uuid,
            "cycle_started_at": cycle_started_at(ctx.config.job_batch_id).isoformat(),
            "starting_at": _utc(window.starting_at).isoformat(),
            "expiring_at": _utc(window.expiring_at).isoformat(),
        },
        extra=extra,
    )
    return CheckResult(passed=False, event=event, updates=updates)


class ProviderMaintenanceCheck:
    """Fail an idle node inside its maintenance window before VerifyX, the registry pull, the
    download-speed probe and the rental probe; a node with a rental passes to TenantEnforcementCheck."""

    check_id = "executor.validate.provider_maintenance"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        from .rented_machine import _rental_of_validated_miner  # rented_machine imports this module

        window = maintenance_window_at_cycle_start(ctx)
        rented_executor = (
            _rental_of_validated_miner(ctx, ctx.state.rented_data.executors.get(ctx.executor.uuid))
            if window
            else None
        )
        if window is None or (rented_executor and rented_executor.pods):
            event = render_message(
                Msg.NONE, ctx=ctx, check_id=self.check_id, what={"executor_uuid": ctx.executor.uuid}
            )
            return CheckResult(passed=True, event=event)
        return maintenance_failure(ctx, self.check_id, window)
