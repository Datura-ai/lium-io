from __future__ import annotations

from dataclasses import replace

from ..models import build_msg
from ..pipeline import CheckResult, Context


class PortConnectivityCheck:
    """Verify Docker port mappings by running the batch verifier exactly like before.

    Connectivity failures used to abort the task immediately because miners could not be
    rented. This check preserves that contract and updates sysbox state for later scoring.
    """

    check_id = "executor.validate.port_connectivity"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        if ctx.rented:
            event = build_msg(
                event="Port connectivity skipped for rented executor",
                reason="PORT_CONNECTIVITY_SKIPPED",
                severity="info",
                category="runtime",
                impact="Proceed",
                check_id=self.check_id,
                ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
            )
            return CheckResult(passed=True, event=event)

        redis_service = ctx.services.redis
        renting_in_progress = await redis_service.renting_in_progress(ctx.miner_hotkey, ctx.executor.uuid)
        extra = {**ctx.default_extra, "renting_in_progress": renting_in_progress}

        if renting_in_progress:
            event = build_msg(
                event="Renting already in progress",
                reason="RENTING_IN_PROGRESS",
                severity="info",
                category="runtime",
                impact="Proceed",
                what={"renting_in_progress": True},
                check_id=self.check_id,
                ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
            )
            return CheckResult(
                passed=True,
                event=event,
                updates={"default_extra": extra, "renting_in_progress": True},
            )

        if not all([ctx.config.job_batch_id, ctx.config.port_private_key, ctx.config.port_public_key]):
            event = build_msg(
                event="Port connectivity configuration missing",
                reason="PORT_CONNECTIVITY_CONFIG_MISSING",
                severity="error",
                category="runtime",
                impact="Validation halted",
                remediation="Validator bug: missing port verification config in context",
                check_id=self.check_id,
                ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
            )
            return CheckResult(passed=False, event=event)

        connectivity_service = ctx.services.connectivity
        result = await connectivity_service.verify_ports(
            ctx.ssh,
            ctx.config.job_batch_id or "",
            ctx.miner_hotkey,
            ctx.executor,
            ctx.config.port_private_key or "",
            ctx.config.port_public_key or "",
            ctx.state.sysbox_runtime,
        )

        if not result.success:
            event = build_msg(
                event="Port verification failed",
                reason="PORT_VERIFY_FAILED",
                severity="error",
                category="runtime",
                impact="Score set to 0",
                remediation="Check Docker access and port mappings, then retry validation.",
                what={"details": result.log_text},
                check_id=self.check_id,
                ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
            )
            updated_state = replace(ctx.state, sysbox_runtime=result.sysbox_runtime)
            return CheckResult(
                passed=False,
                event=event,
                updates={"default_extra": extra, "state": updated_state},
            )

        event = build_msg(
            event="Port verification completed",
            reason="PORT_VERIFY_OK",
            severity="info",
            category="runtime",
            impact="Proceed",
            what={"message": result.log_text},
            check_id=self.check_id,
            ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
        )
        updated_state = replace(ctx.state, sysbox_runtime=result.sysbox_runtime)
        return CheckResult(
            passed=True,
            event=event,
            updates={
                "default_extra": extra,
                "state": updated_state,
            },
        )
