from __future__ import annotations

from typing import Awaitable, Callable

from ..models import build_msg
from ..pipeline import CheckResult, Context


class PortConnectivityCheck:
    """Verify Docker port mappings by running the batch verifier exactly like before.

    Connectivity failures used to abort the task immediately because miners could not be
    rented. This check preserves that contract and updates sysbox state for later scoring.
    """

    check_id = "executor.validate.port_connectivity"
    fatal = True

    def __init__(
        self,
        *,
        renting_checker: Callable[[str, str], Awaitable[bool]],
        verifier: Callable[[object, str, str, object, str, str, bool], Awaitable[object]],
        private_key: str,
        public_key: str,
        job_batch_id: str,
    ):
        self.renting_checker = renting_checker
        self.verifier = verifier
        self.private_key = private_key
        self.public_key = public_key
        self.job_batch_id = job_batch_id

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

        renting_in_progress = await self.renting_checker(ctx.miner_hotkey, ctx.executor.uuid)
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

        result = await self.verifier(
            ctx.ssh,
            self.job_batch_id,
            ctx.miner_hotkey,
            ctx.executor,
            self.private_key,
            self.public_key,
            ctx.sysbox_runtime,
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
            return CheckResult(
                passed=False,
                event=event,
                updates={"default_extra": extra, "sysbox_runtime": result.sysbox_runtime},
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
        return CheckResult(
            passed=True,
            event=event,
            updates={
                "default_extra": extra,
                "sysbox_runtime": result.sysbox_runtime,
            },
        )
