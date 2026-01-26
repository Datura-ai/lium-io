from __future__ import annotations

from core.config import settings
from ..messages import RentalVerificationMessages as Msg, render_message
from ..pipeline import CheckResult, Context


class RentalVerificationCheck:
    """Verify executor rental status via backend API health check.

    This check calls the backend API to verify that the executor can be successfully
    rented and is healthy. This is an additional verification step beyond checking
    the Redis RENTAL_SUCCEED_MACHINE_SET that provides real-time rental verification.
    """

    check_id = "executor.validate.rental_verification"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        """Run rental verification check via backend API.

        Args:
            ctx: Pipeline context

        Returns:
            CheckResult with verification status
        """
        # Skip if rental verification is disabled
        if settings.SKIP_RENTAL_VERIFICATION:
            event = render_message(
                Msg.SKIPPED,
                ctx=ctx,
                check_id=self.check_id,
                what={"skipped": True},
            )
            return CheckResult(
                passed=True,
                event=event,
                updates={},
            )

        # Get required info from context
        backend_client = ctx.services.backend
        executor = ctx.executor
        miner_hotkey = ctx.miner_hotkey

        # Get verified ports from PortConnectivityCheck
        verified_ports = ctx.state.specs.get("verified_ports", []) if ctx.state.specs else []

        # Fail if no verified ports are available (safety check - should be caught by PortConnectivityCheck)
        if not verified_ports:
            event = render_message(
                Msg.FAILED,
                ctx=ctx,
                check_id=self.check_id,
                what={
                    "verified": False,
                    "executor_uuid": executor.uuid,
                    "error": "No verified ports available for rental verification",
                },
                remediation="Port connectivity check should have failed - this is a safety check",
            )
            return CheckResult(
                passed=False,
                event=event,
                updates={},
            )

        # Use the first verified port
        container_port = verified_ports[0]

        try:
            # Call backend API to verify executor health
            response = await backend_client.check_executor_health(
                miner_address=executor.address,
                miner_port=executor.port,
                miner_hotkey=miner_hotkey,
                container_port=container_port,
                executor_id=executor.uuid,
            )

            # Handle API failure (None response) - fail this executor
            if response is None:
                event = render_message(
                    Msg.API_ERROR,
                    ctx=ctx,
                    check_id=self.check_id,
                    what={
                        "error": "API returned None",
                        "executor_uuid": executor.uuid,
                    },
                )
                return CheckResult(
                    passed=False,  # Fail this executor, continue with others
                    event=event,
                    updates={},
                )

            # Check if verification was successful
            if response.success:
                event = render_message(
                    Msg.VERIFIED,
                    ctx=ctx,
                    check_id=self.check_id,
                    what={
                        "verified": True,
                        "executor_uuid": executor.uuid,
                        "details": response.details or {},
                    },
                )
                return CheckResult(
                    passed=True,
                    event=event,
                    updates={},
                )
            else:
                # Verification failed - this is fatal
                event = render_message(
                    Msg.FAILED,
                    ctx=ctx,
                    check_id=self.check_id,
                    what={
                        "verified": False,
                        "executor_uuid": executor.uuid,
                        "error": response.error or "Unknown error",
                        "details": response.details or {},
                    },
                )
                return CheckResult(
                    passed=False,  # Fatal check - halt validation
                    event=event,
                    updates={},
                )

        except Exception as e:
            # Handle unexpected errors - fail this executor
            event = render_message(
                Msg.API_ERROR,
                ctx=ctx,
                check_id=self.check_id,
                what={
                    "error": str(e),
                    "executor_uuid": executor.uuid,
                },
            )
            return CheckResult(
                passed=False,  # Fail this executor, continue with others
                event=event,
                updates={},
            )
