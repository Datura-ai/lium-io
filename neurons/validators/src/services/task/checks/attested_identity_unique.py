from __future__ import annotations

from core.config import settings
from services.attested_identity import AttestedIdentityRegistry

from ..messages import AttestedIdentityMessages as Msg, render_message
from ..pipeline import CheckResult, Context


class AttestedIdentityUniqueCheck:
    """No two executors may claim the same hardware-bound identity (DAH-2582).

    The attack this closes is challenge forwarding: register N executors, forward every
    challenge to one real CVM, return N valid, fresh, correctly-bound quotes. Every existing
    check passes, because every quote genuinely is fresh and genuinely does come from a TDX
    guest — the nonce proves liveness, and liveness is not distinctness.

    What a CVM cannot do is lie about which hardware it is. The GPU ueids, the guest's instance
    and device ids, and the host key its report_data is bound to all come out of signed evidence.
    If one of them shows up under two executors, those executors are one machine, and the proof
    of that is a signature.

    So a collision fails **both** sides. Failing only the one seen second would let a miner
    choose which registration survives by controlling the order they are validated in; failing
    both means the fraud costs the attacker every node it touched.

    Runs in observe mode until `ENABLE_ATTESTED_IDENTITY_UNIQUENESS` is set. That is the task's
    own acceptance path — 48 hours over the fleet showing zero collisions among genuinely
    distinct hosts — and it matters because one of these identifiers could turn out to be less
    unique than it looks. Enforcing on day one would risk failing the whole fleet over a shared
    constant; observing costs nothing and produces the evidence either way.
    """

    check_id = "executor.validate.attested_identity"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        identities = ctx.state.attested_identities or {}
        values = [value for values in identities.values() for value in values if value]
        if not values:
            # A node with no attested identity is not a CVM, or did not attest this cycle.
            # Whether that is allowed at all is the fail-closed quote check's question, not
            # this one's — this check compares identities and there are none to compare.
            return CheckResult(
                passed=True,
                event=render_message(Msg.NOT_ATTESTED, ctx=ctx, check_id=self.check_id),
            )

        registry = AttestedIdentityRegistry(ctx.services.redis)
        collisions = await registry.check_and_record(
            executor_uuid=str(ctx.executor.uuid), identities=identities
        )

        if not collisions:
            return CheckResult(
                passed=True,
                event=render_message(
                    Msg.UNIQUE,
                    ctx=ctx,
                    check_id=self.check_id,
                    what={"identity_count": len(values)},
                ),
            )

        what = {
            "executor_uuid": str(ctx.executor.uuid),
            "miner_hotkey": ctx.miner_hotkey,
            "collisions": [
                {
                    "class": collision.identity_class,
                    "value": collision.value,
                    "also_claimed_by": collision.other_executor_uuid,
                }
                for collision in collisions
            ],
            "detail": "; ".join(collision.describe() for collision in collisions),
        }

        if not settings.ENABLE_ATTESTED_IDENTITY_UNIQUENESS:
            return CheckResult(
                passed=True,
                event=render_message(
                    Msg.COLLISION_OBSERVED, ctx=ctx, check_id=self.check_id, what=what
                ),
            )

        return CheckResult(
            passed=False,
            event=render_message(Msg.COLLISION, ctx=ctx, check_id=self.check_id, what=what),
            # Same consequence as the duplicate-executor check: the node's verified job is
            # cleared, so it does not keep earning on a verification taken before the collision
            # was known.
            updates={"clear_verified_job_info": True},
        )
