"""Cross-node identity uniqueness — the defence against one CVM answering for many (DAH-2582).

A nonce proves that whatever answered was alive just now. It does not prove that two answers
came from two machines: a miner can register N executors, forward every challenge to one real
CVM, and get N valid, fresh, correctly-bound quotes back. Freshness is liveness, not
distinctness.

What separates the two is that a CVM cannot lie about *which* hardware it is. Three identifiers
in the evidence are bound to physical things the node does not choose:

    gpu_ueid          each GPU's own attested identity, from the NVIDIA evidence claims
    cvm_instance_id   the guest instance, from the verifier's app_info
    cvm_device_id     the device binding, from the same place
    pinned_host_key   the SSH host key the quote's report_data is bound to

If any one of them appears under two executors, those two executors are the same machine. The
evidence is signed by hardware, so this is provable rather than suspicious — which is why a
collision is fatal for **every** executor involved rather than for whichever one was seen
second. Picking a victim would let a miner farm the ordering; failing all of them makes the
attack cost strictly more than it earns.

**Why Redis, and why fleet-wide.** The registry is shared across every miner this validator
sees, because the interesting collision is between two *different* miners' registrations — the
same-miner case is what the older duplicate-executor check already catches. Entries expire, so
a node that legitimately re-registers under a new executor id after a rebuild eventually stops
colliding with its own past.

**Observe mode is the default and is the point.** The task's own acceptance bar is 48 hours of
observation showing zero collisions among genuinely distinct hosts before enforcement is turned
on. Some of these identifiers may turn out to be less unique than they look — `cvm_instance_id`
in particular is produced by the verifier and its scope across instances is exactly what the
observation window is there to establish. Shipping this enforcing would risk failing the whole
fleet over a shared constant; shipping it observing costs nothing and produces the evidence.
"""

import logging
from dataclasses import dataclass

from core.config import settings
from core.utils import _m, get_extra_info

logger = logging.getLogger(__name__)

# One Redis key per identity value. A key per value rather than a set per executor because the
# question being asked is "who else claims this?", and that is a lookup rather than a scan.
KEY_PREFIX = "attested_identity"

# The classes compared, and the order they are reported in — most specific hardware first, so a
# message names the GPU before it names the guest that was holding it.
IDENTITY_CLASSES = ("gpu_ueid", "cvm_device_id", "cvm_instance_id", "pinned_host_key")


@dataclass(frozen=True)
class Collision:
    """One identity claimed by two executors. Both are at fault; neither is 'the duplicate'."""

    identity_class: str
    value: str
    other_executor_uuid: str

    def describe(self) -> str:
        # The value is truncated: it is an identifier, not a secret, but a full ueid in a log
        # line is noise and the first 16 characters are already unambiguous in practice.
        return (
            f"{self.identity_class}={self.value[:16]}… is also claimed by executor "
            f"{self.other_executor_uuid}"
        )


class AttestedIdentityRegistry:
    """Remembers which executor claimed which hardware-bound identity, and for how long."""

    def __init__(self, redis_service) -> None:
        self._redis = redis_service

    @property
    def ttl_seconds(self) -> int:
        return settings.ATTESTED_IDENTITY_TTL_SECONDS

    def _key(self, identity_class: str, value: str) -> str:
        return f"{KEY_PREFIX}:{identity_class}:{value}"

    async def check_and_record(
        self, *, executor_uuid: str, identities: dict[str, list[str]]
    ) -> list[Collision]:
        """Record this executor's identities and return every one already claimed elsewhere.

        Recording happens even when a collision is found, deliberately. The registry's job is to
        answer "who claims this", and a fleet where the second claimant is silently not recorded
        would report the collision once and then forget one side of it — which makes the same
        pair look like a fresh collision on every cycle and gives an operator no way to tell a
        persistent fraud from a flapping one.

        Redis being unavailable returns no collisions and is logged. Unknown is not "guilty":
        the alternative is a Redis outage failing every CVM in the fleet, and the identities are
        still in the signed evidence, so nothing is lost but a cycle of detection.
        """
        if self._redis is None:
            return []

        collisions: list[Collision] = []
        for identity_class in IDENTITY_CLASSES:
            for value in identities.get(identity_class, []) or []:
                if not value:
                    continue
                try:
                    holder = await self._redis.get(self._key(identity_class, value))
                except Exception as exc:  # noqa: BLE001 - a registry outage must not fail a node
                    logger.warning(_m(
                        "Could not read the attested-identity registry",
                        extra=get_extra_info({"error": str(exc), "class": identity_class}),
                    ))
                    return collisions

                holder = holder.decode() if isinstance(holder, bytes) else holder
                if holder and holder != executor_uuid:
                    collisions.append(
                        Collision(
                            identity_class=identity_class,
                            value=value,
                            other_executor_uuid=holder,
                        )
                    )

                try:
                    await self._redis.set_with_expiration(
                        self._key(identity_class, value), executor_uuid, self.ttl_seconds
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(_m(
                        "Could not record an attested identity",
                        extra=get_extra_info({"error": str(exc), "class": identity_class}),
                    ))
        return collisions
