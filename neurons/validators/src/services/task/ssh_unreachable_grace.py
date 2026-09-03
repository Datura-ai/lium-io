"""DAH-2748: one cycle of grace when the validator cannot open SSH to a node.

A refused connection is not proof that the node is bad — sshd throttling, a reboot or a
network blip all look the same from here. Zeroing the score on the first cycle takes the
provider's pay for something we did not measure. The first unreachable cycle therefore
carries the last good score forward; the second one in a row scores zero.

The count lives in Redis, one key per node, and any complete cycle clears it. The backend
counts the same two cycles on its own row (`executor.validation_failure_streak`) and closes
new rentals at the same point: a forgiven cycle still arrives without machine specs, which
is what the backend counts.
"""

import logging
from dataclasses import dataclass
from typing import Any

from services.const import SSH_UNREACHABLE_STREAK_TTL_SECONDS
from services.task.models import ValidationEvent, build_msg

logger = logging.getLogger(__name__)

SSH_UNREACHABLE_REASON_CODE = "EXECUTOR_SSH_UNREACHABLE"
# The first unreachable cycle is forgiven; the second in a row is not.
SSH_UNREACHABLE_CYCLES_BEFORE_ZERO_SCORE = 2


@dataclass(frozen=True)
class UnreachableVerdict:
    """What this cycle scores, and why."""

    score: float
    streak: int
    forgiven: bool


class SshUnreachableGrace:
    def __init__(self, redis_service: Any) -> None:
        self.redis_service = redis_service

    @staticmethod
    def _streak_key(executor_uuid: str) -> str:
        return f"executor:{executor_uuid}:ssh_unreachable_streak"

    @staticmethod
    def _last_score_key(executor_uuid: str) -> str:
        return f"executor:{executor_uuid}:last_score"

    async def score_for_unreachable_cycle(self, executor_uuid: str) -> UnreachableVerdict:
        """Count this failure and say what the cycle scores."""
        try:
            previous_streak: int = await self._read_int(self._streak_key(executor_uuid))
            last_good_score: float = await self._read_float(self._last_score_key(executor_uuid))
        except Exception as error:
            # Our own outage must not cost the provider anything: forgive and count nothing.
            logger.warning(
                "Cannot read the SSH grace state for %s, forgiving this cycle: %s", executor_uuid, error
            )
            return UnreachableVerdict(score=0.0, streak=1, forgiven=True)

        streak: int = previous_streak + 1
        await self._write_streak(executor_uuid, streak)

        forgive: bool = streak < SSH_UNREACHABLE_CYCLES_BEFORE_ZERO_SCORE and last_good_score > 0
        score: float = last_good_score if forgive else 0.0
        return UnreachableVerdict(score=score, streak=streak, forgiven=forgive)

    async def record_successful_cycle(self, executor_uuid: str, score: float) -> None:
        """Clear the streak and remember the score this cycle earned."""
        try:
            await self.redis_service.delete(self._streak_key(executor_uuid))
            if score > 0:
                await self.redis_service.set(self._last_score_key(executor_uuid), str(score))
        except Exception as error:
            logger.warning("Cannot store the SSH grace state for %s: %s", executor_uuid, error)

    @staticmethod
    def build_event(
        *,
        executor_uuid: str,
        host: str,
        port: int | None,
        error: str,
        streak: int,
        forgiven: bool,
    ) -> ValidationEvent:
        impact: str = (
            "Score kept for this cycle. The next failure in a row scores zero."
            if forgiven
            else "Score is zero for this cycle and new rentals are closed until a check succeeds."
        )
        return build_msg(
            event="Validator cannot open SSH to this node",
            reason=SSH_UNREACHABLE_REASON_CODE,
            severity="error",
            category="connectivity",
            impact=impact,
            remediation=(
                "Check that sshd on the node accepts the validator on its management port, "
                "and that no firewall or rate limit rejects the connection."
            ),
            what={
                "executor_uuid": executor_uuid,
                "ssh_host": host,
                "ssh_port": port,
                "error": error,
                "consecutive_failures": streak,
            },
        )

    async def _read_int(self, key: str) -> int:
        raw = await self.redis_service.get(key)
        return int(raw) if raw is not None else 0

    async def _read_float(self, key: str) -> float:
        raw = await self.redis_service.get(key)
        return float(raw) if raw is not None else 0.0

    async def _write_streak(self, executor_uuid: str, streak: int) -> None:
        try:
            await self.redis_service.set(
                self._streak_key(executor_uuid), str(streak), ex=SSH_UNREACHABLE_STREAK_TTL_SECONDS
            )
        except TypeError:
            # Older RedisService.set has no expiry argument.
            await self.redis_service.set(self._streak_key(executor_uuid), str(streak))
