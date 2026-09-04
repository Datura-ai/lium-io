"""DAH-2748: validation errors that mean "we could not reach something".

An availability error says the platform could not talk to a machine, not that the machine
failed a check. One is enough to take the node off the market: a node nobody can reach cannot
serve a customer, and offering it produces a failed rental.

The class is shared on purpose. Every future reachability check joins it by emitting an event
with this category — the validator reaching the node, the container reaching Docker Hub, the
container reaching Hugging Face. Nothing else has to change: the backend hides any node whose
last cycle carried an availability error, and the provider portal shows the reason.
"""

from enum import StrEnum

from services.task.models import ValidationEvent, build_msg

AVAILABILITY_CATEGORY = "availability"


class AvailabilityErrorCode(StrEnum):
    """Who could not reach what. Add a member for every new reachability check."""

    EXECUTOR_SSH_UNREACHABLE = "EXECUTOR_SSH_UNREACHABLE"


def build_availability_event(
    *,
    code: AvailabilityErrorCode,
    event: str,
    impact: str,
    remediation: str,
    what: dict,
) -> ValidationEvent:
    return build_msg(
        event=event,
        reason=str(code),
        severity="error",
        category=AVAILABILITY_CATEGORY,
        impact=impact,
        remediation=remediation,
        what=what,
    )


def availability_error_code(event: ValidationEvent | None) -> str | None:
    """The reason code when this event is an availability error, otherwise None."""
    if event is None or event.category != AVAILABILITY_CATEGORY:
        return None
    return event.reason_code


def build_ssh_unreachable_event(
    *, executor_uuid: str, host: str, port: int | None, error: str
) -> ValidationEvent:
    return build_availability_event(
        code=AvailabilityErrorCode.EXECUTOR_SSH_UNREACHABLE,
        event="Validator cannot open SSH to this node",
        impact="The node is hidden from the market and cannot take new rentals until a check succeeds.",
        remediation=(
            "Check that sshd on the node accepts the validator on its management port, "
            "and that no firewall or rate limit rejects the connection."
        ),
        what={
            "executor_uuid": executor_uuid,
            "ssh_host": host,
            "ssh_port": port,
            "error": error,
        },
    )
