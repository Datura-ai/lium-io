"""DAH-2748: validation errors that mean "someone could not reach something".

An availability error says the platform could not talk to a machine or a service, not that the
machine failed a check. One is enough to take the node off the market: a node nobody can reach
cannot serve a customer, and offering it produces a failed rental.

The class is shared on purpose. A new reachability check joins it by naming who could not reach
what and emitting the event — the validator reaching the node today, the container reaching
Docker Hub or Hugging Face tomorrow. Nothing downstream changes: the backend hides any node
whose last cycle carried an availability error, and the provider portal shows the reason.
"""

from enum import StrEnum

from services.task.models import ValidationEvent, build_msg

AVAILABILITY_CATEGORY = "availability"


class Reacher(StrEnum):
    """Who tried to reach something."""

    VALIDATOR = "validator"
    CONTAINER = "container"


class Reached(StrEnum):
    """What could not be reached. Add a member for every new reachability check."""

    EXECUTOR_SSH = "executor_ssh"


class AvailabilityErrorCode(StrEnum):
    """The code the backend stores and the portal shows. One per check."""

    EXECUTOR_SSH_UNREACHABLE = "EXECUTOR_SSH_UNREACHABLE"


def build_availability_event(
    *,
    code: AvailabilityErrorCode,
    reacher: Reacher,
    reached: Reached,
    event: str,
    impact: str,
    remediation: str,
    what: dict,
) -> ValidationEvent:
    """One availability error. `reacher` and `reached` say who could not reach what."""
    return build_msg(
        event=event,
        reason=str(code),
        severity="error",
        category=AVAILABILITY_CATEGORY,
        impact=impact,
        remediation=remediation,
        what={"reacher": str(reacher), "reached": str(reached), **what},
    )


def first_availability_error_code(events: list[ValidationEvent] | None) -> str | None:
    """The code of the first availability error in this cycle, if it had one.

    The whole list is read, not only the last event: a reachability check that fails early
    still hides the node, whatever the pipeline reports afterwards.
    """
    for event in events or []:
        if event.is_availability_error:
            return event.reason_code
    return None


def build_ssh_unreachable_event(
    *, executor_uuid: str, host: str, port: int | None, error: str
) -> ValidationEvent:
    return build_availability_event(
        code=AvailabilityErrorCode.EXECUTOR_SSH_UNREACHABLE,
        reacher=Reacher.VALIDATOR,
        reached=Reached.EXECUTOR_SSH,
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
