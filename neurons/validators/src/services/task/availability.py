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

from services.task.models import AVAILABILITY_CATEGORY, ValidationEvent, build_msg


class ReachSource(StrEnum):
    """Who tried to reach something."""

    VALIDATOR = "validator"
    CONTAINER = "container"


class ReachTarget(StrEnum):
    """What could not be reached. Add a member for every new reachability check."""

    EXECUTOR_SSH = "executor_ssh"


class AvailabilityErrorCode(StrEnum):
    """The code the backend stores and the portal shows. One per check."""

    EXECUTOR_SSH_UNREACHABLE = "EXECUTOR_SSH_UNREACHABLE"


def build_availability_event(
    *,
    code: AvailabilityErrorCode,
    reach_source: ReachSource,
    reach_target: ReachTarget,
    event_text: str,
    impact: str,
    remediation: str,
    what_we_saw: dict[str, str | int | None],
) -> ValidationEvent:
    """One availability error, naming who could not reach what."""
    return build_msg(
        event=event_text,
        reason=str(code),
        severity="error",
        category=AVAILABILITY_CATEGORY,
        impact=impact,
        remediation=remediation,
        what={
            "reach_source": str(reach_source),
            "reach_target": str(reach_target),
            **what_we_saw,
        },
    )


def availability_error_codes(events: list[ValidationEvent] | None) -> list[str]:
    """Every availability error this cycle raised, in the order the checks ran.

    A cycle can fail more than one reachability check — the image registry and the model hub,
    say — and the provider needs to see all of them, so the whole event list is read and every
    code is kept. Duplicates are dropped: one code per check.
    """
    return list(dict.fromkeys(event.reason_code for event in events or [] if event.is_availability_error))


def build_ssh_unreachable_event(
    *, executor_uuid: str, host: str, port: int | None, error: str
) -> ValidationEvent:
    return build_availability_event(
        code=AvailabilityErrorCode.EXECUTOR_SSH_UNREACHABLE,
        reach_source=ReachSource.VALIDATOR,
        reach_target=ReachTarget.EXECUTOR_SSH,
        event_text="Validator cannot open SSH to this node",
        impact="The node is hidden from the market and cannot take new rentals until a check succeeds.",
        remediation=(
            "Check that sshd on the node accepts the validator on its management port, "
            "and that no firewall or rate limit rejects the connection."
        ),
        what_we_saw={
            "executor_uuid": executor_uuid,
            "ssh_host": host,
            "ssh_port": port,
            "error": error,
        },
    )
