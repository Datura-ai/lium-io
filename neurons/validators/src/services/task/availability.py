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
from typing import Any

from pydantic import BaseModel, Field
from services.task.models import AVAILABILITY_CATEGORY, JobResult, ValidationEvent, build_msg


class ReachSource(StrEnum):
    """Who tried to reach something."""

    VALIDATOR = "validator"
    CONTAINER = "container"


class ReachTarget(StrEnum):
    """What could not be reached. Add a member for every new reachability check."""

    EXECUTOR_SSH = "executor_ssh"


# The longest peer-supplied text a reading may keep: the node writes it, the portal shows it.
MAX_PEER_TEXT_LENGTH = 500
# Above this share of one cycle failing at the connect, the validator itself is the suspect and
# the cycle reports nothing about reachability.
FLEET_SHARE_THAT_MEANS_OUR_OWN_OUTAGE = 0.5


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


class AvailabilityError(BaseModel):
    """One reachability check that failed, in the shape the provider portal renders.

    A cycle can fail more than one — the image registry and the model hub, say — and each needs
    its own line: what could not be reached, and what we saw when we tried.
    """

    reason_code: str
    reach_source: str
    reach_target: str
    message: str
    remediation: str | None = None
    what_we_saw: dict[str, Any] = Field(default_factory=dict)


def availability_errors(events: list[ValidationEvent] | None) -> list[AvailabilityError]:
    """Every availability error this cycle raised, in the order the checks ran.

    The whole event list is read, not only the last one: a check that fails early still hides
    the node. One entry per reason code — a check that fires twice is still one problem.
    """
    errors: dict[str, AvailabilityError] = {}
    for event in events or []:
        if not event.is_availability_error or event.reason_code in errors:
            continue
        seen = dict(event.what_we_saw)
        errors[event.reason_code] = AvailabilityError(
            reason_code=event.reason_code,
            reach_source=str(seen.pop("reach_source", "")),
            reach_target=str(seen.pop("reach_target", "")),
            message=event.event,
            remediation=event.remediation,
            what_we_saw=seen,
        )
    return list(errors.values())


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
            # The node's own sshd writes this text, and the provider portal renders it, so the
            # node is not allowed to store a reading of any length it likes.
            "error": error[:MAX_PEER_TEXT_LENGTH],
        },
    )


def is_our_own_outage(unreachable_count: int, checked_count: int) -> bool:
    """True when so much of one cycle failed at the connect that the validator is the suspect.

    One node that refuses SSH is the node's problem. Most of a cycle refusing at once is ours —
    our egress, our DNS, our keys — and hiding the whole market over it is worse than listing a
    node nobody can reach for one more cycle. The backend's hourly sweep guards `active` the
    same way (DAH-2658).
    """
    if checked_count == 0:
        return False
    return unreachable_count / checked_count > FLEET_SHARE_THAT_MEANS_OUR_OWN_OUTAGE


def silence_availability_errors_on_our_own_outage(job_results: list[JobResult]) -> int:
    """Report nothing about reachability when the cycle looks like our own outage.

    Returns how many results were silenced, so the caller can log it; 0 means the cycle is
    trusted and every result keeps what it found.
    """
    checked_results = [result for result in job_results if result.availability_errors is not None]
    unreachable_results = [result for result in checked_results if result.availability_errors]
    if not is_our_own_outage(len(unreachable_results), len(checked_results)):
        return 0
    for result in checked_results:
        # None means "this cycle did not check": the backend leaves the stored errors alone.
        result.availability_errors = None
    return len(checked_results)
