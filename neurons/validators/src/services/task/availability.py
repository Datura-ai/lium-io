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
    MINER = "miner"


class ReachTarget(StrEnum):
    """What could not be reached. Add a member for every new reachability check."""

    EXECUTOR_SSH = "executor_ssh"
    # The executor's HTTP API, the miner's side of the node.
    EXECUTOR_API = "executor_api"
    # A card on the node (GpuLossHoldCheck): the validator cannot reach it through NVML.
    GPU = "gpu"
    # The node as a whole: a fatal check failed (failed_check_event).
    NODE = "node"


# The longest peer-supplied text a reading may keep: the node writes it, the portal shows it.
MAX_PEER_TEXT_LENGTH = 500
# Above this share of one cycle failing at the connect, the validator itself is the suspect and
# the cycle reports nothing about reachability.
FLEET_SHARE_THAT_MEANS_OUR_OWN_OUTAGE = 0.5
# Below this many checked nodes a share says nothing: one bad node out of two is already "most of
# them". A validator this small also cannot empty the market, so it reports what it found.
SMALLEST_FLEET_THAT_CAN_SHOW_AN_OUTAGE = 5


class AvailabilityErrorCode(StrEnum):
    """The code the backend stores and the portal shows. One per check."""

    EXECUTOR_SSH_UNREACHABLE = "EXECUTOR_SSH_UNREACHABLE"
    # DAH-3558: the backend lists the executor as rented, the miner answered the wave without it.
    RENTED_EXECUTOR_NOT_LISTED = "RENTED_EXECUTOR_NOT_LISTED"
    # An unrented node the miner lists (known_executor_ids) but could not reach to install our key.
    EXECUTOR_UNREACHABLE_BY_MINER = "EXECUTOR_UNREACHABLE_BY_MINER"
    # DAH-4457: a GPU dropped out of NVML and the node has not yet read clean and passed the load test (GpuLossHoldCheck).
    GPU_LOST = "GPU_LOST"
    # DAH-4683: NVML reports a card's memory broken (GpuMemoryHealthCheck).
    GPU_MEMORY_HEALTH_FAILED = "GPU_MEMORY_HEALTH_FAILED"
    # DAH-5395: a run ended by a failure that hides the node at once (failed_check_event).
    VERIFICATION_FAILED = "VERIFICATION_FAILED"
    # DAH-5351: a busy card's clock collapsed by a thermal slowdown, or still unproven since (GpuThermalThrottleCheck).
    GPU_THERMAL_COLLAPSE = "GPU_THERMAL_COLLAPSE"


# Faults the node itself proved: never a sign of our own outage, and never silenced with one.
NODE_GPU_FAULT_CODES = frozenset(
    {
        AvailabilityErrorCode.GPU_LOST,
        AvailabilityErrorCode.GPU_MEMORY_HEALTH_FAILED,
        AvailabilityErrorCode.GPU_THERMAL_COLLAPSE,
        # only raised for REASONS_THAT_HIDE_AT_ONCE, all read off the node's own report or probe
        AvailabilityErrorCode.VERIFICATION_FAILED,
    }
)


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


# The executor reports its own sshd's host key and the validator pins it to this address and port,
# so an untrusted key means a different sshd answered there: in practice the host's own sshd
# (a provider on Discord, 11 Oct 2026: the node SSH port was the host's SSH port). Firewall advice
# misleads. The key is re-read from the miner every cycle, so a recreated executor (new key, same
# port) fails one check and then passes. A repeat usually means the port; rarely the executor's
# sshd still serves a key it no longer reports (host-key files rotated without a reload).
SSH_HOST_KEY_MISMATCH_REMEDIATION = (
    "If this repeats every check, the SSH server on this port is not presenting the host key the "
    "executor reports. Usually another SSH server answers the port (the host's own sshd, or a "
    "port forward to another machine): set the node SSH port (and the public SSH port forward, if "
    "any) to a port that reaches the executor container, not the host's SSH port, then re-run "
    "`lium mine`. If the port already reaches the executor, restart the executor container so its "
    "SSH server loads the key it reports. A single failure while the executor container was being "
    "recreated clears by itself on the next check."
)


def build_ssh_unreachable_event(
    *, executor_uuid: str, host: str, port: int | None, error: str, host_key_mismatch: bool = False
) -> ValidationEvent:
    return build_availability_event(
        code=AvailabilityErrorCode.EXECUTOR_SSH_UNREACHABLE,
        reach_source=ReachSource.VALIDATOR,
        reach_target=ReachTarget.EXECUTOR_SSH,
        event_text="Validator cannot open SSH to this node",
        impact="The node is hidden from the market and cannot take new rentals until a check succeeds.",
        remediation=SSH_HOST_KEY_MISMATCH_REMEDIATION
        if host_key_mismatch
        else (
            "Check that sshd on the node accepts the validator on its management port, "
            "and that no firewall or rate limit rejects the connection. If your host NATs or "
            "proxies inbound SSH, set EXECUTOR_SSH_SOURCE_PIN=false in the executor .env and "
            "restart the executor."
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


def build_rented_executor_not_listed_event(
    *, executor_uuid: str, host: str, port: int | None, miner_hotkey: str
) -> ValidationEvent:
    """DAH-3558: a rented node the miner left out of its answer to the wave.

    No pipeline runs for a node the miner does not list, so without this event the wave writes
    nothing about it. This event is the wave's record: the backend lists the executor as rented,
    the miner did not return it. Why the miner left it out is the miner's business (it may drop
    a node before it ever tries to reach it), so the text says only that.
    """
    return build_availability_event(
        code=AvailabilityErrorCode.RENTED_EXECUTOR_NOT_LISTED,
        reach_source=ReachSource.MINER,
        reach_target=ReachTarget.EXECUTOR_API,
        event_text="Rented node missing from the miner's answer",
        impact=(
            "Score 0 for this cycle; the node is hidden from the market until a cycle lists it "
            "again. The backend's staleness sweep reads this row as the node not answering."
        ),
        remediation=(
            "The miner did not return this node in its answer to the validator. Check that the "
            "node is up and that the miner lists it."
        ),
        what_we_saw={
            "executor_uuid": executor_uuid,
            "executor_ip_address": host,
            "executor_port": port,
            "miner_hotkey": miner_hotkey,
        },
    )


def build_executor_unreachable_by_miner_event(*, executor_uuid: str, miner_hotkey: str) -> ValidationEvent:
    """An unrented node the miner lists for this validator and could not reach this wave.

    The miner's answer carries the node's id in ``known_executor_ids`` and not in ``executors``:
    it tried to install the validator's key and the node did not take it. Without this event the
    wave writes nothing and the node only "stops reporting". The miner sends no address for it.
    """
    return build_availability_event(
        code=AvailabilityErrorCode.EXECUTOR_UNREACHABLE_BY_MINER,
        reach_source=ReachSource.MINER,
        reach_target=ReachTarget.EXECUTOR_API,
        event_text="Miner could not reach this node to install the validator's key",
        impact="Score 0 for this cycle; the node is hidden from the market until a cycle reaches it.",
        remediation=(
            "Check that the executor is running and that its API port accepts the miner's "
            "/upload_ssh_key request."
        ),
        what_we_saw={"executor_uuid": executor_uuid, "miner_hotkey": miner_hotkey},
    )


def is_our_own_outage(unreachable_count: int, checked_count: int) -> bool:
    """True when so much of one cycle failed at the connect that the validator is the suspect.

    One node that refuses SSH is the node's problem. Most of a cycle refusing at once is ours —
    our egress, our DNS, our keys — and hiding the whole market over it is worse than listing a
    node nobody can reach for one more cycle. The backend's hourly sweep guards `active` the
    same way (DAH-2658).

    A cycle with only a few nodes is exempt: there the share carries no signal, and a validator
    that small cannot empty the market anyway.
    """
    if checked_count < SMALLEST_FLEET_THAT_CAN_SHOW_AN_OUTAGE:
        return False
    return unreachable_count / checked_count > FLEET_SHARE_THAT_MEANS_OUR_OWN_OUTAGE


def _is_the_validators_own_reading(job_result: JobResult) -> bool:
    """False for a result the validator wrote without contacting the node.

    A RENTED_EXECUTOR_NOT_LISTED row (DAH-3558) is the miner's answer, not a connect the validator
    made, so it says nothing about our egress, DNS or keys: it is left out of the outage ratio on
    both sides, and the silencer leaves it alone. A wave-wide drop by one miner would otherwise
    read as our outage and silence real SSH failures on the rest of the cycle.
    """
    errors = job_result.availability_errors
    if errors is None:
        return False
    return not errors or any(error.get("reach_source") != ReachSource.MINER for error in errors)


def _checked_and_unreachable(job_results: list[JobResult]) -> tuple[list[JobResult], list[JobResult]]:
    """The two sides of the outage share: the validator's own readings, and those that failed at the connect."""
    checked_results = [result for result in job_results if _is_the_validators_own_reading(result)]
    # a lost or broken GPU is the node's own fault, never our outage: not counted in the share
    unreachable_results = [
        result
        for result in checked_results
        if any(
            error.get("reason_code") not in NODE_GPU_FAULT_CODES
            for error in result.availability_errors or []
        )
    ]
    return checked_results, unreachable_results


def silence_availability_errors_on_our_own_outage(job_results: list[JobResult]) -> int:
    """Report nothing about reachability when the cycle looks like our own outage.

    Returns how many results were silenced, so the caller can log it; 0 means the cycle is
    trusted and every result keeps what it found. Only the validator's own readings count and
    are silenced (``_is_the_validators_own_reading``).
    """
    checked_results, unreachable_results = _checked_and_unreachable(job_results)
    if not is_our_own_outage(len(unreachable_results), len(checked_results)):
        return 0
    for result in checked_results:
        # None means "this cycle did not check": the backend leaves the stored errors alone.
        # A lost or broken GPU is the node's own fault: its entry stays, the node stays hidden.
        node_gpu_fault_entries = [
            error
            for error in result.availability_errors or []
            if error.get("reason_code") in NODE_GPU_FAULT_CODES
        ]
        result.availability_errors = node_gpu_fault_entries or None
    return len(checked_results)


class RunningOutageTally:
    """DAH-5635: one cycle's outage share so far, fed each miner's results as its job returns.

    An unrented node that failed is published at once, not after the slowest miner in the fleet,
    but only while the share so far is trusted: at least SMALLEST_FLEET_THAT_CAN_SHOW_AN_OUTAGE
    checked results, and not what ``silence_availability_errors_on_our_own_outage`` would call
    our own outage. A result from another job_batch_id is ignored. The cycle starts a fresh one.
    """

    def __init__(self, job_batch_id: str, rented_executor_ids: frozenset[str] = frozenset()):
        self.job_batch_id = job_batch_id
        # the backend's rented list at cycle start: a failure is built before the result knows its rentals
        self.rented_executor_ids = rented_executor_ids
        self.checked_count = 0
        self.unreachable_count = 0
        # executor uuids published early this cycle
        self.published_early: set[str] = set()

    def add(self, job_results: list[JobResult]) -> list[JobResult]:
        """Count one miner's results; return those that may be published now (none while the share is untrusted)."""
        checked, unreachable = _checked_and_unreachable(
            [result for result in job_results if result.job_batch_id == self.job_batch_id]
        )
        self.checked_count += len(checked)
        self.unreachable_count += len(unreachable)
        if self.checked_count < SMALLEST_FLEET_THAT_CAN_SHOW_AN_OUTAGE or is_our_own_outage(
            self.unreachable_count, self.checked_count
        ):
            return []
        # DAH-5635: a rented node waits for the cycle, so its pod's reboot streak counts once
        return [
            result
            for result in checked
            if result.availability_errors
            and not result.is_successful
            and not result.is_rented
            and str(result.executor_info.uuid).lower() not in self.rented_executor_ids
        ]

    def restore_published_early(
        self, job_results: list[JobResult], published_before: dict[str, list[dict]]
    ) -> None:
        """After the cycle-end silencer fired: a node published early gets back the list the last cycle
        published for it (published_before), so a node hidden before this cycle stays hidden and one
        listed is listed again. None (stored errors left alone, the early ones included) when this
        process has not published it yet. The silencer's GPU-fault entries stay on top.
        """
        for result in job_results:
            uuid = str(result.executor_info.uuid)
            if uuid not in self.published_early:
                continue
            before = published_before.get(uuid.lower())
            if before is None:
                continue
            kept = result.availability_errors or []
            result.availability_errors = kept + [error for error in before if error not in kept]


# DAH-5395: the failure reasons that hide the node at once. Any other failed run keeps the backend's usual rule.
REASONS_THAT_HIDE_AT_ONCE = frozenset({"GPU_DETAILS_MISMATCH", "GPU_UUID_CHANGED", "NVLINK_FAULT"})


def failed_check_event(ok: bool, events: list[ValidationEvent]) -> ValidationEvent | None:
    """The availability entry for a run ended by a reason in REASONS_THAT_HIDE_AT_ONCE, else None.

    The backend hides the node on it and lists it again on the next cycle that sends an empty list.
    A run that already raised an availability error gets none, so nothing is doubled.
    """
    if ok or not events or any(e.is_availability_error for e in events):
        return None
    last_event = events[-1]
    if last_event.reason_code not in REASONS_THAT_HIDE_AT_ONCE:
        return None
    return build_availability_event(
        code=AvailabilityErrorCode.VERIFICATION_FAILED,
        reach_source=ReachSource.VALIDATOR,
        reach_target=ReachTarget.NODE,
        event_text="The node failed a verification check",
        impact="Hidden and takes no new rentals until a check passes; a rental already on it is not touched",
        remediation="See the failed check's message for the cause. The node is listed again on the first check "
        "that passes.",
        what_we_saw={"failed_check": last_event.check_id, "failed_reason": last_event.reason_code},
    )
