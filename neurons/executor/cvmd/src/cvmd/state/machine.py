"""The node state machine.

Names come from the CVM v2 architecture doc §04. Every transition is driven by a fact about the
host — a process observed in `/proc`, a descriptor observed open, a port observed bindable — not
by what cvmd last recorded.

`TEARDOWN` and `SWITCHING` split the crossing between validation and rented (FR-C5, FR-C7) at the
moment the two halves stop being the same question. In `TEARDOWN` cvmd is still stopping the CVM:
asking the guest to power off, then escalating. In `SWITCHING` the process is gone and the node is
still not free — the GPUs, the guest's memory and the forwarded ports come back to the host after
the process does, and on a large-memory TDX guest that tail is the slowest part of the whole
crossing. Both are the window FR-C7 calls "switching": neither rentable nor reachable, and not
offline either.
"""

from enum import StrEnum


class NodeState(StrEnum):
    RECONCILING = "RECONCILING"
    LAUNCHING = "LAUNCHING"
    VALIDATION_RUNNING = "VALIDATION_RUNNING"
    RENTER_RUNNING = "RENTER_RUNNING"
    TEARDOWN = "TEARDOWN"
    SWITCHING = "SWITCHING"
    FAILED = "FAILED"


# Legal edges, explicit. An illegal transition raises — a state machine that silently coerces an
# unexpected edge is one that reports a state the host is not in.
#
# RECONCILING reaches the running states directly because reconciliation discovers what is
# already on the host: a daemon restarted under a live renter CVM must be able to land on
# RENTER_RUNNING without pretending to launch it (DAH-2576 relies on this).
LEGAL_TRANSITIONS: dict[NodeState, frozenset[NodeState]] = {
    NodeState.RECONCILING: frozenset(
        {
            NodeState.LAUNCHING,
            NodeState.VALIDATION_RUNNING,
            NodeState.RENTER_RUNNING,
            NodeState.TEARDOWN,
            NodeState.FAILED,
        }
    ),
    NodeState.LAUNCHING: frozenset(
        {NodeState.VALIDATION_RUNNING, NodeState.RENTER_RUNNING, NodeState.FAILED}
    ),
    NodeState.VALIDATION_RUNNING: frozenset({NodeState.TEARDOWN, NodeState.FAILED}),
    NodeState.RENTER_RUNNING: frozenset({NodeState.TEARDOWN, NodeState.FAILED}),
    # RECONCILING as well as SWITCHING, because a teardown call that found nothing to stop has
    # no process to wait for and no resources to reclaim.
    NodeState.TEARDOWN: frozenset({NodeState.SWITCHING, NodeState.RECONCILING, NodeState.FAILED}),
    # Deliberately NOT to LAUNCHING. FR-C6 forbids starting the next CVM before the last one's
    # resources are actually free, and leaving that edge out is what enforces it structurally
    # rather than by remembering to check.
    NodeState.SWITCHING: frozenset({NodeState.RECONCILING, NodeState.FAILED}),
    # Recovery is deliberate, not automatic: FAILED goes back through reconciliation so the
    # daemon re-derives the host's real state instead of assuming the failure cleared.
    NodeState.FAILED: frozenset({NodeState.RECONCILING}),
}


class IllegalTransition(Exception):
    def __init__(self, current: NodeState, requested: NodeState) -> None:
        allowed = ", ".join(sorted(LEGAL_TRANSITIONS[current])) or "(none)"
        super().__init__(f"{current} -> {requested} is not a legal edge; allowed: {allowed}")
        self.current = current
        self.requested = requested


def is_legal(current: NodeState, requested: NodeState) -> bool:
    return requested in LEGAL_TRANSITIONS[current]
