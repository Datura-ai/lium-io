"""The node state machine.

Names come from the CVM v2 architecture doc §04. The backend's "SWITCHING" is its surface word
for any non-terminal transition, not a state cvmd holds.

DAH-2575 ships the machine and its persistence only. Nothing here drives a transition from real
host facts — reconciling against QEMU/VFIO is DAH-2576's job, and until then RECONCILING is just
the initial state.
"""

from enum import StrEnum


class NodeState(StrEnum):
    RECONCILING = "RECONCILING"
    LAUNCHING = "LAUNCHING"
    VALIDATION_RUNNING = "VALIDATION_RUNNING"
    RENTER_RUNNING = "RENTER_RUNNING"
    TEARDOWN = "TEARDOWN"
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
    NodeState.TEARDOWN: frozenset({NodeState.RECONCILING, NodeState.FAILED}),
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
