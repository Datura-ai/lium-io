"""State machine edges and persistence."""

import itertools

import pytest
from cvmd.state.machine import LEGAL_TRANSITIONS, IllegalTransition, NodeState, is_legal
from cvmd.state.store import STATE_FILENAME, StateDocument, StateStore

EPOCH = "1970-01-01T00:00:00+00:00"

LEGAL_EDGES = [
    (current, target) for current, targets in LEGAL_TRANSITIONS.items() for target in targets
]
ILLEGAL_EDGES = [
    (current, target)
    for current, target in itertools.product(NodeState, NodeState)
    if not is_legal(current, target)
]


def _drive(store: StateStore, path: list[NodeState]) -> None:
    for step in path:
        store.transition(step)


class TestTransitions:
    @pytest.mark.parametrize(("current", "target"), LEGAL_EDGES, ids=lambda v: str(v))
    def test_every_legal_edge_is_reachable(self, state_dir, current, target):
        store = StateStore(state_dir)
        store._document = StateDocument(state=current, entered_at=EPOCH, transition_id=0)
        assert store.transition(target).state is target

    @pytest.mark.parametrize(("current", "target"), ILLEGAL_EDGES, ids=lambda v: str(v))
    def test_every_illegal_edge_raises(self, state_dir, current, target):
        store = StateStore(state_dir)
        store._document = StateDocument(state=current, entered_at=EPOCH, transition_id=0)
        with pytest.raises(IllegalTransition):
            store.transition(target)

    def test_no_state_transitions_to_itself(self):
        """A self-edge would let a caller 'confirm' a state and silently reset entered_at."""
        for state in NodeState:
            assert state not in LEGAL_TRANSITIONS[state]

    def test_failed_recovers_only_through_reconciling(self):
        assert LEGAL_TRANSITIONS[NodeState.FAILED] == frozenset({NodeState.RECONCILING})

    def test_a_stopped_cvm_leaves_the_node_switching_not_free(self):
        """Stopping the CVM ends TEARDOWN, not the crossing: the GPUs, the guest's memory and
        the forwarded ports come back to the host after the process does."""
        assert NodeState.SWITCHING in LEGAL_TRANSITIONS[NodeState.TEARDOWN]

    def test_switching_cannot_reach_launching(self):
        """FR-C6 enforced by the machine rather than by remembering to check it. A node whose
        last CVM's memory is still draining has no edge to LAUNCHING to take."""
        assert LEGAL_TRANSITIONS[NodeState.SWITCHING] == frozenset(
            {NodeState.RECONCILING, NodeState.FAILED}
        )

    def test_illegal_transition_names_what_was_allowed(self, state_dir):
        """The error has to be actionable — 'illegal transition' alone is not."""
        store = StateStore(state_dir)  # starts in RECONCILING; the self-edge is illegal
        with pytest.raises(IllegalTransition) as excinfo:
            store.transition(NodeState.RECONCILING)

        message = str(excinfo.value)
        assert "RECONCILING -> RECONCILING" in message
        assert "LAUNCHING" in message, "the error should list the edges that were allowed"


class TestPersistence:
    def test_round_trip(self, state_dir):
        store = StateStore(state_dir)
        _drive(store, [NodeState.LAUNCHING, NodeState.RENTER_RUNNING])

        reloaded = StateStore(state_dir)
        assert reloaded.state is NodeState.RENTER_RUNNING
        assert reloaded.document.transition_id == 2

    def test_fresh_dir_starts_reconciling(self, state_dir):
        store = StateStore(state_dir)
        assert store.state is NodeState.RECONCILING
        assert store.document.last_error is None

    def test_initial_document_is_persisted_immediately(self, state_dir):
        """A node that has never transitioned must still have a stable state document.

        Every freshly installed node is in exactly this position. Without the write, each restart
        mints a new `entered_at` and the task's `kill -9` verification fails on the most common
        path in the fleet.
        """
        before = StateStore(state_dir).document
        assert (state_dir / STATE_FILENAME).exists()

        assert StateStore(state_dir).document == before

    def test_document_is_stable_across_repeated_restarts(self, state_dir):
        first = StateStore(state_dir).document
        for _ in range(3):
            assert StateStore(state_dir).document == first

    def test_unreadable_state_file_fails_loud_not_silent_fresh(self, state_dir):
        """A corrupt state file must be visible on /v1/state, not look like a clean boot."""
        (state_dir / STATE_FILENAME).write_text("{ not json")

        store = StateStore(state_dir)
        assert store.state is NodeState.RECONCILING
        assert store.document.last_error is not None
        assert "unreadable or malformed" in store.document.last_error

    def test_unknown_state_name_is_treated_as_unreadable(self, state_dir):
        """A file from a future schema must not resolve to a state this build cannot handle."""
        (state_dir / STATE_FILENAME).write_text(
            '{"version": 1, "state": "WARP_DRIVE", "entered_at": "x", "transition_id": 1}'
        )
        store = StateStore(state_dir)
        assert store.state is NodeState.RECONCILING
        assert store.document.last_error is not None

    def test_leftover_tmp_file_does_not_break_reload(self, state_dir):
        """Atomic-write crash: a partial tmp file is left behind, the real file is still good."""
        store = StateStore(state_dir)
        store.transition(NodeState.LAUNCHING)
        (state_dir / f".{STATE_FILENAME}.partial.tmp").write_bytes(b'{"version": 1, "state": "FAI')

        assert StateStore(state_dir).state is NodeState.LAUNCHING

    def test_record_error_keeps_the_state(self, state_dir):
        store = StateStore(state_dir)
        store.transition(NodeState.LAUNCHING)
        store.record_error("vfio bind failed")

        reloaded = StateStore(state_dir)
        assert reloaded.state is NodeState.LAUNCHING
        assert reloaded.document.last_error == "vfio bind failed"


class TestStateEndpoint:
    def test_state_endpoint_reflects_a_transition(self, client, app, validator_key):
        from conftest import signed_request

        app.state.store.transition(NodeState.LAUNCHING)
        response = signed_request(client, validator_key, "GET", "/v1/state")

        assert response.status_code == 200
        assert response.json()["state"] == "LAUNCHING"

    def test_state_survives_a_restart(self, client, config, app, validator_key):
        from conftest import signed_request
        from cvmd.app import create_app
        from fastapi.testclient import TestClient

        app.state.store.transition(NodeState.LAUNCHING)
        app.state.store.transition(NodeState.VALIDATION_RUNNING)
        before = signed_request(client, validator_key, "GET", "/v1/state").json()

        with TestClient(create_app(config), raise_server_exceptions=False) as restarted:
            after = signed_request(restarted, validator_key, "GET", "/v1/state").json()

        assert after == before
