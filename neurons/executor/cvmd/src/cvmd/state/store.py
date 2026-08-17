"""Persistence for the node state — written atomically on every edge, reloaded on daemon start."""

import logging
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from cvmd.atomic import read_json, write_json_durable
from cvmd.state.machine import IllegalTransition, NodeState, is_legal

logger = logging.getLogger(__name__)

STATE_FILENAME = "state.json"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class StateDocument:
    state: NodeState
    entered_at: str
    transition_id: int
    last_error: str | None = None

    def to_json(self) -> dict:
        return {
            "version": SCHEMA_VERSION,
            "state": str(self.state),
            "entered_at": self.entered_at,
            "transition_id": self.transition_id,
            "last_error": self.last_error,
        }


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def initial_document(last_error: str | None = None) -> StateDocument:
    return StateDocument(
        state=NodeState.RECONCILING,
        entered_at=_now_iso(),
        transition_id=0,
        last_error=last_error,
    )


def _decode(raw) -> StateDocument | None:
    if not isinstance(raw, dict) or raw.get("version") != SCHEMA_VERSION:
        return None
    try:
        state = NodeState(raw["state"])
        return StateDocument(
            state=state,
            entered_at=str(raw["entered_at"]),
            transition_id=int(raw["transition_id"]),
            last_error=raw.get("last_error"),
        )
    except (KeyError, TypeError, ValueError):
        return None


class StateStore:
    """Holds the current node state and keeps `state.json` in step with it.

    A state file that cannot be read does not silently become a fresh one: the daemon starts in
    RECONCILING with the load failure recorded in `last_error`, so the condition is visible on
    `/v1/state` instead of looking like a clean boot.
    """

    def __init__(self, state_dir: Path) -> None:
        self._path = state_dir / STATE_FILENAME
        self._document, synthesized = self._load()

        # A synthesized document is written out immediately. Without this, a node that has not
        # transitioned yet — which is every freshly installed node — has no state file, so each
        # restart mints a new `entered_at` and `/v1/state` reports a different document after a
        # kill -9. The state has to be stable from first boot, not from first transition.
        if synthesized:
            self.persist()

    @property
    def document(self) -> StateDocument:
        return self._document

    @property
    def state(self) -> NodeState:
        return self._document.state

    def _load(self) -> tuple[StateDocument, bool]:
        """Return the document and whether it had to be synthesized rather than read."""
        if not self._path.exists():
            logger.info("no state file at %s — starting in %s", self._path, NodeState.RECONCILING)
            return initial_document(), True

        decoded = _decode(read_json(self._path))
        if decoded is None:
            message = (
                f"state file at {self._path} was unreadable or malformed; reset to RECONCILING"
            )
            logger.error(message)
            return initial_document(last_error=message), True
        return decoded, False

    def transition(self, requested: NodeState, *, last_error: str | None = None) -> StateDocument:
        """Move to `requested`, persisting before returning. Raises IllegalTransition on a bad edge."""
        current = self._document.state
        if not is_legal(current, requested):
            raise IllegalTransition(current, requested)

        self._document = StateDocument(
            state=requested,
            entered_at=_now_iso(),
            transition_id=self._document.transition_id + 1,
            last_error=last_error,
        )
        self.persist()
        return self._document

    def record_error(self, message: str) -> StateDocument:
        """Attach an error to the current state without moving. Used for non-fatal faults."""
        self._document = replace(self._document, last_error=message)
        self.persist()
        return self._document

    def persist(self) -> None:
        write_json_durable(self._path, self._document.to_json())
