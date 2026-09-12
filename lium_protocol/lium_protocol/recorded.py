"""The recorded messages that ship with the protocol (`lium_protocol/recorded/*.json`): one per wire type
and one per HTTP body, as the peers' models serialise them today. This package's tests, the validator's
compatibility test (neurons/validators/tests/test_protocol_compat.py) and the backend's replay test read
them through `recorded()`; a consumer that vendors the package gets them with it."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

RECORDED_DIR = Path(__file__).parent / "recorded"


def recorded(direction: str) -> list[dict[str, Any]]:
    """[{"expect": <model name>, "message": <object>, "note": …}, …] for one direction:
    validator_to_backend · backend_to_validator · socket_replies · http."""
    document = json.loads((RECORDED_DIR / f"{direction}.json").read_text())
    assert document["direction"] == direction
    return document["messages"]


def json_keys(value: Any) -> Any:
    """The shape a message has after a JSON round trip (enums → values, tuples → lists), so two
    serialisations of the same message compare equal."""
    return json.loads(json.dumps(value, default=lambda o: getattr(o, "value", str(o))))
