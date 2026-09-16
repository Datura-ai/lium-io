"""Every recorded message (lium_protocol/recorded/) parses to the model it names, loses no field on the
way, and re-parses from its own serialisation. The recordings are the wire as each peer's current models
emit it; the validator's own copy of this test (neurons/validators/tests/test_protocol_compat.py) runs
the same files through the validator's models."""

import pytest

from lium_protocol import BACKEND_MESSAGES, VALIDATOR_MESSAGES, ValidatorMessageType
from lium_protocol.backend_to_validator import SOCKET_REPLIES
from lium_protocol.http import HTTP_MODELS
from lium_protocol.recorded import json_keys, recorded

REGISTRIES = {"validator_to_backend": VALIDATOR_MESSAGES, "backend_to_validator": BACKEND_MESSAGES}


def _ids(direction: str) -> list[str]:
    return [f"{entry['expect']}[{i}]" for i, entry in enumerate(recorded(direction))]


@pytest.mark.parametrize("direction", list(REGISTRIES))
def test_recordings_cover_every_wire_type(direction: str) -> None:
    covered = {entry["message"]["message_type"] for entry in recorded(direction)}
    assert covered == set(REGISTRIES[direction].models()), "a wire type without a recording"


def test_recordings_cover_every_http_body_and_socket_reply() -> None:
    assert {entry["expect"] for entry in recorded("http")} == set(HTTP_MODELS)
    assert {entry["expect"] for entry in recorded("socket_replies")} == set(SOCKET_REPLIES)


@pytest.mark.parametrize("entry", recorded("validator_to_backend"), ids=_ids("validator_to_backend"))
def test_validator_message_parses_and_keeps_every_field(entry: dict) -> None:
    _check_message(VALIDATOR_MESSAGES, entry)


@pytest.mark.parametrize("entry", recorded("backend_to_validator"), ids=_ids("backend_to_validator"))
def test_backend_message_parses_and_keeps_every_field(entry: dict) -> None:
    _check_message(BACKEND_MESSAGES, entry)


def _check_message(registry, entry: dict) -> None:
    model = registry.parse_obj(entry["message"])
    assert type(model).__name__ == entry["expect"]
    dumped = json_keys(model.model_dump(mode="json"))
    # every key the peer put on the wire is a field here — nothing is silently dropped
    missing = set(entry["message"]) - set(dumped)
    assert missing == set(), f"fields without a model field: {sorted(missing)}"
    # and the serialisation is the wire's own values for those keys
    for key, value in entry["message"].items():
        assert dumped[key] == _normalise(value), key
    assert registry.parse_obj(dumped) == model


@pytest.mark.parametrize(
    "entry",
    recorded("http") + recorded("socket_replies"),
    ids=_ids("http") + _ids("socket_replies"),
)
def test_body_parses_and_keeps_every_field(entry: dict) -> None:
    model_cls = {**HTTP_MODELS, **SOCKET_REPLIES}[entry["expect"]]
    model = model_cls.model_validate(entry["message"])
    dumped = json_keys(model.model_dump(mode="json"))
    if isinstance(entry["message"], list):  # a RootModel body: every item, every key
        assert len(dumped) == len(entry["message"])
        for item, seen in zip(entry["message"], dumped, strict=True):
            assert set(item) - set(seen) == set()
        return
    missing = set(entry["message"]) - set(dumped)
    assert missing == set(), f"fields without a model field: {sorted(missing)}"
    assert model_cls.model_validate(dumped) == model


def _normalise(value):
    """Datetimes re-render with the same instant but pydantic's spelling of the offset; compare those by
    value. Everything else must be byte-for-byte what was recorded."""
    if isinstance(value, str) and value.endswith("Z") and "T" in value:
        from datetime import datetime

        try:
            instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return value
        return instant.isoformat().replace("+00:00", "Z")
    return json_keys(value)


def test_failed_request_codes_the_backend_copy_lacks_are_on_the_wire() -> None:
    """The recording carries `AttestationError` / `ContainerStopFailed`: the validator emits them
    (payload_models), the backend's current enum has neither — the drift this package makes visible."""
    entries = [e for e in recorded("validator_to_backend") if e["expect"] == "FailedContainerRequest"]
    codes = {e["message"].get("error_code") for e in entries}
    types = {e["message"]["error_type"] for e in entries}
    assert "AttestationError" in codes and "ContainerStopFailed" in types
    assert ValidatorMessageType.FailedRequest.value == "FailedRequest"
