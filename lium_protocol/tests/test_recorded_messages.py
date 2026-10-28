"""Every recorded message (lium_protocol/recorded/) parses to the model it names, loses no field on the
way, and re-parses from its own serialisation. The recordings are the wire as each peer's current models
emit it; the validator's own copy of this test (neurons/validators/tests/test_protocol_compat.py) runs
the same files through the validator's models."""

import pytest

from lium_protocol import BACKEND_MESSAGES, VALIDATOR_MESSAGES
from lium_protocol.recorded import json_keys, recorded


def _ids(direction: str) -> list[str]:
    return [f"{entry['expect']}[{i}]" for i, entry in enumerate(recorded(direction))]


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


@pytest.mark.parametrize("entry", recorded("backend_to_validator"), ids=_ids("backend_to_validator"))
def test_command_id_is_optional_and_ignored_by_old_message_model(entry: dict) -> None:
    from lium_protocol.backend_to_validator import ServerRequest
    from lium_protocol.base import Message

    model = BACKEND_MESSAGES.parse_obj(entry["message"])
    if isinstance(model, ServerRequest):
        assert model.command_id == ""
        payload = {**entry["message"], "command_id": "retry"}
        assert BACKEND_MESSAGES.parse_obj(payload).command_id == "retry"
        assert "command_id" not in Message.model_validate(model.model_dump()).model_dump()
    else:
        assert "command_id" not in type(model).model_fields


@pytest.mark.parametrize("message_type", ["SshPubKeyAdded", "FailedRequest"])
def test_ssh_key_answer_command_id_is_optional(message_type: str) -> None:
    message = {"message_type": message_type, "miner_hotkey": "m", "executor_id": "e", "pod_id": "p", "msg": "x"}

    assert VALIDATOR_MESSAGES.parse_obj(message).command_id == ""
    assert VALIDATOR_MESSAGES.parse_obj({**message, "command_id": "restore:1"}).command_id == "restore:1"
