"""The registries: every wire type has exactly one model, dispatch is by `message_type`, and what is
not a message of this protocol is a ProtocolError — the error type both peers' loops already catch."""

import json

import pytest

from lium_protocol import (
    BACKEND_MESSAGES,
    VALIDATOR_MESSAGES,
    ProtocolError,
)
from lium_protocol.validator_to_backend import ContainerDeleted

A_DELETED = {
    "message_type": "ContainerDeleted",
    "miner_hotkey": "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
    "executor_id": "6f1d2c3b-4a5e-4f60-9b7c-8d9e0f1a2b3c",
    "pod_id": "0b1c2d3e-4f50-4617-8293-a4b5c6d7e8f9",
}


def test_parse_dispatches_on_message_type() -> None:
    message = VALIDATOR_MESSAGES.parse(json.dumps(A_DELETED))
    assert type(message) is ContainerDeleted
    assert message.workload_kind.value == "CUSTOMER_RENTAL"  # the default
    assert (message.sent_at, message.forwarded_at, message.queue_depth) == (None, None, None)


def test_a_refused_message_never_carries_the_input_into_the_error() -> None:
    """A BackupContainerRequest missing one field: the error names the field, not the credentials next to it."""
    message = {
        "message_type": "BackupContainerRequest",
        "miner_hotkey": "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
        "executor_id": "6f1d2c3b-4a5e-4f60-9b7c-8d9e0f1a2b3c",
        "pod_id": "0b1c2d3e-4f50-4617-8293-a4b5c6d7e8f9",
        "auth_token": "placeholder-jwt-that-must-not-leak",
        "repository_password": "placeholder-password-that-must-not-leak",
    }
    with pytest.raises(ProtocolError) as excinfo:
        BACKEND_MESSAGES.parse_obj(message)
    errors = json.loads(excinfo.value.msg)  # the error list itself, not a string of it
    assert {e["loc"][0] for e in errors} >= {"source_volume", "backup_volume_info", "backup_log_id"}
    assert "must-not-leak" not in excinfo.value.msg
    assert all("input" not in e and "url" not in e for e in errors)


